/**
 * Wrap a model so every generate goes through the BYOK concurrency limiter
 * (process-wide rate limit) AND reports BYOK failures (drives the
 * `byok.llm_errors_threshold` notification). Inside `LLM.run` it also arms the
 * failover round-trip deadline on each provider request, and reports each
 * provider answer to the circuit breaker.
 *
 * Done at the MODEL level (AI SDK `wrapLanguageModel`) so any agent runner stays
 * model-agnostic — the failure reporter is injected directly (no AsyncLocalStorage).
 */
import { wrapLanguageModel, type LanguageModel } from 'ai';

// The wrapped model and its generate-result type, derived FROM `wrapLanguageModel`
// so they track whichever AI SDK version is installed (no hand-imported V-suffixed
// names to drift). `WrappedModel` is `LanguageModelV*` — a subtype of the public
// `LanguageModel`, so callers are unaffected while `.doGenerate` stays visible.
type WrappedModel = ReturnType<typeof wrapLanguageModel>;
type ByokGenerateResult = Awaited<ReturnType<WrappedModel['doGenerate']>>;

import type { NormalizedModel } from '@libs/llm/byok-config';
import {
    runWithBYOKLimiter,
    getLimiterForSlot,
} from '@libs/llm/byok-to-vercel';
import { estimateTextTokens } from '@libs/llm/token-estimate';
import {
    attachClassification,
    classifyLLMError,
    LlmErrorCategory,
    type ClassifiedErrorInfo,
} from '@libs/llm/error-classifier';
import { extractProviderMessage } from '@libs/llm/review-error-diagnostics';
import {
    isProviderDeadlineError,
    ProviderDeadlineError,
    ROUND_TRIP_SETTLE_GRACE_MS,
    ROUND_TRIP_SLOT_HOLD_CAP_MS,
    type ProviderDeadlineSource,
} from '@libs/llm/provider-deadline-error';
import { createLogger } from '@libs/core/log/logger';
import {
    MAX_REFUSAL_RETRIES,
    rejectedReasoning,
    rememberRefusal,
    withoutReasoningField,
    withoutRefusedFields,
    withReasoningOff,
} from '@libs/llm/rejected-reasoning-field';

const logger = createLogger('ByokModelWrapper');

/**
 * PRE-call token estimate for the tpm reservoir. Serializes the wire prompt the
 * same way the provider sees it, then counts with the SHARED tiktoken estimator
 * (`estimateTextTokens`) — NOT a char/4 heuristic (flat-4 under-counts dense
 * code ~1.6×). This is the ONE seam with `params.prompt`.
 */
function estimatePromptTokens(prompt: unknown): number {
    if (prompt == null) return 0;
    const text =
        typeof prompt === 'string' ? prompt : safeSerialize(prompt);
    return estimateTextTokens(text);
}

function safeSerialize(value: unknown): string {
    try {
        return JSON.stringify(value) ?? '';
    } catch {
        return String(value);
    }
}

/**
 * POST-call real token total for reconcile. Reads `doGenerate().usage` — the
 * only seam with the actual usage — preferring `totalTokens`, falling back to
 * `inputTokens + outputTokens`. Returns undefined when usage is absent so the
 * reservoir leaves the pre-call estimate standing as the net debit.
 */
function extractUsageTotal(result: unknown): number | undefined {
    const usage = (result as { usage?: Record<string, unknown> } | null)
        ?.usage;
    if (!usage) return undefined;
    if (typeof usage.totalTokens === 'number') return usage.totalTokens;
    const input =
        typeof usage.inputTokens === 'number' ? usage.inputTokens : 0;
    const output =
        typeof usage.outputTokens === 'number' ? usage.outputTokens : 0;
    const sum = input + output;
    return sum > 0 ? sum : undefined;
}

/**
 * One failover attempt's round-trip deadline and answer hook, built per attempt
 * by `LLM.run`. Absent everywhere outside `LLM.run`, which means no deadline.
 */
export interface ByokRoundTrip {
    /** The deadline for one provider request, or undefined when none is
     *  configured. */
    ms?: number;
    /** `round-trip` at the full knob value; `caller-budget` when a one-shot
     *  caller's shorter `timeoutMs` set it. Only the first trips the breaker. */
    source: ProviderDeadlineSource;
    /** Read when each request is armed: true only while a cascade is possible
     *  (a next chain entry exists and no loop step has run). */
    isActive: () => boolean;
    /** Called on each provider answer, deadline or not: a half-open breaker
     *  probe resolves on the first one. */
    onAnswered?: () => void;
}

export interface WrapByokModelOptions {
    byokConfig?: NormalizedModel;
    organizationId?: string;
    provider?: string;
    /** @deprecated No-op since 04b-02 — the limiter keys off the single resolved
     *  slot, not a `main`/`fallback`/`internal` role. Kept on
     *  the type so existing callers passing `role: 'main'` still compile; remove
     *  in a later cleanup wave. */
    role?: 'main' | 'fallback' | 'internal';
    queueTimeoutMs?: number;
    roundTrip?: ByokRoundTrip;
    reporter?: (input: {
        organizationId?: string;
        provider: string;
        /** The provider's own sentence when it gave one, else `err.message`. */
        errorMessage: string;
        category?: LlmErrorCategory;
        httpStatus?: number;
        routingRefusal?: boolean;
        /** Our round-trip deadline killed the request. It is only armed when a
         *  next chain entry exists, so the call moved on to that entry. */
        deadline?: boolean;
    }) => void;
}

/** The armed deadline for one request: its signal, and whether it fired
 *  before the caller's own. */
interface ArmedDeadline {
    signal: AbortSignal;
    firedFirst: () => boolean;
    error: ProviderDeadlineError;
    disarm: () => void;
}

/** The caller's cancel, as the SDK would have thrown it. */
function callerAbortReason(signal: AbortSignal | undefined): unknown {
    return (
        signal?.reason ??
        new DOMException('This operation was aborted', 'AbortError')
    );
}

function armDeadline(
    ms: number,
    source: ProviderDeadlineSource,
    callerSignal: AbortSignal | undefined,
    target: { provider?: string; model?: string },
): ArmedDeadline {
    const controller = new AbortController();
    let first: 'deadline' | 'caller' | undefined = callerSignal?.aborted
        ? 'caller'
        : undefined;
    const onCaller = () => {
        first ??= 'caller';
    };
    callerSignal?.addEventListener('abort', onCaller, { once: true });
    const timer = setTimeout(() => {
        first ??= 'deadline';
        controller.abort();
    }, ms);
    // An undefined caller signal is dropped, not handed to AbortSignal.any.
    const signals = [callerSignal, controller.signal].filter(
        (s): s is AbortSignal => !!s,
    );
    return {
        signal: signals.length > 1 ? AbortSignal.any(signals) : signals[0],
        firedFirst: () => first === 'deadline',
        // No `cause`: the provider's AbortError on it would read as a caller
        // cancel to the failover gate.
        error: new ProviderDeadlineError({
            deadlineMs: ms,
            source,
            phase: 'round-trip',
            provider: target.provider,
            model: target.model,
        }),
        disarm: () => {
            clearTimeout(timer);
            callerSignal?.removeEventListener('abort', onCaller);
        },
    };
}

export function wrapByokModel(
    model: LanguageModel,
    opts: WrapByokModelOptions,
): WrappedModel {
    return wrapLanguageModel({
        model: model as any,
        middleware: {
            specificationVersion: 'v3',
            wrapGenerate: async ({ doGenerate, params, model: inner }: any) => {
                const slot = opts.byokConfig;
                // Settles the CALLER when a provider ignores the deadline's
                // abort; the limiter slot stays held meanwhile (see `call`).
                let settleCaller: ((err: unknown) => void) | undefined;
                const callerSettled = new Promise<never>((_, reject) => {
                    settleCaller = reject;
                });
                const send = (p: any) =>
                    p === params ? doGenerate() : inner.doGenerate(p);
                // One provider request. The round-trip deadline is armed here,
                // per request: after limiter admission (queue wait never
                // counts), anew for each refusal re-ask, and only while a
                // cascade is possible.
                const call = async (p: any) => {
                    const roundTrip = opts.roundTrip;
                    const ms = roundTrip?.ms;
                    if (!roundTrip || !ms || ms <= 0 || !roundTrip.isActive()) {
                        const answer = await send(p);
                        roundTrip?.onAnswered?.();
                        return answer;
                    }
                    const deadline = armDeadline(
                        ms,
                        roundTrip.source,
                        p?.abortSignal,
                        {
                            provider: opts.provider ?? slot?.provider,
                            model: slot?.model,
                        },
                    );
                    const pending = send({
                        ...p,
                        abortSignal: deadline.signal,
                    });
                    // A proxy that ignores the abort never settles `pending`.
                    // The caller is released after a grace, but this request
                    // (and so the limiter slot) stays open until `pending`
                    // settles, capped: the limiter's count keeps matching the
                    // sockets actually open, as agent-loop-call explains.
                    let grace: ReturnType<typeof setTimeout> | undefined;
                    let cap: ReturnType<typeof setTimeout> | undefined;
                    const capped = new Promise<never>((_, reject) => {
                        grace = setTimeout(() => {
                            if (deadline.firedFirst()) {
                                settleCaller?.(deadline.error);
                            }
                        }, ms + ROUND_TRIP_SETTLE_GRACE_MS);
                        // By the cap one of the two signals has fired: the
                        // request ends as whichever did.
                        cap = setTimeout(
                            () =>
                                reject(
                                    deadline.firedFirst()
                                        ? deadline.error
                                        : callerAbortReason(p?.abortSignal),
                                ),
                            ms + ROUND_TRIP_SLOT_HOLD_CAP_MS,
                        );
                    });
                    try {
                        const answer = await Promise.race([pending, capped]);
                        roundTrip.onAnswered?.();
                        return answer;
                    } catch (err) {
                        // Whatever the provider threw once the deadline fired
                        // first (its AbortError, a wrapped network error) is
                        // our deadline; a caller abort stays the caller's.
                        throw deadline.firedFirst() ? deadline.error : err;
                    } finally {
                        clearTimeout(grace);
                        clearTimeout(cap);
                        deadline.disarm();
                    }
                };
                const generate = async () => {
                    const sent = withoutRefusedFields(
                        params,
                        slot,
                        opts.organizationId,
                    );
                    // A strict upstream refuses one unknown field per answer
                    // (Go's decoder stops at the first), so each reasoning field
                    // we added can cost one more ask, never more than that: a
                    // retry needs a field still present to remove, and the count
                    // is capped besides.
                    let attempt = sent;
                    for (let retries = 0; ; retries++) {
                        try {
                            return await call(attempt);
                        } catch (err) {
                            const refusal = rejectedReasoning(err);
                            const retry =
                                refusal && retries < MAX_REFUSAL_RETRIES
                                    ? refusal.off
                                        ? withReasoningOff(attempt, refusal.field)
                                        : withoutReasoningField(
                                              attempt,
                                              refusal.field,
                                          )
                                    : undefined;
                            if (!refusal || !retry) throw err;
                            const { field, off } = refusal;
                            rememberRefusal(
                                slot,
                                opts.organizationId,
                                field,
                                Date.now(),
                                off,
                            );
                            logger.warn({
                                message: off
                                    ? `Upstream refused \`${field}\` with tools; asking again with it off`
                                    : `Upstream refused \`${field}\`; asking again without it`,
                                context: 'ByokModelWrapper',
                                metadata: {
                                    field,
                                    provider: slot?.provider,
                                    baseURL: slot?.baseURL,
                                    model: slot?.model,
                                    organizationId: opts.organizationId,
                                },
                            });
                            attempt = retry;
                        }
                    }
                };
                const run = async () => {
                    try {
                        return await generate();
                    } catch (err) {
                        // Classify (so downstream can read the canonical category)
                        // and report — never let the reporter mask the LLM error.
                        let classified: ClassifiedErrorInfo | undefined;
                        if (err && typeof err === 'object') {
                            classified = classifyLLMError(
                                err,
                                opts.provider,
                            );
                            attachClassification(err, classified);

                            // Arm the slot's cooldown ONLY on a classified
                            // RATE_LIMIT (429-rate) when the slot opted in via
                            // cooldownMs. A QUOTA_EXCEEDED (429-billing) and a
                            // TRANSIENT (5xx/network) NEVER arm. Arming is a
                            // DELAY, not a retry: the limiter holds the next
                            // admission; the reporter and rethrow below are
                            // untouched. Reuses the classify already computed in
                            // this catch — no re-classification.
                            const cooldownSlot = opts.byokConfig;
                            const cooldownMs = cooldownSlot?.cooldownMs;
                            if (
                                classified.category ===
                                    LlmErrorCategory.RATE_LIMIT &&
                                !!cooldownMs &&
                                cooldownMs > 0
                            ) {
                                getLimiterForSlot({
                                    slot: cooldownSlot,
                                    organizationId: opts.organizationId,
                                })?.armCooldown(cooldownMs);
                            }
                        }
                        try {
                            opts.reporter?.({
                                organizationId: opts.organizationId,
                                provider:
                                    opts.provider ??
                                    opts.byokConfig?.provider ??
                                    'unknown',
                                // For an AI SDK `APICallError` the message
                                // only restates the status ("Provider returned
                                // error"); the cause is in the body (#1871).
                                errorMessage:
                                    extractProviderMessage(err) ??
                                    (err instanceof Error
                                        ? err.message
                                        : String(err ?? 'unknown')),
                                category: classified?.category,
                                httpStatus: classified?.httpStatus,
                                routingRefusal: classified?.routingRefusal,
                                ...(isProviderDeadlineError(err)
                                    ? { deadline: true }
                                    : {}),
                            });
                        } catch {
                            /* reporter failures must not surface */
                        }
                        throw err;
                    }
                };

                // The limiter keys off the ONE resolved slot the org configured
                // for this task (`slot`, above).

                // tpm reservoir (hybrid): this wrapper is the ONE seam with BOTH
                // the pre-call prompt AND the post-call usage. Estimate the
                // prompt tokens to DEBIT the reservoir at admission, and hand the
                // usage extractor so the limiter RECONCILES estimate vs actual
                // after the call. Only when the slot carries tpm — otherwise zero
                // estimation overhead and the exact 05-01 path (rpm/concurrency).
                const hasTpm = !!slot?.tpm && slot.tpm > 0;
                const estimatedTokens = hasTpm
                    ? estimatePromptTokens(params?.prompt)
                    : undefined;

                // Pin T to the SDK's generate-result type so inference doesn't
                // collapse to `unknown` (the middleware return type would then
                // reject it). `run` yields exactly this, and `extractUsageTotal`
                // takes `unknown`, so nothing else constrains T.
                const limited = runWithBYOKLimiter<ByokGenerateResult>(
                    {
                        slot,
                        organizationId: opts.organizationId,
                        abortSignal: params?.abortSignal,
                        queueTimeoutMs: opts.queueTimeoutMs,
                        estimatedTokens,
                        getUsageTokens: hasTpm ? extractUsageTotal : undefined,
                    },
                    run,
                    'llm-call',
                );
                // Without a deadline nothing settles the caller early, so the
                // call is exactly what it was.
                return opts.roundTrip?.ms
                    ? Promise.race([limited, callerSettled])
                    : limited;
            },
        },
    });
}
