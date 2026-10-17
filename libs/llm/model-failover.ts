/**
 * Runtime model FAILOVER — cascade `primary → fallbacks[0] → …` when a call fails.
 *
 * The routing layer (`resolveTaskSlot`) already decides WHICH model runs a task
 * and stamps the org's configured fallback chain onto `slot.fallbacks`. This
 * owns the OTHER half: when the model that ran actually FAILS at call time in a
 * way a different model can fix, re-run the call on the next chain entry. It is
 * the OUTERMOST retry — it wraps the executor, which already owns the same-model
 * recovery (SDK per-step retry, the one-shot D-00c re-issue, the limiter's
 * cooldown). So a model is only abandoned after it has exhausted its OWN retries.
 *
 * The chain is the ordered list the org configured (`routing.fallbackModelIds`,
 * at most MAX_FALLBACKS entries after the primary). Each entry is a deliberate
 * backup (often a cheaper/steadier model), not "the next tier" — the selection
 * precedence (per-agent → default) is a SEPARATE concern owned by the router,
 * not re-walked here. The cascade stops at the first success, at an error that
 * must not cascade, once an attempt marked itself unsafe to restart, or when
 * the caller's signal has aborted. Between calls, the circuit breaker
 * (`provider-circuit.ts`) lets the cascade skip an entry whose model keeps
 * failing, so the next call does not pay to learn the outage again.
 */
import { AsyncLocalStorage } from 'node:async_hooks';

import { createLogger } from '@libs/core/log/logger';
import { LLM_ERROR_TAG, LLM_SUCCESS_TAG } from '@libs/llm/log-tags';
import type { NormalizedModel } from '@libs/llm/byok-config';
import { getLimiterForSlot } from '@libs/llm/byok-limiter';
import { LLM_CALL_TIMEOUT_MS } from '@libs/llm/llm-call';
import {
    circuitCountsFailure,
    circuitKeyFor,
    circuitLabelsFor,
    LLM_CIRCUIT_TAG,
    providerCircuit,
    retryAfterMsOf,
    type CircuitAdmission,
    type CircuitState,
} from '@libs/llm/provider-circuit';
import {
    failoverRoundTripDeadlineMs,
    isProviderDeadlineError,
    MAX_ROUND_TRIP_DEADLINE_MS,
} from '@libs/llm/provider-deadline-error';
import { REGISTRY } from '@libs/llm/providers';
import {
    classifyLLMError,
    isAbortOrHardTimeout,
    isCallerAbortOrBudget,
    isProviderTimeout,
    isTerminalCategory,
    LlmErrorCategory,
    retriesWereExhausted,
} from '@libs/llm/error-classifier';

const logger = createLogger('ModelFailover');

/** Why a chain entry did not answer: the classified category, or the
 *  timeout / breaker reason that decided the hop. */
export type FailoverHopReason =
    LlmErrorCategory | 'provider-timeout' | 'deadline' | 'circuit-open';

/** One chain entry that was tried (or skipped) without answering. */
export interface FailoverHop {
    model: string;
    /** Position in the runtime chain: 0 = the primary. */
    chainIndex: number;
    outcome: 'failed' | 'skipped';
    reason: FailoverHopReason;
    elapsedMs: number;
}

/** A call that failed on one model and was answered by a later one. */
export interface ModelFailoverEvent {
    runName: string;
    /** The primary — the model the call was routed to. */
    failedModel: string;
    /** The chain entry that answered. */
    usedModel: string;
    /** Position of `usedModel` in the runtime chain (1 = first fallback). */
    chainIndex: number;
    /** Every entry before it, in order, and why each did not answer. */
    hops: FailoverHop[];
}

const failoverRecorder = new AsyncLocalStorage<ModelFailoverEvent[]>();

/**
 * Runs `fn` and returns every call inside it that only succeeded on a fallback
 * model. A failover is otherwise invisible to the caller: the call returns the
 * same value it would have on the primary, so a review that ran on the backup
 * model reads exactly like one that did not.
 */
export async function recordModelFailovers<T>(
    fn: () => Promise<T>,
): Promise<{ value: T; failovers: ModelFailoverEvent[] }> {
    const failovers: ModelFailoverEvent[] = [];
    const value = await failoverRecorder.run(failovers, fn);
    return { value, failovers };
}

/**
 * Should a failed call cascade to the FALLBACK model? True only when a DIFFERENT
 * model can plausibly fix it — the whole point of a per-org fallback:
 *  - TERMINAL model-specific failures (bad/expired key, out of credit, unknown
 *    model, access denied): the primary provably can't serve this org/task.
 *  - A TRANSIENT (5xx / network) blip that OUTLIVED the executor's same-model
 *    retries — persistent enough that the provider itself looks down.
 *
 * Deliberately NOT cascaded (a peer model doesn't help, or another layer owns it):
 *  - RATE_LIMIT that the limiter can still wait out — it owns backoff/cooldown, and
 *    swapping models on a 429 would defeat the rate gate. A 429 that already
 *    EXHAUSTED the executor's same-model retries DOES cascade: backoff is the cure
 *    for a real rate limit, so outliving it disproves the label (see
 *    `retriesWereExhausted`).
 *  - CONTEXT_OVERFLOW — the prompt is too big; a same-class fallback won't fit it.
 *  - A caller abort or a Kodus budget (hard-timeout race, limiter queue
 *    timeout) — the failure is a cancel or a spent budget, not the model; the
 *    next model would inherit the same spent budget.
 *  - UNKNOWN — unclassified; stay conservative rather than spend a 2nd billed call.
 *
 * A PROVIDER timeout (504/408/524, an undici timeout, our round-trip deadline)
 * is the opposite case and does cascade: the route is down, not the budget.
 * `LLM_PROVIDER_TIMEOUT_CASCADE=off` restores the old prose gate, which could
 * not tell the two apart.
 */
export function shouldFailoverToNextModel(err: unknown): boolean {
    if (providerTimeoutCascadeEnabled()) {
        // The caller check runs first: a cancel wrapped around a provider
        // timeout is still a cancel.
        if (isCallerAbortOrBudget(err)) {
            return false;
        }
        if (isProviderTimeout(err)) {
            return true;
        }
    } else if (isAbortOrHardTimeout(err)) {
        return false;
    }
    const { category } = classifyLLMError(err);
    switch (category) {
        case LlmErrorCategory.RATE_LIMIT:
            // A 429 the limiter can wait out must NOT swap models — that would
            // defeat the rate gate and hammer providers. But a 429 that already
            // OUTLIVED the executor's own exponential backoff is not a queue:
            // backoff is the remedy for a genuine rate limit, so surviving it
            // disproves the reading. It is the same test this function already
            // applies to TRANSIENT ("persistent enough that the provider itself
            // looks down"), owed to 429 for the same reason.
            //
            // This is what makes the cascade independent of vendor prose. Z.AI
            // words a spent balance as "Insufficient balance … Please recharge"
            // and a 429; matching phrases missed it and the org's fallback sat
            // idle through the outage. Exhaustion would have caught it without
            // knowing a single word of it — and catches the next vendor too.
            //
            // The per-slot limiter still holds: the cooldown armed on the
            // primary keeps protecting it while the FALLBACK (a different slot,
            // a different limiter) serves. The gate is delayed, not defeated.
            return retriesWereExhausted(err);
        // CONTEXT_OVERFLOW: the prompt is too big, and a same-class fallback
        // will not fit it either.
        //
        // UNKNOWN stays conservative even when retries were exhausted: exhaustion
        // says the failure is PERSISTENT, not what it is, so a 2nd billed call
        // would be a guess. Widening it is a separate decision from this one.
        case LlmErrorCategory.CONTEXT_OVERFLOW:
        case LlmErrorCategory.UNKNOWN:
            return false;
        case LlmErrorCategory.TRANSIENT:
            return true;
        default:
            return isTerminalCategory(category);
    }
}

/**
 * Kill switch for the provider-timeout cascade. On by default (it is a bug
 * fix); `LLM_PROVIDER_TIMEOUT_CASCADE=off` puts back the prose gate
 * (`isAbortOrHardTimeout`) both here and in the one-shot D-00c re-issue. Read
 * per call, so no module state caches a stale value.
 */
export function providerTimeoutCascadeEnabled(
    env: Readonly<Record<string, string | undefined>> = process.env,
): boolean {
    const raw = env.LLM_PROVIDER_TIMEOUT_CASCADE?.trim().toLowerCase();
    return !(raw === 'off' || raw === 'false' || raw === '0');
}

/**
 * The round-trip deadline for one chain entry: the env knob, raised to the
 * provider's declared minimum (a cold-start provider declares one) and kept
 * under the clamp. Undefined when the knob is off.
 */
export function roundTripDeadlineFor(
    slot: NormalizedModel | undefined,
): number | undefined {
    const knob = failoverRoundTripDeadlineMs();
    if (knob === undefined) return undefined;
    const floor =
        slot && REGISTRY.has(slot.provider)
            ? REGISTRY.get(slot.provider).minRoundTripDeadlineMs
            : undefined;
    return Math.min(Math.max(knob, floor ?? 0), MAX_ROUND_TRIP_DEADLINE_MS);
}

/** A probe lease outlives the probe's own deadline by this margin. */
const PROBE_LEASE_MARGIN_MS = 10_000;

/** Handed to each attempt so it can veto its own retry. */
export interface FailoverAttemptControl {
    /** True once this attempt called `markUnsafeToRetry`. */
    isUnsafe(): boolean;
    /**
     * True when a later chain entry will run if this attempt fails in a
     * cascade-worthy way. The one-shot D-00c re-issue reads it: a provider
     * timeout is thrown to the next entry instead of being re-issued on the
     * same stalled route.
     */
    readonly hasNext: boolean;
    /**
     * Mark THIS attempt as no longer safely restartable — it has emitted output a
     * fresh run would DUPLICATE (an agent-loop step already ran + mutated shared
     * runner state). After this is called, a cascade-worthy error still propagates
     * instead of retrying on the fallback. One-shot calls are atomic (nothing is
     * committed until they return) and never call this, so they always fail over.
     */
    markUnsafeToRetry(): void;
    /**
     * The provider answered a request of this attempt. Wired to the BYOK
     * wrapper's answer hook: a half-open circuit probe resolves here, on the
     * first answer, not when a long loop finally ends. Idempotent.
     */
    onAnswered(): void;
    /** Position of this attempt in the runtime chain: 0 = the primary. */
    readonly chainIndex: number;
    /** The breaker state this attempt was admitted under; undefined when no
     *  breaker applies (no org, the managed default, mode off). */
    readonly circuitState?: CircuitState;
    /** Why the previous chain entry did not answer; undefined on the primary. */
    readonly failoverReason?: FailoverHopReason;
}

/** Distinct, order-preserving attempts. Drops a chain entry that resolves to the
 *  same model as an earlier one; the managed-default attempt (`undefined`) is only
 *  ever the sole entry — never queued as a fallback after a real slot. */
function distinctAttempts(
    slots: Array<NormalizedModel | undefined>,
): Array<NormalizedModel | undefined> {
    const out: Array<NormalizedModel | undefined> = [];
    const seen = new Set<string>();
    for (const slot of slots) {
        if (!slot) {
            if (out.length === 0) {
                out.push(slot);
            }
            continue;
        }
        const key = slot.byokModelId ?? slot.model;
        if (seen.has(key)) {
            continue;
        }
        seen.add(key);
        out.push(slot);
    }
    return out.length > 0 ? out : [undefined];
}

const ATTEMPTED_SLOT = Symbol('attemptedSlot');
const FAILOVER_HOPS = Symbol('failoverHops');

/**
 * Which model actually ran on the attempt that failed.
 *
 * The cascade walks the chain, so a terminal failure belongs to whichever
 * attempt was LAST — not to the slot resolved before the run. A caller that
 * reports the resolved slot after a fallback also failed names a model/provider
 * pair that never co-occurred, and does it in exactly the case the report exists
 * for: both routes down.
 *
 * Stamped here because this is the only place that knows. Non-enumerable, like
 * `attachClassification`, so it stays out of JSON.stringify and span recordings.
 */
export function attachAttemptedSlot<T extends object>(
    err: T,
    slot: NormalizedModel | undefined,
): T {
    if (!slot) return err;
    try {
        Object.defineProperty(err, ATTEMPTED_SLOT, {
            value: { model: slot.model, provider: slot.provider },
            enumerable: false,
            writable: false,
            configurable: true,
        });
    } catch (error) {
        // Frozen or exotic error object. The caller degrades to the pre-run
        // resolved slot, which is what it did before this existed — but that is
        // the wrong-model report this stamp exists to prevent, so a silent
        // failure here would make a WRONG diagnostic untraceable. Debug, not
        // warn: it changes nothing a user sees and the fallback is correct.
        logger.debug({
            message: `${LLM_ERROR_TAG} could not stamp the attempted slot on a failed call`,
            context: 'attachAttemptedSlot',
            metadata: {
                slotModel: slot.model,
                slotProvider: slot.provider,
                reason: error instanceof Error ? error.message : String(error),
            },
        });
    }
    return err;
}

/** The stamped attempt, when the error came through the failover. */
export function readAttemptedSlot(
    err: unknown,
): { model?: string; provider?: string } | undefined {
    if (!err || typeof err !== 'object') return undefined;
    return (err as Record<symbol, unknown>)[ATTEMPTED_SLOT] as
        { model?: string; provider?: string } | undefined;
}

/** The hops a call made before it failed for good, when it came through the
 *  failover (the last one is the failure that was thrown). Non-enumerable on
 *  the error, like the attempted slot. */
export function readFailoverHops(err: unknown): FailoverHop[] | undefined {
    if (!err || typeof err !== 'object') return undefined;
    return (err as Record<symbol, unknown>)[FAILOVER_HOPS] as
        FailoverHop[] | undefined;
}

function attachFailoverHops<T extends object>(err: T, hops: FailoverHop[]): T {
    try {
        Object.defineProperty(err, FAILOVER_HOPS, {
            value: hops,
            enumerable: false,
            writable: false,
            configurable: true,
        });
    } catch {
        // Frozen or exotic error object: the terminal warn already lists the
        // hops, which is the record that matters.
    }
    return err;
}

/** The reason a hop is recorded under: our deadline and a provider timeout
 *  are named as such (they classify TRANSIENT), anything else by category. */
function hopReason(
    err: unknown,
    category: LlmErrorCategory,
): FailoverHopReason {
    if (isProviderDeadlineError(err)) {
        return 'deadline';
    }
    if (providerTimeoutCascadeEnabled() && isProviderTimeout(err)) {
        return 'provider-timeout';
    }
    return category;
}

/** Ask the circuit breaker about one chain entry. Undefined when no breaker
 *  applies: mode off, the managed default, a BYOK slot without an org. */
function admitAttempt(
    slot: NormalizedModel | undefined,
    isLast: boolean,
    opts: { organizationId?: string },
): CircuitAdmission | undefined {
    const key = circuitKeyFor(slot, opts.organizationId);
    if (!key || providerCircuit.currentMode() === 'off') return undefined;
    return providerCircuit.admit(key, {
        isLast,
        labels: circuitLabelsFor(slot, opts.organizationId),
        probeLeaseMs:
            (roundTripDeadlineFor(slot) ?? LLM_CALL_TIMEOUT_MS) +
            PROBE_LEASE_MARGIN_MS,
        isCoolingDown: () =>
            !!getLimiterForSlot({
                slot,
                organizationId: opts.organizationId,
            })?.isInCooldown(),
    });
}

/**
 * Tell the breaker how an attempt failed. Only a failure that says the
 * provider is unwell AND moved the call on counts (`circuitCountsFailure`);
 * anything else just hands a probe lease back. A failure after
 * `markUnsafeToRetry` still counts — the generation on the ticket decides
 * whether it is still news.
 */
function reportAttemptFailure(
    ticket: CircuitAdmission['ticket'],
    err: unknown,
): void {
    if (!ticket) return;
    if (circuitCountsFailure(err) && shouldFailoverToNextModel(err)) {
        providerCircuit.onFailure(ticket, {
            retryAfterMs: retryAfterMsOf(err),
        });
    } else {
        providerCircuit.release(ticket);
    }
}

/**
 * Run `runOne` on the primary slot; on a cascade-worthy failure it could not
 * recover from, re-run on the next chain entry (the tail of `slots`), in order.
 * Returns the first success; re-throws the LAST error when every attempt is
 * exhausted, the failure is not cascade-worthy, the attempt was marked unsafe,
 * or `opts.signal` has aborted (the caller's budget is gone, so the next entry
 * is not started). With a single slot this is a thin pass-through — `runOne` is
 * invoked exactly once.
 *
 * Before each attempt the circuit breaker is asked (`admitAttempt`): an entry
 * whose model keeps failing is skipped — recorded as a `circuit-open` hop —
 * unless it is the last, which always runs. The attempt reports its first
 * provider answer and its counted failures back to the breaker.
 */
export async function runWithModelFailover<T>(
    slots: Array<NormalizedModel | undefined>,
    runOne: (
        slot: NormalizedModel | undefined,
        control: FailoverAttemptControl,
    ) => Promise<T>,
    opts: { runName: string; organizationId?: string; signal?: AbortSignal },
): Promise<T> {
    const attempts = distinctAttempts(slots);
    const hops: FailoverHop[] = [];
    const nameOf = (i: number) => attempts[i]?.model ?? 'managed-default';

    for (let i = 0; i < attempts.length; i++) {
        const attemptSlot = attempts[i];
        const isLastEntry = i >= attempts.length - 1;
        const admission = admitAttempt(attemptSlot, isLastEntry, opts);
        if (admission?.decision === 'skip') {
            hops.push({
                model: nameOf(i),
                chainIndex: i,
                outcome: 'skipped',
                reason: 'circuit-open',
                elapsedMs: 0,
            });
            logger.log({
                message: `${LLM_CIRCUIT_TAG} ${opts.runName}: "${nameOf(i)}" skipped (circuit ${admission.state}); trying "${nameOf(i + 1)}"`,
                context: 'runWithModelFailover',
                metadata: {
                    runName: opts.runName,
                    organizationId: opts.organizationId,
                    chainIndex: i,
                    circuitState: admission.state,
                    fromModelId: attemptSlot?.byokModelId,
                    toModelId: attempts[i + 1]?.byokModelId,
                },
            });
            continue;
        }
        if (admission?.wouldSkip) {
            logger.log({
                message: `${LLM_CIRCUIT_TAG} ${opts.runName}: "${nameOf(i)}" would-skip (circuit ${admission.state}, shadow mode)`,
                context: 'runWithModelFailover',
                metadata: {
                    runName: opts.runName,
                    organizationId: opts.organizationId,
                    chainIndex: i,
                    circuitState: admission.state,
                    modelId: attemptSlot?.byokModelId,
                },
            });
        }
        const ticket = admission?.ticket ?? null;

        let unsafeToRetry = false;
        let answered = false;
        const onAnswered = () => {
            if (answered) return;
            answered = true;
            providerCircuit.onSuccess(ticket);
        };
        const control: FailoverAttemptControl = {
            hasNext: !isLastEntry,
            isUnsafe: () => unsafeToRetry,
            markUnsafeToRetry: () => {
                unsafeToRetry = true;
            },
            onAnswered,
            chainIndex: i,
            circuitState: admission?.state,
            failoverReason: hops.at(-1)?.reason,
        };
        const startedAt = Date.now();

        try {
            const result = await runOne(attemptSlot, control);
            // A result is an answer, whether or not the executor reported it.
            onAnswered();
            if (i > 0) {
                failoverRecorder.getStore()?.push({
                    runName: opts.runName,
                    failedModel: nameOf(0),
                    usedModel: nameOf(i),
                    chainIndex: i,
                    hops: [...hops],
                });
            }
            // DEBUG level: completes the [LLM-ERROR]/[LLM-SUCCESS] pair at the one
            // chokepoint every LLM.run funnels through, WITHOUT flooding prod —
            // one success line per call is too much at info, so it stays off
            // unless debug logging is enabled (then a full call trace is greppable
            // by tag). `usedFallback` flags a call that only survived via failover.
            logger.debug({
                message: `${LLM_SUCCESS_TAG} ${opts.runName}: "${nameOf(i)}" ok${i > 0 ? ' (via fallback)' : ''}`,
                context: 'runWithModelFailover',
                metadata: {
                    runName: opts.runName,
                    organizationId: opts.organizationId,
                    modelId: attempts[i]?.byokModelId,
                    usedFallback: i > 0,
                    chainIndex: i,
                },
            });
            return result;
        } catch (err) {
            const { category } = classifyLLMError(err);
            const reason = hopReason(err, category);
            hops.push({
                model: nameOf(i),
                chainIndex: i,
                outcome: 'failed',
                reason,
                elapsedMs: Date.now() - startedAt,
            });
            reportAttemptFailure(ticket, err);

            const callerAborted = !!opts.signal?.aborted;
            if (
                isLastEntry ||
                unsafeToRetry ||
                callerAborted ||
                !shouldFailoverToNextModel(err)
            ) {
                // Terminal failure: no more attempts (or this error must not
                // cascade). Emit ONE greppable [LLM-ERROR] line here — the single
                // chokepoint every LLM.run call funnels through — so a failed LLM
                // call is findable in the logs with model + classified cause,
                // instead of dying as an unlogged re-throw. WARN level: a terminal
                // LLM failure is a user config/billing/provider problem, not an
                // app outage (mirrors the failover swap below).
                const earlier = hops
                    .slice(0, -1)
                    .map((h) => `"${h.model}" ${h.outcome} (${h.reason})`);
                logger.warn({
                    message: `${LLM_ERROR_TAG} ${opts.runName}: "${nameOf(i)}" failed (${category}) — ${(err as Error)?.message ?? 'unknown error'}${earlier.length ? ` [after: ${earlier.join(', ')}]` : ''}`,
                    context: 'runWithModelFailover',
                    metadata: {
                        runName: opts.runName,
                        organizationId: opts.organizationId,
                        category,
                        reason,
                        modelId: attempts[i]?.byokModelId,
                        chainIndex: i,
                        exhausted: isLastEntry,
                        callerAborted,
                        hops,
                    },
                });
                // Record WHICH attempt this was, and the hops before it, before
                // it leaves: downstream reporting otherwise re-reads the pre-run
                // resolved slot and names the primary for a failure that
                // belongs to a fallback.
                throw typeof err === 'object' && err !== null
                    ? attachFailoverHops(
                          attachAttemptedSlot(err, attempts[i]),
                          hops,
                      )
                    : err;
            }

            // WARN not ERROR: a terminal failure is a user config/billing problem
            // (mirrors llmErrorLogLevel), and the run still succeeds via the
            // fallback — the interesting signal is the SWAP, not an outage.
            logger.warn({
                message: `${LLM_ERROR_TAG} [model-failover] ${opts.runName}: "${nameOf(i)}" failed (${reason}); cascading to fallback "${nameOf(i + 1)}"`,
                context: 'runWithModelFailover',
                metadata: {
                    runName: opts.runName,
                    organizationId: opts.organizationId,
                    category,
                    reason,
                    chainIndex: i,
                    fromModelId: attempts[i]?.byokModelId,
                    toModelId: attempts[i + 1]?.byokModelId,
                },
            });
        }
    }

    // Unreachable: `distinctAttempts` guarantees ≥1 attempt, and the LAST one always
    // throws in the catch above (`isLastEntry`). This satisfies the compiler's need for a
    // terminal path without carrying a dead "last error" across iterations.
    throw new Error(
        `runWithModelFailover(${opts.runName}): exhausted attempts without returning`,
    );
}
