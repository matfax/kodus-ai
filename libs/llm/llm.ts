/**
 * The ONE way to call an LLM.
 *
 * Like any SDK (`client.chat.completions.create`, `generateText`), you call one
 * method and get the result. The difference from a raw SDK: you don't pick the
 * `model` — you give the `task`, and the org's BYOK config resolves the model +
 * key + routing. Everything the three BYOK missions own lives INSIDE this call,
 * so no caller re-implements or threads any of it:
 *   1. access  — router → slot → model (BYOK key or managed default)
 *   2. tracing — the observability span (pulled from the app singleton here, so
 *                callers never pass an `observabilityService`)
 *   3. router  — task → slot via the org's stored routing config
 *
 * Shape:
 *   - `schema` present  → structured output, returns the parsed object
 *   - no `schema`       → plain text, returns the string
 *
 * Model source: pass a pre-resolved `byokConfig` slot (the caller already
 * routed), OR a `{ config, task }` pair to route here — routing is pure (the org
 * config is fetched upstream), so this stays in @libs/llm with no DB dependency.
 *
 * The tool-driven ("agent") path is NOT here: an agent is a LOOP around this
 * primitive (tools, MCP, N steps) — orchestration ABOVE the call, driven by the
 * agent runner, not by BYOK. It consumes the same assembled model.
 */
import { z } from 'zod';
import type { generateText, ModelMessage, Schema } from 'ai';
import type {
    NormalizedModel,
    BYOKConfig,
    LlmTask,
} from '@libs/llm/byok-config';
import { resolveTaskSlot } from '@libs/llm/resolve-task-model';
import { asServedSlot } from '@libs/llm/served-slot';
import type { RequestContext } from '@libs/llm/routing-strategy';
import {
    runStructuredReviewCall,
    runTextReviewCall,
    type BaseReviewCallParams,
} from '@libs/llm/structured-review-call';
import {
    runAgentLoopCall,
    type AgentLoopLeg,
    type AgentLoopSeams,
} from '@libs/llm/agent-loop-call';
import {
    loopContinueEnabled,
    roundTripDeadlineFor,
    runWithModelFailover,
    type FailoverAttemptControl,
    type ResumeInfo,
} from '@libs/llm/model-failover';
import type { ByokRoundTrip } from '@libs/llm/byok-model-wrapper';
import { LLM_CALL_TIMEOUT_MS } from '@libs/llm/llm-call';

/** The raw AI SDK result an agent-loop call returns (steps / usage / text). */
export type AgentLoopResult = Awaited<ReturnType<typeof generateText>>;

export interface LlmRequest extends Omit<
    BaseReviewCallParams,
    'byokConfig' | 'user' | 'hasNextModel' | 'roundTrip'
> {
    /** A pre-resolved slot — the caller already routed task → slot. */
    byokConfig?: NormalizedModel;
    /** OR route here: the org's stored BYOK config + the task. Pure routing. */
    config?: BYOKConfig | null;
    task?: LlmTask;
    /** Optional routing context (repo/provider hints) forwarded to the strategy. */
    ctx?: RequestContext;

    /** One-shot content: the user turn. Required for one-shot; the loop path
     *  uses `messages` instead. */
    user?: string;

    // ── agent-loop mode (presence of `loop` runs the tool-driven loop) ──
    /** The loop seams (tools / maxSteps / policy stopWhen·prepareStep·onStepFinish),
     *  built by the harness runner. Present → LLM.run runs the multi-step loop. */
    loop?: AgentLoopSeams;
    /** The conversation array for the loop path (the runner assembles it). */
    messages?: ModelMessage[];
    /** Cancellation / hard-timeout signal (loop path; the caller composes it). */
    signal?: AbortSignal;
    /** BYOK failure reporter (byok-error-counter hook). */
    reporter?: (input: {
        organizationId?: string;
        provider: string;
        errorMessage: string;
    }) => void;
    /** Concurrency-limiter queue timeout (loop path; the finder threads its own). */
    queueTimeoutMs?: number;
    /** Provider override for the limiter (loop path; defaults to the slot's). */
    provider?: string;
    /** Pre-built providerOptions override (loop path) — the finder's config-derived
     *  reasoning; unset → LLM.run derives it from the slot. */
    providerOptions?: Record<string, unknown>;
    /** Override the slot's sampling temperature. */
    temperature?: number;
    /** Fallback max-output when the slot omits one (loop path). */
    maxOutputTokens?: number;

    // NOTE: no `observabilityService` — LLM owns the span via the app singleton.
    // A service's injected instance IS that same singleton, so passing it was a
    // no-op; callers never thread observability.
}

/**
 * Resolve the ONE slot for this call: an explicit slot wins; otherwise route
 * from `{ config, task }`; otherwise undefined → the managed default (resolved
 * inside the executor).
 */
function resolveSlot(req: LlmRequest): NormalizedModel | undefined {
    if (req.byokConfig) {
        return asServedSlot(req.byokConfig);
    }
    if (req.config && req.task) {
        return resolveTaskSlot(req.config, req.task, { ctx: req.ctx }).slot;
    }
    return undefined;
}

/**
 * Strip the routing-only fields — what reaches the one-shot executor is a clean
 * slot call. The slot is NOT set here: `run` injects the per-attempt slot (primary
 * or fallback) so the failover cascade can re-issue on a different model.
 * Observability is NOT threaded here either: the executor reads it from the LLM
 * observability port, so no caller (nor this) ever passes it.
 */
function toExecutorParams(
    req: LlmRequest,
): Omit<BaseReviewCallParams, 'byokConfig' | 'hasNextModel' | 'roundTrip'> {
    const {
        config: _config,
        task: _task,
        ctx: _ctx,
        // loop-only fields never reach the one-shot executor
        loop: _loop,
        messages: _messages,
        signal: _signal,
        reporter: _reporter,
        // a pre-resolved slot is re-injected per attempt by `run`, not here
        byokConfig: _byokConfig,
        // temperature / maxOutputTokens / providerOptions DO pass through — the
        // one-shot executor honors them as fixed-tuning overrides (demo paths).
        ...rest
    } = req;
    return rest as Omit<
        BaseReviewCallParams,
        'byokConfig' | 'hasNextModel' | 'roundTrip'
    >;
}

/**
 * The runner's loop seams as one attempt runs them.
 *
 * - Tools: `toolsForModel`, when the runner supplied it, builds this attempt's
 *   tool map for its own model, so a chain entry gets tools it can accept.
 * - Steps: the leg's budget, which is less than the loop's when this attempt
 *   continues a failed one.
 * - Failover guard: the FIRST emitted step vetoes a plain restart for the
 *   current attempt. A step's tool calls mutate shared runner state (its step
 *   array + policy progress), so restarting on the fallback would double-count
 *   them. Only `continueFrom` moves such an attempt on, by continuing instead.
 */
function loopForAttempt(
    loop: AgentLoopSeams,
    leg: AgentLoopLeg,
    attemptSlot: NormalizedModel | undefined,
    control: FailoverAttemptControl,
): AgentLoopSeams {
    const { toolsForModel, continueFrom: _continueFrom, ...seams } = loop;
    const inner = loop.onStepFinish;
    return {
        ...seams,
        maxSteps: leg.maxSteps,
        tools: toolsForModel ? toolsForModel(attemptSlot?.model) : loop.tools,
        onStepFinish: (event: unknown) => {
            control.markUnsafeToRetry();
            return inner?.(event);
        },
    };
}

/**
 * One attempt's round-trip deadline, handed to the BYOK wrapper.
 *
 * It exists only to buy a cascade, so it is active only while one is possible:
 * a next chain entry exists and (loop mode) no step has run or the loop can
 * continue on that entry. The last entry, the only entry and the steps of a
 * loop that cannot continue behave exactly as without it. A one-shot caller with a shorter `timeoutMs` than the knob gets that
 * budget as its deadline (`caller-budget`): the call cascades at the budget
 * instead of failing as a plain timeout, but the breaker does not count it.
 *
 * The answer hook reports to the breaker on every attempt, deadline or not.
 */
function roundTripFor(
    slot: NormalizedModel | undefined,
    control: FailoverAttemptControl,
    opts: { callerBudgetMs?: number; canContinue?: boolean } = {},
): ByokRoundTrip {
    const { callerBudgetMs, canContinue = false } = opts;
    const knob = roundTripDeadlineFor(slot);
    const fromCaller =
        knob !== undefined &&
        callerBudgetMs !== undefined &&
        callerBudgetMs < knob;
    return {
        ms: fromCaller ? callerBudgetMs : knob,
        source: fromCaller ? 'caller-budget' : 'round-trip',
        isActive: () =>
            control.hasNext && (canContinue || !control.isUnsafe()),
        onAnswered: () => control.onAnswered(),
    };
}

/**
 * The span attributes of one attempt: the caller's, plus where the attempt sits
 * in the chain and why it got there. Added only when they say something (a
 * fallback attempt, a breaker that is not CLOSED), so a call that ran on its
 * primary records exactly the attributes it always did.
 */
function attemptAttrs(
    attrs: Record<string, unknown> | undefined,
    control: FailoverAttemptControl,
): Record<string, unknown> | undefined {
    const extra: Record<string, unknown> = {
        // `usedFallback` is already on the span: a chain entry's slot carries
        // it from routing.
        ...(control.chainIndex > 0
            ? {
                  chainIndex: control.chainIndex,
                  failoverReason: control.failoverReason,
              }
            : {}),
        ...(control.circuitState && control.circuitState !== 'closed'
            ? { circuitState: control.circuitState }
            : {}),
    };
    return Object.keys(extra).length ? { ...(attrs ?? {}), ...extra } : attrs;
}

export class LLM {
    /** Agent-loop call — a `loop` (tools + policies) runs the multi-step loop,
     *  returning the raw AI SDK result (steps / usage / text). */
    static run(
        req: LlmRequest & { loop: AgentLoopSeams; schema?: undefined },
    ): Promise<AgentLoopResult>;
    /** Structured call — a `schema` yields the parsed, typed object. */
    static run<S extends z.ZodType | Schema>(
        req: LlmRequest & { schema: S },
    ): Promise<
        S extends z.ZodType ? z.infer<S> : S extends Schema<infer T> ? T : never
    >;
    /** Text call — no `schema` yields the raw generated string. */
    static run(req: LlmRequest & { schema?: undefined }): Promise<string>;
    static run(
        req: LlmRequest & { schema?: z.ZodType | Schema },
    ): Promise<unknown> {
        // Resolve the routing decision ONCE. The primary slot carries its runtime
        // fallback chain in `.fallbacks` (stamped by resolveTaskSlot), so the
        // cascade is primary → fallbacks in order; a slot without fallbacks (or
        // the managed default, undefined) makes `runWithModelFailover` a
        // single-attempt pass-through.
        const slot = resolveSlot(req);
        const attempts = [slot, ...(slot?.fallbacks ?? [])];
        const failoverOpts = {
            runName: req.runName,
            organizationId: req.organizationId,
        };

        // Agent-loop mode: a tool-driven, multi-step call. LLM.run resolves the
        // model + tuning + reasoning + cache + span; the runner passed only the
        // loop seams. Returns the raw result for the runner to map onto RunState.
        if (req.loop) {
            const loop = req.loop;
            const continueFrom = loopContinueEnabled()
                ? loop.continueFrom
                : undefined;
            // What the next attempt runs. A continuation replaces it with the
            // leg the runner built from the failed attempt's progress.
            let leg: AgentLoopLeg = {
                messages: req.messages ?? [],
                maxSteps: loop.maxSteps,
            };
            const resumeAfterSteps = continueFrom
                ? (info: ResumeInfo) => {
                      const next = continueFrom(info);
                      if (!next) return false;
                      leg = next;
                      return true;
                  }
                : undefined;
            return runWithModelFailover(
                attempts,
                (attemptSlot, control) =>
                    runAgentLoopCall({
                        byokConfig: attemptSlot,
                        system: req.system,
                        messages: leg.messages,
                        loop: loopForAttempt(loop, leg, attemptSlot, control),
                        runName: req.runName,
                        spanName: req.spanName,
                        attrs: attemptAttrs(req.attrs, control),
                        organizationId: req.organizationId,
                        reporter: req.reporter,
                        queueTimeoutMs: req.queueTimeoutMs,
                        provider: req.provider,
                        providerOptions: req.providerOptions,
                        telemetryMetadata: req.telemetryMetadata,
                        signal: req.signal,
                        maxOutputTokens: req.maxOutputTokens,
                        temperature: req.temperature,
                        roundTrip: roundTripFor(attemptSlot, control, {
                            canContinue: !!continueFrom,
                        }),
                    }),
                // The loop's signal is the caller's budget: once it has aborted,
                // no further chain entry is started on it. One-shot calls carry
                // no caller signal (toExecutorParams drops it), so none is passed.
                {
                    ...failoverOpts,
                    signal: req.signal,
                    ...(resumeAfterSteps ? { resumeAfterSteps } : {}),
                },
            );
        }

        // Pull the schema out first so a text call can never carry a stray
        // `schema` prop into the text executor. One-shot calls are atomic, so they
        // never veto failover — a primary failure re-issues cleanly on the fallback.
        // `hasNextModel` tells D-00c whether a provider timeout has somewhere
        // to cascade, so it throws instead of re-issuing on the stalled route.
        const { schema, ...rest } = req;
        const baseParams = toExecutorParams(rest);
        return runWithModelFailover(
            attempts,
            (attemptSlot, control) => {
                const params: BaseReviewCallParams = {
                    ...baseParams,
                    byokConfig: attemptSlot,
                    attrs: attemptAttrs(baseParams.attrs, control),
                    hasNextModel: control.hasNext,
                    roundTrip: roundTripFor(attemptSlot, control, {
                        callerBudgetMs:
                            baseParams.timeoutMs ?? LLM_CALL_TIMEOUT_MS,
                    }),
                };
                return schema
                    ? runStructuredReviewCall({ ...params, schema })
                    : runTextReviewCall(params);
            },
            failoverOpts,
        );
    }
}
