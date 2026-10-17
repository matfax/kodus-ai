/**
 * Thrown when a provider request outlives the round-trip deadline Kodus armed
 * on it — a deadline that only exists to buy a cascade to the next chain entry.
 *
 * Typed, not a message: the failover gate reads the CLASS (`isProviderTimeout`),
 * never prose, so a provider that stalls is told apart from a caller that
 * cancelled. It is deliberately not an `APICallError`, so the AI SDK never
 * retries it on the same model.
 *
 * Thrown by the BYOK wrapper (`byok-model-wrapper.ts`), which arms the
 * deadline only while a cascade is possible: a next chain entry exists and no
 * agent-loop step has run yet.
 */
import { createLogger } from '@libs/core/log/logger';
import { MAX_FALLBACKS } from '@libs/llm/byok-fallback-chain';
import { AGENT_TIMEOUT_MS, LLM_CALL_TIMEOUT_MS } from '@libs/llm/llm-call';

const logger = createLogger('ProviderDeadline');

export type ProviderDeadlineSource = 'round-trip' | 'caller-budget';

/** Which wait the deadline bounded. Only `round-trip` is armed today; the
 *  streaming phases leave room for stall detection in the same gate. */
export type ProviderDeadlinePhase = 'round-trip' | 'first-chunk' | 'chunk-gap';

export interface ProviderDeadlineErrorParams {
    deadlineMs: number;
    source: ProviderDeadlineSource;
    phase: ProviderDeadlinePhase;
    provider?: string;
    model?: string;
}

export class ProviderDeadlineError extends Error {
    readonly deadlineMs: number;
    readonly source: ProviderDeadlineSource;
    readonly phase: ProviderDeadlinePhase;
    readonly provider?: string;
    readonly model?: string;

    constructor(params: ProviderDeadlineErrorParams) {
        const target = [params.provider, params.model]
            .filter(Boolean)
            .join('/');
        super(
            `[PROVIDER-DEADLINE] ${target || 'provider'} gave no answer within ${params.deadlineMs / 1000}s (${params.phase})`,
        );
        this.name = 'ProviderDeadlineError';
        this.deadlineMs = params.deadlineMs;
        this.source = params.source;
        this.phase = params.phase;
        this.provider = params.provider;
        this.model = params.model;
    }
}

/** Class check that also holds across module copies (jest module isolation,
 *  a duplicated bundle), where `instanceof` alone would miss. */
export function isProviderDeadlineError(
    err: unknown,
): err is ProviderDeadlineError {
    return (
        err instanceof ProviderDeadlineError ||
        (!!err &&
            typeof err === 'object' &&
            (err as { name?: unknown }).name === 'ProviderDeadlineError')
    );
}

// ─── The knob ────────────────────────────────────────────────────────────────

export const ROUND_TRIP_DEADLINE_ENV = 'LLM_FAILOVER_ROUND_TRIP_DEADLINE_MS';

/** How long a provider may take to answer one request before the call moves
 *  to the next chain entry. Only ever armed while a cascade is possible. */
export const DEFAULT_ROUND_TRIP_DEADLINE_MS = 5 * 60 * 1000;

/**
 * The ceiling the knob is clamped to, from the two bounds it must respect:
 *  - strictly below `LLM_CALL_TIMEOUT_MS`, or the call budget (and undici's
 *    headersTimeout beside it) fires first and the deadline is decoration;
 *  - `MAX_FALLBACKS × deadline ≤ AGENT_TIMEOUT_MS / 2`, so an agent loop whose
 *    every non-last entry stalls still leaves the last entry half its budget.
 */
export const MAX_ROUND_TRIP_DEADLINE_MS = Math.min(
    LLM_CALL_TIMEOUT_MS - 1,
    Math.floor(AGENT_TIMEOUT_MS / 2 / MAX_FALLBACKS),
);

/** Grace before the wrapper stops waiting for a provider that ignored the
 *  deadline's abort, and settles the caller itself. */
export const ROUND_TRIP_SETTLE_GRACE_MS = 5_000;

/** How long the limiter slot stays held for such a provider, past the
 *  deadline, so the limiter's count keeps matching the open sockets. */
export const ROUND_TRIP_SLOT_HOLD_CAP_MS = 60_000;

const warnedKnobValues = new Set<string>();

function warnKnobOnce(raw: string, message: string, used?: number): void {
    if (warnedKnobValues.has(raw)) return;
    warnedKnobValues.add(raw);
    logger.warn({
        message: `${ROUND_TRIP_DEADLINE_ENV}=${raw}: ${message}`,
        context: 'failoverRoundTripDeadlineMs',
        metadata: { raw, used, max: MAX_ROUND_TRIP_DEADLINE_MS },
    });
}

/**
 * `LLM_FAILOVER_ROUND_TRIP_DEADLINE_MS`, parsed and clamped. Unset →
 * `DEFAULT_ROUND_TRIP_DEADLINE_MS`; `0` or `off` → undefined (no deadline is
 * ever armed); an unparseable value keeps the default; a value past
 * `MAX_ROUND_TRIP_DEADLINE_MS` is clamped to it. Read per call, so no module
 * state caches a stale value (each odd value warns once).
 */
export function failoverRoundTripDeadlineMs(
    env: Readonly<Record<string, string | undefined>> = process.env,
): number | undefined {
    const raw = env[ROUND_TRIP_DEADLINE_ENV]?.trim();
    if (!raw) return DEFAULT_ROUND_TRIP_DEADLINE_MS;
    const lower = raw.toLowerCase();
    if (lower === 'off' || lower === 'false') return undefined;
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 0) {
        warnKnobOnce(
            raw,
            'not a duration in ms; using the default',
            DEFAULT_ROUND_TRIP_DEADLINE_MS,
        );
        return DEFAULT_ROUND_TRIP_DEADLINE_MS;
    }
    if (n === 0) return undefined;
    if (n > MAX_ROUND_TRIP_DEADLINE_MS) {
        warnKnobOnce(
            raw,
            'above the ceiling; clamped',
            MAX_ROUND_TRIP_DEADLINE_MS,
        );
        return MAX_ROUND_TRIP_DEADLINE_MS;
    }
    return Math.max(1, Math.floor(n));
}
