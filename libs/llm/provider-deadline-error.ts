/**
 * Thrown when a provider request outlives the round-trip deadline Kodus armed
 * on it — a deadline that only exists to buy a cascade to the next chain entry.
 *
 * Typed, not a message: the failover gate reads the CLASS (`isProviderTimeout`),
 * never prose, so a provider that stalls is told apart from a caller that
 * cancelled. It is deliberately not an `APICallError`, so the AI SDK never
 * retries it on the same model.
 *
 * Nothing throws it yet: the per-request deadline is armed by the BYOK wrapper
 * in a later change. The class exists now so the gate and the classifier are
 * already shaped for it.
 */
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
