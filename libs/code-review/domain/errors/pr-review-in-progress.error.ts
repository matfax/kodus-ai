import type { CommandReviewFeedbackTarget } from '@libs/code-review/infrastructure/adapters/services/codeReviewHandlerService.service';

/**
 * Thrown when a review cannot start because another run already holds the
 * PR — either the distributed lock or an in-flight execution row.
 *
 * A user-issued `@kody review` throws it as a `command`; an automatic review
 * (a push) as a `catch-up`, unless catch-ups are switched off
 * (`reviewCatchUpEnabled`), in which case the push is dropped quietly.
 *
 * Returning instead of throwing is what made the request vanish (#1700):
 * the caller chain ignores the returned string and marks the job COMPLETED,
 * so nothing retries and nothing reaches the PR. Throwing hands the
 * decision to the job processor, which can reschedule the job for after
 * the holder finishes.
 */
export class PrReviewInProgressError extends Error {
    readonly name = 'PrReviewInProgressError';

    /** Which gate refused it — for logging and metrics only. */
    readonly gate: 'lock' | 'execution';

    /** A person's `@kody review`, or a push that waits to catch up. */
    readonly kind: 'command' | 'catch-up';

    /** The head commit the refused push carried, when known: a catch-up
     *  whose push an earlier review already covered is not run again. */
    readonly pushHeadSha?: string;

    /** Everything needed to answer the user on the PR, if we give up. */
    readonly target: CommandReviewFeedbackTarget;

    /**
     * When the run holding this PR stops being visible to the gate that
     * refused us. Retrying past it is unsafe: the holder may still be
     * running while both gates have gone blind to it, so the retry would
     * start a second concurrent review.
     */
    readonly holderVisibleUntil?: Date;

    constructor(params: {
        gate: 'lock' | 'execution';
        target: CommandReviewFeedbackTarget;
        holderVisibleUntil?: Date;
        message?: string;
        kind?: 'command' | 'catch-up';
        pushHeadSha?: string;
    }) {
        super(
            params.message ??
                `Code review already in progress for PR#${params.target?.pullRequest?.number}`,
        );
        this.gate = params.gate;
        this.kind = params.kind ?? 'command';
        this.target = params.target;
        this.holderVisibleUntil = params.holderVisibleUntil;
        this.pushHeadSha = params.pushHeadSha;
    }
}

/**
 * Duck-typed so an error that crossed a serialization boundary is still
 * recognized, matching how the other workflow errors are detected.
 */
export function isPrReviewInProgressError(
    error: unknown,
): error is PrReviewInProgressError {
    if (error instanceof PrReviewInProgressError) return true;

    return (
        typeof error === 'object' &&
        error !== null &&
        (error as { name?: unknown }).name === 'PrReviewInProgressError' &&
        typeof (error as { target?: unknown }).target === 'object'
    );
}
