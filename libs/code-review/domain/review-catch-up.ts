/**
 * Catch-up reviews: a push refused because a review of the same PR was
 * running waits for that review to finish and then runs, instead of being
 * dropped. Its commits would otherwise only be reviewed if another push or
 * `@kody review` came later.
 */

/**
 * Kill switch. On by default; `CODE_REVIEW_CATCH_UP=off` drops a refused
 * push quietly again, as before catch-ups existed. Read per call.
 */
export function reviewCatchUpEnabled(
    env: Readonly<Record<string, string | undefined>> = process.env,
): boolean {
    const raw = env.CODE_REVIEW_CATCH_UP?.trim().toLowerCase();
    return !(raw === 'off' || raw === 'false' || raw === '0');
}
