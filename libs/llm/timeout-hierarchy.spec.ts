/**
 * The ceilings are a nesting, not four independent numbers.
 *
 * Each layer must leave room for the one inside it. Raise an inner one past
 * its outer and the outer fires first, which is how a timeout becomes
 * decoration: the number in the constant stops describing when anything
 * actually happens. That is exactly how `LLM_CALL_TIMEOUT_MS` and undici's
 * `headersTimeout` came to be pinned to each other by a comment — this locks
 * the relationship the comment describes.
 *
 * Sized off production, not preference: 14 days of `AgentReviewStage`
 * durations (n=94) gave p50 3.1 min, p95 17.8, p99 29.5, max 32.2, with 1.1%
 * already past the old 30-minute ceiling. The 2026-09-03 hang ran 98 minutes.
 * The gap between "large PR" and "stuck" is wide, and the agent ceiling has to
 * sit inside it.
 */
import { AGENT_TIMEOUT_MS, LLM_CALL_TIMEOUT_MS, EMBED_TIMEOUT_MS } from '@libs/llm/llm-call';
import { PROBE_TIMEOUT_MS } from '@libs/llm/probe-slot-call';
import { MAX_FALLBACKS } from '@libs/llm/byok-fallback-chain';
import {
    DEFAULT_ROUND_TRIP_DEADLINE_MS,
    MAX_ROUND_TRIP_DEADLINE_MS,
    ROUND_TRIP_SETTLE_GRACE_MS,
    failoverRoundTripDeadlineMs,
} from '@libs/llm/provider-deadline-error';

// undici headersTimeout (fetch-timeouts.ts), mirrored for the same reason as
// the job timeout below.
const UNDICI_HEADERS_TIMEOUT_MS = 20 * 60 * 1000;

// Mirrored from job-processor-router.service.ts. Duplicated on purpose: libs/llm
// must not depend on the workflow layer, and a silent drift here is exactly
// what this test exists to catch — if that constant moves, this fails and
// someone reads both.
const CODE_REVIEW_PROCESS_TIMEOUT_MS = 105 * 60 * 1000;

describe('LLM timeout hierarchy — every layer leaves room for the one inside', () => {
    it('an agent loop fits inside the review job that runs it', () => {
        expect(AGENT_TIMEOUT_MS).toBeLessThan(CODE_REVIEW_PROCESS_TIMEOUT_MS);
    });

    it('leaves the job real headroom for the stages around the agent', () => {
        // The pipeline still has to fetch files, post the initial comment, and
        // afterwards create comments and generate the summary. A loop that
        // consumed the whole job budget would be killed mid-publish.
        const headroom = CODE_REVIEW_PROCESS_TIMEOUT_MS - AGENT_TIMEOUT_MS;
        expect(headroom).toBeGreaterThanOrEqual(20 * 60 * 1000);
    });

    it('one model round-trip fits inside the loop that may make several', () => {
        expect(LLM_CALL_TIMEOUT_MS).toBeLessThan(AGENT_TIMEOUT_MS);
    });

    it('covers the slowest legitimate review observed in production', () => {
        // max 32.2 min over 14 days (n=94). Enforcing a ceiling at or under
        // that would kill real reviews of large PRs.
        expect(AGENT_TIMEOUT_MS).toBeGreaterThan(33 * 60 * 1000);
    });

    it('still cuts the pathological case, which ran 98 minutes', () => {
        expect(AGENT_TIMEOUT_MS).toBeLessThan(98 * 60 * 1000);
    });

    it('keeps the short-lived ceilings short — they answer a person, not a pipeline', () => {
        expect(PROBE_TIMEOUT_MS).toBeLessThan(LLM_CALL_TIMEOUT_MS);
        expect(EMBED_TIMEOUT_MS).toBeLessThan(LLM_CALL_TIMEOUT_MS);
    });

    // Both bounds matter, in opposite directions.
    //
    // Floor: the probe forwards the slot's configured reasoning budget, so on
    // a thinking model it waits for real thinking. It sat at 15s, which
    // reported working credentials as broken.
    //
    // Ceiling: it still blocks a form. Nothing above interactive scale is a
    // probe any more, which is what stops this drifting up to meet the call
    // ceiling the next time someone sees a timeout.
    it('gives the probe room for a thinking model without ceasing to be interactive', () => {
        expect(PROBE_TIMEOUT_MS).toBeGreaterThanOrEqual(60 * 1000);
        expect(PROBE_TIMEOUT_MS).toBeLessThanOrEqual(2 * 60 * 1000);
    });

    describe('the failover round-trip deadline', () => {
        // It exists to buy a cascade. If it were not inside the call budget,
        // the call budget (or undici beside it) would fire first, as a plain
        // timeout that does not cascade.
        it('sits strictly inside one model round-trip, which sits inside undici and the loop', () => {
            expect(MAX_ROUND_TRIP_DEADLINE_MS).toBeLessThan(
                LLM_CALL_TIMEOUT_MS,
            );
            expect(LLM_CALL_TIMEOUT_MS).toBeLessThanOrEqual(
                UNDICI_HEADERS_TIMEOUT_MS,
            );
            expect(UNDICI_HEADERS_TIMEOUT_MS).toBeLessThan(AGENT_TIMEOUT_MS);
        });

        it('leaves the last chain entry half the loop budget, even when every other entry stalls', () => {
            expect(
                MAX_FALLBACKS * MAX_ROUND_TRIP_DEADLINE_MS,
            ).toBeLessThanOrEqual(AGENT_TIMEOUT_MS / 2);
        });

        it('defaults to 5 minutes, inside the clamp, with room for the settle grace', () => {
            expect(DEFAULT_ROUND_TRIP_DEADLINE_MS).toBe(5 * 60 * 1000);
            expect(DEFAULT_ROUND_TRIP_DEADLINE_MS).toBeLessThanOrEqual(
                MAX_ROUND_TRIP_DEADLINE_MS,
            );
            expect(
                MAX_ROUND_TRIP_DEADLINE_MS + ROUND_TRIP_SETTLE_GRACE_MS,
            ).toBeLessThan(LLM_CALL_TIMEOUT_MS);
        });

        it('reads the knob, turns off on 0/off, and clamps what is too large', () => {
            const knob = (v?: string) =>
                failoverRoundTripDeadlineMs(
                    v === undefined
                        ? {}
                        : { LLM_FAILOVER_ROUND_TRIP_DEADLINE_MS: v },
                );
            expect(knob()).toBe(DEFAULT_ROUND_TRIP_DEADLINE_MS);
            expect(knob('')).toBe(DEFAULT_ROUND_TRIP_DEADLINE_MS);
            expect(knob('90000')).toBe(90_000);
            expect(knob('0')).toBeUndefined();
            expect(knob('0.0')).toBeUndefined();
            expect(knob('off')).toBeUndefined();
            expect(knob('soon')).toBe(DEFAULT_ROUND_TRIP_DEADLINE_MS);
            expect(knob(String(LLM_CALL_TIMEOUT_MS))).toBe(
                MAX_ROUND_TRIP_DEADLINE_MS,
            );
        });
    });
});
