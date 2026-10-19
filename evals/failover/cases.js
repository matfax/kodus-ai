// The failover cases: what the primary model does wrong, and what the call must
// do about it. Each case runs on its own primary model (`failover-<id>`) on the
// faulty fake server, so the breaker state one case leaves (keyed per model)
// never reaches another; the fallback, the eval-fake model on the healthy
// server, is shared and never fails.
//
// `fault` is what the faulty server answers that model with: a mode of
// evals/shared/fake-llm-server.js, or a function of the request's number (1 =
// the first request for that model) returning one, or null to answer. `run`
// drives the calls through the context run.js hands it and returns the case's
// detail, one { what, ok } per thing it checks, or { detail, measures } when
// the case also measures one of the README's metrics.
//
// The context (run.js):
//   call(id, { signal, chain })  one LLM.run loop call routed to the case's
//                                primary; resolves { value, error, failovers,
//                                hops, answeredBy, ms, primaryRequests }
//   review(id, { chain, env, healthy })
//                                one finder run (real spec, real runner)
//                                routed to the case's primary; resolves
//                                { state, findings, hops, answeredBy, ms,
//                                primaryRequests }
//   bodiesOf(id)                 the request bodies a faulty-server model got
//   setFault(id, fault)          change what the primary answers from now on
//   circuitState(id)             the breaker state of the case's primary
//   sleep(ms), DEADLINE_MS, OPEN_MS, UNREAD, REVIEW_MAX_STEPS
const hop = (h, outcome, reason) => !!h && h.outcome === outcome && h.reason === reason;
const hopsText = (hops) => (hops.length ? hops.map((h) => `${h.model} ${h.outcome} (${h.reason})`).join(', ') : 'none');

// The call ran on `who` ('primary' | 'fallback') and completed its loop: a
// tool step, then a final answer.
function completedOn(r, who) {
    const steps = r.value?.steps?.length ?? 0;
    return {
        what: `the call completes on the ${who} (answered by ${r.answeredBy ?? `nobody: ${r.error?.message ?? 'no result'}`}, ${steps} steps)`,
        ok: !r.error && r.answeredBy === who && steps >= 2,
    };
}

// A primary that answers its first `steps` requests (a finder run's readFile,
// then reportFinding), then fails with `mode`.
const after = (steps, mode) => (n) => (n <= steps ? null : mode);

// The finder run finished on a fallback, without an error.
function reviewCompletedOnFallback(r) {
    return {
        what: `the run completes on a fallback (status ${r.state.status}, answered by ${r.answeredBy ?? 'nobody'})`,
        ok: r.state.status !== 'error' && !!r.answeredBy && r.answeredBy !== 'primary',
    };
}

const reportsOf = (r) => r.state.artifacts.filter((a) => a.type === 'reportFinding');

const CASES = [
    {
        id: 'control-healthy-primary',
        checks: 'Answers on a healthy primary with no hop and no call to the fallback (control: nothing to cascade)',
        fault: null,
        run: async (ctx) => {
            const r = await ctx.call('control-healthy-primary');
            return [completedOn(r, 'primary'), { what: `no hop is recorded (hops: ${hopsText(r.hops)})`, ok: r.hops.length === 0 }];
        },
    },
    {
        id: '504-html',
        checks: 'Moves a call whose primary answers an HTML 504 "Gateway Timeout" page to the fallback, as a provider-timeout hop',
        fault: '504-html',
        run: async (ctx) => {
            const r = await ctx.call('504-html');
            const detail = [completedOn(r, 'fallback'), { what: `the primary's hop is failed (provider-timeout) (hops: ${hopsText(r.hops)})`, ok: r.hops.length === 1 && hop(r.hops[0], 'failed', 'provider-timeout') }];
            return { detail, measures: { html504Cascaded: r.answeredBy === 'fallback' } };
        },
    },
    {
        id: '503-json',
        checks: 'Moves a call whose primary keeps answering a JSON 503 to the fallback once the same-model retries are spent, as a TRANSIENT hop',
        fault: '503',
        run: async (ctx) => {
            const r = await ctx.call('503-json');
            return [
                completedOn(r, 'fallback'),
                { what: `the primary's hop is failed (TRANSIENT) (hops: ${hopsText(r.hops)})`, ok: r.hops.length === 1 && hop(r.hops[0], 'failed', 'TRANSIENT') },
                // The SDK retries a 5xx on the same model before it cascades.
                { what: `the primary is retried before the cascade (${r.primaryRequests} requests)`, ok: r.primaryRequests > 1 },
            ];
        },
    },
    {
        id: 'hang',
        checks: 'Gives up on a primary that never answers at the round-trip deadline, without retrying it, and the fallback answers',
        fault: 'hang',
        run: async (ctx) => {
            const r = await ctx.call('hang');
            const limit = ctx.DEADLINE_MS + 2000;
            const detail = [
                completedOn(r, 'fallback'),
                { what: `the primary's hop is failed (deadline) (hops: ${hopsText(r.hops)})`, ok: r.hops.length === 1 && hop(r.hops[0], 'failed', 'deadline') },
                { what: `the call ends within the deadline + 2s (${r.ms} ms, limit ${limit} ms)`, ok: r.ms <= limit },
                { what: `the hung primary is asked once, not retried (${r.primaryRequests} requests)`, ok: r.primaryRequests === 1 },
            ];
            return { detail, measures: { hungPrimaryGiveUpMs: r.hops[0]?.elapsedMs ?? null } };
        },
    },
    {
        id: '429-once',
        checks: 'Keeps a call on the primary when one 429 is followed by an answer: the same-model retry handles it and the fallback never runs',
        fault: (n) => (n === 1 ? '429' : null),
        run: async (ctx) => {
            const r = await ctx.call('429-once');
            return [
                completedOn(r, 'primary'),
                { what: `no hop is recorded (hops: ${hopsText(r.hops)})`, ok: r.hops.length === 0 },
                { what: `the 429 was retried on the primary (${r.primaryRequests} requests)`, ok: r.primaryRequests >= 3 },
            ];
        },
    },
    {
        id: '429-exhausted',
        checks: 'Moves a call to the fallback only once a 429 outlives the same-model retries, and never opens the circuit: the next call tries the primary again',
        fault: '429',
        run: async (ctx) => {
            const first = await ctx.call('429-exhausted');
            const second = await ctx.call('429-exhausted');
            return [
                completedOn(first, 'fallback'),
                { what: `the primary's hop is failed (RATE_LIMIT) (hops: ${hopsText(first.hops)})`, ok: first.hops.length === 1 && hop(first.hops[0], 'failed', 'RATE_LIMIT') },
                { what: `the primary's circuit stays closed (${ctx.circuitState('429-exhausted')})`, ok: ctx.circuitState('429-exhausted') === 'closed' },
                { what: `the next call asks the primary again, not skipped (hops: ${hopsText(second.hops)}, ${second.primaryRequests} requests)`, ok: hop(second.hops[0], 'failed', 'RATE_LIMIT') && second.primaryRequests > 0 },
            ];
        },
    },
    {
        id: 'caller-abort',
        checks: "Ends the call when the caller aborts while the primary hangs: the fallback never starts and the breaker does not count it",
        fault: 'hang',
        run: async (ctx) => {
            const abortAfterMs = Math.round(ctx.DEADLINE_MS / 4);
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), abortAfterMs);
            const r = await ctx.call('caller-abort', { signal: controller.signal });
            clearTimeout(timer);
            return [
                { what: `the call fails with the caller's abort (${r.error ? r.error.name || 'Error' : `answered by ${r.answeredBy}`})`, ok: !!r.error && r.callerAbort },
                { what: `only the primary was tried (hops: ${hopsText(r.hops)})`, ok: r.hops.length === 1 && r.hops[0].chainIndex === 0 },
                { what: `it ends before the deadline would have (${r.ms} ms, deadline ${ctx.DEADLINE_MS} ms)`, ok: r.ms < ctx.DEADLINE_MS },
                { what: `the primary's circuit stays closed (${ctx.circuitState('caller-abort')})`, ok: ctx.circuitState('caller-abort') === 'closed' },
            ];
        },
    },
    {
        id: 'chain',
        checks: 'Walks a three-model chain in order: a 504 on the primary and a hang on the first fallback both cascade, and the second fallback answers',
        fault: '504-html',
        run: async (ctx) => {
            const r = await ctx.call('chain', { chain: [{ id: 'chain-hop', fault: 'hang' }] });
            return [
                completedOn(r, 'fallback'),
                { what: `the answer came from chain entry 2 (${r.failovers[0]?.chainIndex ?? 'none'})`, ok: r.failovers[0]?.chainIndex === 2 },
                {
                    what: `hops in order: primary provider-timeout, then the first fallback deadline (hops: ${hopsText(r.hops)})`,
                    ok: r.hops.length === 2 && hop(r.hops[0], 'failed', 'provider-timeout') && hop(r.hops[1], 'failed', 'deadline') && r.hops[1].chainIndex === 1,
                },
            ];
        },
    },
    {
        id: 'circuit',
        checks: 'Opens the circuit of a primary that keeps failing: the next call sends it no request, a later probe reaches it, and a probe it answers closes it',
        fault: '503',
        run: async (ctx) => {
            const detail = [];
            const opened = await ctx.call('circuit');
            detail.push(completedOn(opened, 'fallback'));
            detail.push({ what: `one counted failure opens the primary's circuit (${ctx.circuitState('circuit')})`, ok: ctx.circuitState('circuit') === 'open' });

            const skipped = await ctx.call('circuit');
            detail.push({ what: `the next call skips the open primary (hops: ${hopsText(skipped.hops)})`, ok: skipped.hops.length === 1 && hop(skipped.hops[0], 'skipped', 'circuit-open') });
            detail.push({ what: `and sends it no request (${skipped.primaryRequests} requests)`, ok: skipped.primaryRequests === 0 });
            detail.push(completedOn(skipped, 'fallback'));

            // After the open period one probe reaches the primary; it fails, so the
            // circuit opens again (for twice as long) and the next call skips it.
            await ctx.sleep(ctx.OPEN_MS + 100);
            const probed = await ctx.call('circuit');
            const afterProbe = await ctx.call('circuit');
            detail.push({ what: `after the open period a probe reaches the primary (${probed.primaryRequests} requests, hops: ${hopsText(probed.hops)})`, ok: probed.primaryRequests > 0 && hop(probed.hops[0], 'failed', 'TRANSIENT') });
            detail.push({ what: `the failed probe opens it again: the next call skips it (hops: ${hopsText(afterProbe.hops)}, ${afterProbe.primaryRequests} requests)`, ok: hop(afterProbe.hops[0], 'skipped', 'circuit-open') && afterProbe.primaryRequests === 0 });

            // The primary recovers; the probe after the doubled open period
            // answers and closes the circuit.
            ctx.setFault('circuit', null);
            await ctx.sleep(2 * ctx.OPEN_MS + 100);
            const recovered = await ctx.call('circuit');
            detail.push(completedOn(recovered, 'primary'));
            detail.push({ what: `a probe the primary answers closes its circuit (${ctx.circuitState('circuit')})`, ok: ctx.circuitState('circuit') === 'closed' });
            return { detail, measures: { openCircuitRequests: skipped.primaryRequests } };
        },
    },
    {
        id: 'midloop-hang',
        checks: 'Continues a finder run on the fallback at the deadline when its primary stalls after two steps, keeping its report and handing over what is still unread',
        fault: after(2, 'hang'),
        run: async (ctx) => {
            const r = await ctx.review('midloop-hang', { chain: [{ id: 'midloop-hang-backup', fault: null }] });
            const handover = JSON.stringify(ctx.bodiesOf('midloop-hang-backup')[0]?.messages ?? []);
            const reports = reportsOf(r);
            const limit = ctx.DEADLINE_MS + 2000;
            const detail = [
                reviewCompletedOnFallback(r),
                { what: `the primary's hop is failed (deadline) and continued (hops: ${hopsText(r.hops)}, continued: ${r.hops[0]?.continued})`, ok: r.hops.length === 1 && hop(r.hops[0], 'failed', 'deadline') && r.hops[0].continued === true },
                { what: `the primary's report is kept and the fallback's added (reports at ${reports.map((a) => a.location).join(', ') || 'none'})`, ok: reports.length === 2 && reports[0].location === 'step:1' },
                { what: `the run stays within its step budget (${r.state.steps.length} of ${ctx.REVIEW_MAX_STEPS})`, ok: r.state.steps.length <= ctx.REVIEW_MAX_STEPS },
                { what: 'the fallback is handed the findings already reported', ok: handover.includes('Findings already reported') },
                { what: 'and the code still unread', ok: handover.includes(ctx.UNREAD) },
                { what: `the run ends within the deadline + 2s (${r.ms} ms, limit ${limit} ms)`, ok: r.ms <= limit },
            ];
            return { detail, measures: { midLoopStallGiveUpMs: r.hops[0]?.elapsedMs ?? null } };
        },
    },
    {
        id: 'midloop-503',
        checks: 'Continues a finder run on the fallback when its primary keeps answering 503 after two steps, once the same-model retries are spent',
        fault: after(2, '503'),
        run: async (ctx) => {
            const r = await ctx.review('midloop-503', { chain: [{ id: 'midloop-503-backup', fault: null }] });
            return [
                reviewCompletedOnFallback(r),
                { what: `the primary's hop is failed (TRANSIENT) and continued (hops: ${hopsText(r.hops)})`, ok: r.hops.length === 1 && hop(r.hops[0], 'failed', 'TRANSIENT') && r.hops[0].continued === true },
                { what: `the 503 is retried on the primary first (${r.primaryRequests} requests)`, ok: r.primaryRequests > 3 },
                { what: `both models' reports are in the run (${reportsOf(r).length})`, ok: reportsOf(r).length === 2 },
            ];
        },
    },
    {
        id: 'midloop-all-fail',
        checks: 'Keeps the finding a finder run reported when its primary fails after two steps and the only fallback fails too',
        fault: after(2, '503'),
        run: async (ctx) => {
            const r = await ctx.review('midloop-all-fail', { chain: [{ id: 'midloop-all-fail-backup', fault: '503' }], healthy: false });
            return [
                { what: `the run ends in an error (status ${r.state.status})`, ok: r.state.status === 'error' },
                { what: `the fallback was asked (${ctx.requestsOf('midloop-all-fail-backup')} requests)`, ok: ctx.requestsOf('midloop-all-fail-backup') > 0 },
                { what: `the reported finding is kept (${r.findings.length} findings)`, ok: r.findings.length === 1 },
            ];
        },
    },
    {
        id: 'midloop-switch-off',
        checks: 'With LLM_LOOP_CONTINUE=off, ends a finder run whose primary fails after two steps without asking the fallback, as before continuation existed',
        fault: after(2, '503'),
        run: async (ctx) => {
            const r = await ctx.review('midloop-switch-off', { chain: [{ id: 'midloop-switch-off-backup', fault: null }], env: { LLM_LOOP_CONTINUE: 'off' } });
            return [
                { what: `the run ends in an error (status ${r.state.status})`, ok: r.state.status === 'error' },
                { what: `the fallback is never asked (${ctx.requestsOf('midloop-switch-off-backup')} requests)`, ok: ctx.requestsOf('midloop-switch-off-backup') === 0 },
            ];
        },
    },
];

module.exports = { CASES };
