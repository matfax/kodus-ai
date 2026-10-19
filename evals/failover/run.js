#!/usr/bin/env node
// failover: an org's fallback chain when its primary model fails, stalls or is
// rate-limited, end to end through LLM.run's agent loop: routing
// (resolveTaskSlot stamps the chain), the cascade (runWithModelFailover), the
// circuit breaker, the round-trip deadline, the BYOK wrapper and the real
// openai_compatible provider over HTTP. Only the vendors are fake: the primary
// is a fake-llm-server that answers with faults (cases.js), the fallback the
// eval-fake model.
//
//   node evals/failover/run.js [--gate] [--output=<result.json>] [--case=<id>]
//
// The healthy server is the wiring smoke's (EVAL_FAKE_LLM_URL) when it runs
// there, else one this runner starts. Every case runs in this one process:
// the breaker's state is process-local, which is what the circuit case reads.
//
// Exit: 0 pass, 1 a check failed (--gate), 2 did not measure (evals/AGENTS.md).
const fs = require('fs');
const path = require('path');
const { startFakeLlmServer } = require('../shared/fake-llm-server');
const { TIER0 } = require('../shared/tier0-models');
const { registerTs, countTokens, tokensUsed } = require('../shared/eval-kit');
const { evalQuestion, resultProblems } = require('../shared/eval-result');
const { CASES } = require('./cases');

const args = Object.fromEntries(
    process.argv.slice(2).map((a) => {
        const m = a.match(/^--([^=]+)(?:=(.*))?$/);
        return m ? [m[1], m[2] ?? true] : [a, true];
    }),
);
const MODEL = 'eval-fake';

// Short enough that the hang cases take seconds, not the production default.
const DEADLINE_MS = 1000;
const OPEN_MS = 1000;
// The breaker as production runs it, but opened by one counted failure, so a
// case needs one failing call instead of three. Set here, not inherited, so a
// shell's or a default's change does not move the eval.
const ENV = {
    LLM_PROVIDER_TIMEOUT_CASCADE: 'on',
    LLM_FAILOVER_ROUND_TRIP_DEADLINE_MS: String(DEADLINE_MS),
    LLM_CIRCUIT_MODE: 'enforce',
    LLM_CIRCUIT_FAILURE_THRESHOLD: '1',
    LLM_CIRCUIT_OPEN_MS: String(OPEN_MS),
    LLM_LOOP_CONTINUE: 'on',
};
// A finder run's caller budget: a mid-loop stall the run cannot move past
// ends here instead of at AGENT_TIMEOUT_MS, so the negative control fails
// in seconds rather than hanging.
const REVIEW_BUDGET_MS = 5 * DEADLINE_MS;

// A UUID, like a real org: the breaker keys BYOK calls per org.
const ORG = '00000000-0000-4000-8000-0000000fa110';
const FALLBACK_ID = 'fallback';
const modelOf = (id) => `failover-${id}`;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function main() {
    // libs/common/utils/crypto.ts reads API_CRYPTO_KEY at import time; CI has no .env.
    if (!process.env.API_CRYPTO_KEY) process.env.API_CRYPTO_KEY = '0'.repeat(64);
    Object.assign(process.env, ENV);

    const cases = CASES.filter((c) => !args.case || c.id === args.case);
    if (!cases.length) {
        console.error(`not measured: no case "${args.case}" (have: ${CASES.map((c) => c.id).join(', ')})`);
        return 2;
    }

    // What the faulty server answers each model with, how many requests each
    // model got (for the n-th-request faults), and the bodies it was sent.
    const faults = new Map(CASES.map((c) => [modelOf(c.id), c.fault]));
    const seen = new Map();
    const bodies = new Map();
    const faulty = await startFakeLlmServer({
        fault: (body) => {
            const n = (seen.get(body.model) || 0) + 1;
            seen.set(body.model, n);
            bodies.set(body.model, [...(bodies.get(body.model) || []), body]);
            const fault = faults.get(body.model) ?? null;
            return typeof fault === 'function' ? fault(n) : fault;
        },
    });
    let own = null;
    if (!process.env.EVAL_FAKE_LLM_URL) {
        own = await startFakeLlmServer();
        process.env.EVAL_FAKE_LLM_URL = own.url;
    }

    registerTs();
    countTokens();
    const { encrypt } = require('../../libs/common/utils/crypto.ts');
    const { LLM } = require('../../libs/llm/llm.ts');
    const { LLM_TASK } = require('../../libs/llm/byok-config.ts');
    const { resolveTaskSlot } = require('../../libs/llm/resolve-task-model.ts');
    const { recordModelFailovers, readFailoverHops } = require('../../libs/llm/model-failover.ts');
    const { circuitKeyFor, providerCircuit } = require('../../libs/llm/provider-circuit.ts');
    const { isCallerAbortOrBudget } = require('../../libs/llm/error-classifier.ts');
    const { AiSdkAgentRunner } = require('../../libs/agent-harness/infrastructure/ai-sdk/ai-sdk-agent-runner.ts');
    const { InMemoryToolRegistry } = require('../../libs/agent-harness/infrastructure/tools/in-memory-tool-registry.ts');
    const { buildFinderAgentSpec, extractFindings } = require('../../libs/code-review/infrastructure/agents/core/finder.agent.ts');
    const { tool } = require('ai');
    const { z } = require('zod');

    // The org's stored BYOK config, as getBYOKConfig hands it to routing: the
    // two fake servers as two openai_compatible credentials, one primary model
    // per case on the faulty one, the fallback on the healthy one (the eval-fake
    // route's server). Neither checks the key. Encrypted once per credential:
    // the breaker keys on the ciphertext.
    const credentials = [
        { id: 'faulty', provider: 'openai_compatible', apiKey: encrypt('fake'), settings: { baseURL: faulty.url } },
        { id: 'healthy', provider: 'openai_compatible', apiKey: encrypt('fake'), settings: { baseURL: TIER0[MODEL].baseURL } },
    ];
    // `healthy: false` leaves the eval-fake fallback out, so the chain is only
    // the faulty server's models.
    const configFor = (id, chain = [], { healthy = true } = {}) => {
        const tail = [...chain.map((entry) => entry.id), ...(healthy ? [FALLBACK_ID] : [])];
        return {
            version: 2,
            credentials,
            models: [
                { id, credentialId: 'faulty', model: modelOf(id) },
                ...chain.map((entry) => ({ id: entry.id, credentialId: 'faulty', model: modelOf(entry.id) })),
                { id: FALLBACK_ID, credentialId: 'healthy', model: MODEL },
            ],
            routing: { defaultModelId: id, fallbackModelId: tail[0], fallbackModelIds: tail },
        };
    };

    // One exploration tool: the scripted model calls it, then answers, so a
    // completed call is a two-step loop.
    const loop = {
        tools: { readFile: tool({ description: 'Read a file', inputSchema: z.object({ path: z.string() }), execute: async () => 'file body' }) },
        maxSteps: 4,
    };

    // A finder run as the review runs it: the real finder spec (reportFinding,
    // submitResult, the policies, the hand-over) on the real runner, routed to
    // the case's primary. The coverage ledger is a stub with a fixed debt note,
    // so the hand-over's "still unread" part is a known string.
    const UNREAD = 'WARM pending (1/1): readFile src/app.ts lines 10-20';
    const ledger = {
        markFromToolCall: () => undefined,
        summary: () => ({ totalTargets: 1, pendingTargets: 1, criticalTotal: 0, criticalPending: 0 }),
        debtNote: () => UNREAD,
    };
    const finderTools = new InMemoryToolRegistry([
        { name: 'readFile', description: 'Read a file', inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] }, execute: async () => ({ output: 'file body' }) },
    ]);
    const REVIEW_MAX_STEPS = 6;

    const ctx = {
        DEADLINE_MS,
        OPEN_MS,
        UNREAD,
        REVIEW_MAX_STEPS,
        sleep,
        bodiesOf: (id) => bodies.get(modelOf(id)) || [],
        requestsOf: (id) => seen.get(modelOf(id)) || 0,
        review: async (id, { chain = [], env = {}, healthy = true } = {}) => {
            for (const entry of chain) faults.set(modelOf(entry.id), entry.fault);
            const saved = Object.fromEntries(Object.keys(env).map((k) => [k, process.env[k]]));
            Object.assign(process.env, env);
            const before = seen.get(modelOf(id)) || 0;
            const started = Date.now();
            try {
                const { slot } = resolveTaskSlot(configFor(id, chain, { healthy }), LLM_TASK.codeReview);
                const runner = new AiSdkAgentRunner(slot, { organizationId: ORG });
                const spec = buildFinderAgentSpec({ systemPrompt: 'Find bugs in the changed code.', modelId: slot.model, tools: finderTools, coverageLedger: ledger, maxSteps: REVIEW_MAX_STEPS });
                const { value: state, failovers } = await recordModelFailovers(() =>
                    runner.run(spec, { prompt: 'Review src/app.ts' }, { runId: `failover-eval.${id}`, signal: AbortSignal.timeout(REVIEW_BUDGET_MS) }),
                );
                return {
                    state,
                    findings: extractFindings(state).suggestions,
                    hops: failovers[0]?.hops ?? [],
                    answeredBy: failovers[0]?.usedModel ?? (state.status === 'error' ? null : 'primary'),
                    ms: Date.now() - started,
                    primaryRequests: (seen.get(modelOf(id)) || 0) - before,
                };
            } finally {
                for (const [k, v] of Object.entries(saved)) {
                    if (v === undefined) delete process.env[k];
                    else process.env[k] = v;
                }
            }
        },
        setFault: (id, fault) => faults.set(modelOf(id), fault),
        circuitState: (id) => {
            const { slot } = resolveTaskSlot(configFor(id), LLM_TASK.codeReview);
            return providerCircuit.stateOf(circuitKeyFor(slot, ORG));
        },
        call: async (id, { signal, chain = [] } = {}) => {
            for (const entry of chain) faults.set(modelOf(entry.id), entry.fault);
            const before = seen.get(modelOf(id)) || 0;
            const started = Date.now();
            const out = { value: null, error: null, failovers: [], hops: [], answeredBy: null, callerAbort: false };
            try {
                const { value, failovers } = await recordModelFailovers(() =>
                    LLM.run({
                        config: configFor(id, chain),
                        task: LLM_TASK.codeReview,
                        messages: [{ role: 'user', content: 'Review src/app.ts' }],
                        loop,
                        runName: `failover-eval.${id}`,
                        organizationId: ORG,
                        signal,
                    }),
                );
                out.value = value;
                out.failovers = failovers;
                out.hops = failovers[0]?.hops ?? [];
                out.answeredBy = !failovers.length ? 'primary' : failovers[0].usedModel === MODEL ? 'fallback' : failovers[0].usedModel;
            } catch (error) {
                out.error = error;
                out.hops = readFailoverHops(error) ?? [];
                out.callerAbort = isCallerAbortOrBudget(error);
            }
            out.ms = Date.now() - started;
            out.primaryRequests = (seen.get(modelOf(id)) || 0) - before;
            return out;
        },
    };

    const results = [];
    const measures = {};
    let errored = 0;
    for (const testCase of cases) {
        try {
            const outcome = await testCase.run(ctx);
            const detail = Array.isArray(outcome) ? outcome : outcome.detail;
            Object.assign(measures, Array.isArray(outcome) ? {} : outcome.measures);
            const hits = detail.filter((d) => d.ok).length;
            results.push({ id: testCase.id, checks: testCase.checks, hits, total: detail.length, detail });
            console.log(`${hits === detail.length ? '✅' : '❌'} ${testCase.id.padEnd(24)} ${hits}/${detail.length}`);
            for (const d of detail.filter((x) => !x.ok)) console.log(`     ✗ ${d.what}`);
        } catch (error) {
            errored += 1;
            console.error(`  ${testCase.id}: ${String(error.stack || error).slice(0, 400)}`);
            results.push({ id: testCase.id, checks: testCase.checks, hits: 0, total: 0, unmeasured: true });
        }
    }
    await faulty.close();
    if (own) await own.close();

    const hits = results.reduce((n, r) => n + r.hits, 0);
    const total = results.reduce((n, r) => n + r.total, 0);
    const result = {
        eval: 'failover',
        question: evalQuestion('evals/failover'),
        model: MODEL,
        runs: 1,
        metric: { name: 'of failover checks right', value: total ? hits / total : null, floor: 1, baseline: null },
        cases: results,
        // The README's metrics, from the cases that measure them.
        measures,
        tokens: tokensUsed(),
        infra: { errored, total: cases.length },
    };
    if (args.output) {
        fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });
        fs.writeFileSync(args.output, `${JSON.stringify(result, null, 2)}\n`);
    }
    console.log(`${result.eval}: ${hits}/${total} checks · ${errored} of ${cases.length} cases did not measure`);
    if (Object.keys(measures).length) console.log(`   measures: ${Object.entries(measures).map(([k, v]) => `${k}=${v}`).join(' · ')}`);
    // Deterministic: a case that throws is a broken harness, not a measurement.
    if (errored) {
        console.error(`not measured: ${errored} of ${cases.length} cases failed to run (see above)`);
        return 2;
    }
    const problems = resultProblems(result);
    if (problems.length) {
        console.error(`not measured: the result breaks the shared shape — ${problems.join('; ')}`);
        return 2;
    }
    if (args.gate && hits < total) return 1;
    return 0;
}

main()
    .then((code) => process.exit(code))
    .catch((error) => {
        console.error(error);
        process.exit(2);
    });
