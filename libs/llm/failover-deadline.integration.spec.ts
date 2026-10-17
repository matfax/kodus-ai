/**
 * failover-deadline.integration.spec.ts — the round-trip deadline end to end:
 * `LLM.run` → failover → the real executors (agent loop and one-shot) → the
 * real BYOK wrapper → a stalling model. Only the model build is replaced (each
 * slot gets a scripted mock model) and the logger is silenced; the SDK's own
 * `generateText`, its retries and the limiter all run for real, on real timers
 * with a short deadline.
 *
 * The rules pinned here are the ones that keep the deadline from doing harm:
 * it is armed only while a cascade is possible — a next chain entry exists
 * and no loop step has run — so a single entry and a loop past its first step
 * behave exactly as before.
 */
jest.mock('@libs/core/log/logger', () => ({
    createLogger: () => ({
        warn: jest.fn(),
        log: jest.fn(),
        info: jest.fn(),
        error: jest.fn(),
        debug: jest.fn(),
    }),
}));

const models = new Map<string, unknown>();
jest.mock('@libs/llm/byok-to-vercel', () => ({
    ...jest.requireActual('@libs/llm/byok-to-vercel'),
    buildModelFromSlot: (slot: { model: string } | undefined) => {
        const model = slot && models.get(slot.model);
        if (!model) throw new Error(`no scripted model for ${slot?.model}`);
        return model;
    },
}));

import { APICallError } from '@ai-sdk/provider';
import { tool } from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import { z } from 'zod';

import type { NormalizedModel } from './byok-config';
import { LLM } from './llm';
import { readFailoverHops, recordModelFailovers } from './model-failover';
import { circuitKeyFor, providerCircuit } from './provider-circuit';
import { ProviderDeadlineError } from './provider-deadline-error';

const DEADLINE_MS = 300;
const ORG = 'org-deadline';

type Plan =
    | { kind: 'text'; text: string; delayMs?: number }
    | { kind: 'tool'; delayMs?: number }
    | { kind: 'hang'; ignoresAbort?: boolean }
    | { kind: 'error'; error: Error; delayMs?: number };

/** A model that answers its n-th request by `plans[n]` (the last repeats). */
function scripted(name: string, plans: Plan[]) {
    const signals: Array<AbortSignal | undefined> = [];
    let n = 0;
    const model = new MockLanguageModelV4({
        provider: 'test',
        modelId: name,
        doGenerate: (async (options: { abortSignal?: AbortSignal }) => {
            signals.push(options.abortSignal);
            const plan = plans[Math.min(n++, plans.length - 1)];
            const wait = (ms = 0) =>
                new Promise<void>((resolve, reject) => {
                    const t = setTimeout(resolve, ms);
                    options.abortSignal?.addEventListener('abort', () => {
                        clearTimeout(t);
                        reject(
                            new DOMException(
                                'This operation was aborted',
                                'AbortError',
                            ),
                        );
                    });
                });
            if (plan.kind === 'hang') {
                if (plan.ignoresAbort) return new Promise(() => undefined);
                await wait(60_000);
            }
            if (plan.kind === 'error') {
                await wait(plan.delayMs);
                throw plan.error;
            }
            if (plan.kind === 'tool') {
                await wait(plan.delayMs);
                return {
                    content: [
                        {
                            type: 'tool-call',
                            toolCallId: `call-${n}`,
                            toolName: 'ping',
                            input: '{}',
                        },
                    ],
                    finishReason: { unified: 'tool-calls', raw: 'tool_calls' },
                    usage: {
                        inputTokens: { total: 1 },
                        outputTokens: { total: 1 },
                    },
                    warnings: [],
                };
            }
            const text = plan as { text: string; delayMs?: number };
            await wait(text.delayMs);
            return {
                content: [{ type: 'text', text: text.text }],
                finishReason: { unified: 'stop', raw: 'stop' },
                usage: {
                    inputTokens: { total: 1 },
                    outputTokens: { total: 1 },
                },
                warnings: [],
            };
        }) as any,
    });
    models.set(name, model);
    return { model, signals, calls: () => n };
}

const slot = (model: string): NormalizedModel =>
    ({
        provider: 'openai',
        apiKey: `cipher-${model}`,
        model,
        byokModelId: model,
    }) as NormalizedModel;

const withFallback = (primary: string, fallback: string): NormalizedModel => ({
    ...slot(primary),
    fallbacks: [{ ...slot(fallback), chainIndex: 0, usedFallback: true }],
});

const loopTools = {
    ping: tool({
        inputSchema: z.object({}),
        execute: async () => 'pong',
    }),
};

const runLoop = (byokConfig: NormalizedModel, maxSteps = 5) =>
    LLM.run({
        byokConfig,
        messages: [{ role: 'user', content: 'review this' }],
        loop: { tools: loopTools, maxSteps },
        runName: 'deadline.loop',
        organizationId: ORG,
    });

const transient503 = () =>
    new APICallError({
        message: 'Service Unavailable',
        url: 'https://provider.test/v1',
        requestBodyValues: {},
        statusCode: 503,
        isRetryable: true,
    });

describe('the failover round-trip deadline, end to end', () => {
    const env = { ...process.env };
    beforeEach(() => {
        models.clear();
        providerCircuit.reset();
        process.env.LLM_FAILOVER_ROUND_TRIP_DEADLINE_MS = String(DEADLINE_MS);
    });
    afterEach(() => {
        process.env = { ...env };
        providerCircuit.reset();
    });

    describe('agent loop', () => {
        it('cascades a hung first step to the next entry within the deadline', async () => {
            const primary = scripted('loop-hung', [{ kind: 'hang' }]);
            scripted('loop-fallback', [{ kind: 'text', text: 'done' }]);

            const started = Date.now();
            const { value, failovers } = await recordModelFailovers(() =>
                runLoop(withFallback('loop-hung', 'loop-fallback')),
            );
            expect(value.text).toBe('done');
            expect(Date.now() - started).toBeLessThan(DEADLINE_MS + 2_000);
            expect(failovers[0].hops[0]).toMatchObject({
                model: 'loop-hung',
                reason: 'deadline',
            });
            // Not retried on the same model: the deadline is not an
            // APICallError, so the SDK gives it straight back.
            expect(primary.calls()).toBe(1);
        }, 15_000);

        it('arms nothing after the first step: a stall at step 3 is SDK-retried and the loop completes', async () => {
            const primary = scripted('loop-steady', [
                { kind: 'tool' },
                { kind: 'tool' },
                // Longer than the deadline, then a retryable error: today's
                // behaviour is an SDK retry on the same model.
                {
                    kind: 'error',
                    error: transient503(),
                    delayMs: 2 * DEADLINE_MS,
                },
                { kind: 'text', text: 'done' },
            ]);
            const fallback = scripted('loop-unused', [
                { kind: 'text', text: 'fallback' },
            ]);

            const result = await runLoop(
                withFallback('loop-steady', 'loop-unused'),
            );
            expect(result.text).toBe('done');
            expect(primary.calls()).toBe(4);
            expect(fallback.calls()).toBe(0);
        }, 15_000);

        it('leaves a single entry exactly as before: no deadline at all', async () => {
            scripted('loop-alone', [
                {
                    kind: 'text',
                    text: 'slow but fine',
                    delayMs: 2 * DEADLINE_MS,
                },
            ]);
            const result = await runLoop(slot('loop-alone'));
            expect(result.text).toBe('slow but fine');
        }, 15_000);
    });

    describe('one-shot', () => {
        it("cascades at the caller's shorter budget, and the executor's timer does not win", async () => {
            // The knob (5 min by default) is far above this caller's budget:
            // the deadline takes the budget, as `caller-budget`.
            delete process.env.LLM_FAILOVER_ROUND_TRIP_DEADLINE_MS;
            process.env.LLM_CIRCUIT_FAILURE_THRESHOLD = '1';
            scripted('shot-hung', [{ kind: 'hang' }]);
            scripted('shot-fallback', [{ kind: 'text', text: 'answer' }]);

            const started = Date.now();
            const { value, failovers } = await recordModelFailovers(() =>
                LLM.run({
                    byokConfig: withFallback('shot-hung', 'shot-fallback'),
                    user: 'classify this',
                    runName: 'deadline.oneshot',
                    organizationId: ORG,
                    timeoutMs: DEADLINE_MS,
                }),
            );
            expect(value).toBe('answer');
            expect(Date.now() - started).toBeLessThan(DEADLINE_MS + 2_000);
            expect(failovers[0].hops[0].reason).toBe('deadline');
            // A caller-budget deadline cascades but says nothing about the
            // provider: even at a threshold of one, the breaker stays closed.
            expect(
                providerCircuit.stateOf(circuitKeyFor(slot('shot-hung'), ORG)!),
            ).toBe('closed');
        }, 15_000);

        it('settles a primary that ignores the abort after the grace, then cascades', async () => {
            scripted('shot-deaf', [{ kind: 'hang', ignoresAbort: true }]);
            scripted('shot-fallback-2', [{ kind: 'text', text: 'answer' }]);

            const started = Date.now();
            const { value, failovers } = await recordModelFailovers(() =>
                LLM.run({
                    byokConfig: withFallback('shot-deaf', 'shot-fallback-2'),
                    user: 'classify this',
                    runName: 'deadline.oneshot.deaf',
                    organizationId: ORG,
                }),
            );
            expect(value).toBe('answer');
            const elapsed = Date.now() - started;
            expect(elapsed).toBeGreaterThanOrEqual(DEADLINE_MS + 5_000 - 50);
            expect(elapsed).toBeLessThan(DEADLINE_MS + 5_000 + 2_000);
            expect(failovers[0].hops[0].reason).toBe('deadline');
        }, 20_000);

        it('keeps the last entry free of any deadline', async () => {
            scripted('shot-alone', [
                { kind: 'text', text: 'late answer', delayMs: 2 * DEADLINE_MS },
            ]);
            await expect(
                LLM.run({
                    byokConfig: slot('shot-alone'),
                    user: 'classify this',
                    runName: 'deadline.oneshot.alone',
                    organizationId: ORG,
                }),
            ).resolves.toBe('late answer');
        }, 15_000);

        it("throws the fallback's own failure when it fails too, with the deadline in the hops", async () => {
            scripted('shot-hung-2', [{ kind: 'hang' }]);
            scripted('shot-broken', [
                {
                    kind: 'error',
                    error: Object.assign(new Error('unauthorized'), {
                        statusCode: 401,
                    }),
                },
            ]);
            const thrown = await LLM.run({
                byokConfig: withFallback('shot-hung-2', 'shot-broken'),
                user: 'classify this',
                runName: 'deadline.oneshot.both',
                organizationId: ORG,
            }).catch((e) => e);
            // The terminal error is the last entry's; the first entry's
            // deadline is in the hops, not lost.
            expect(thrown).not.toBeInstanceOf(ProviderDeadlineError);
            expect((thrown as Error).message).toContain('unauthorized');
            expect(readFailoverHops(thrown)?.[0]).toMatchObject({
                model: 'shot-hung-2',
                reason: 'deadline',
            });
        }, 15_000);
    });
});
