/**
 * model-failover.spec.ts — the runtime primary→fallback cascade primitive.
 *
 * The error classifier is MOCKED so these tests pin the primitive's OWN logic
 * (which category cascades, the loop, the unsafe-to-retry veto) independent of
 * the status→category mapping, which `error-classifier.spec.ts` owns. Categories
 * are carried on the thrown error as `__cat`, a caller abort as `__abort` and a
 * provider timeout as `__providerTimeout`. `__legacyAbort` drives the prose gate
 * that `LLM_PROVIDER_TIMEOUT_CASCADE=off` restores.
 */
// One stable spy per module load (the factory closes over it and `createLogger`
// always returns it) so tests can assert the [LLM-ERROR]/[LLM-SUCCESS]
// observability lines this primitive emits. Fetched below via `createLogger`
// rather than an outer const — ESM import hoisting would load model-failover
// (and its module-level createLogger call) before an outer const initialized.
jest.mock('@libs/core/log/logger', () => {
    const log = {
        warn: jest.fn(),
        log: jest.fn(),
        info: jest.fn(),
        error: jest.fn(),
        debug: jest.fn(),
    };
    return { createLogger: () => log };
});

jest.mock('@libs/llm/error-classifier', () => {
    const LlmErrorCategory = {
        AUTH_INVALID: 'AUTH_INVALID',
        QUOTA_EXCEEDED: 'QUOTA_EXCEEDED',
        RATE_LIMIT: 'RATE_LIMIT',
        MODEL_NOT_FOUND: 'MODEL_NOT_FOUND',
        MODEL_ACCESS_DENIED: 'MODEL_ACCESS_DENIED',
        CONTEXT_OVERFLOW: 'CONTEXT_OVERFLOW',
        TRANSIENT: 'TRANSIENT',
        UNKNOWN: 'UNKNOWN',
    };
    const TERMINAL = new Set([
        'AUTH_INVALID',
        'QUOTA_EXCEEDED',
        'MODEL_NOT_FOUND',
        'MODEL_ACCESS_DENIED',
    ]);
    return {
        LlmErrorCategory,
        classifyLLMError: (e: any) => ({ category: e?.__cat ?? 'UNKNOWN' }),
        isTerminalCategory: (c: string) => TERMINAL.has(c),
        isAbortOrHardTimeout: (e: any) =>
            e?.__abort === true || e?.__legacyAbort === true,
        isCallerAbortOrBudget: (e: any) => e?.__abort === true,
        isProviderTimeout: (e: any) => e?.__providerTimeout === true,
        retriesWereExhausted: (e: any) => e?.__exhausted === true,
        // The breaker reads the chain for our deadline; one layer is enough.
        failureChain: (e: any) => (e && typeof e === 'object' ? [e] : []),
    };
});

import { ProviderDeadlineError } from './provider-deadline-error';
import {
    providerTimeoutCascadeEnabled,
    readAttemptedSlot,
    readFailoverHops,
    recordModelFailovers,
    runWithModelFailover,
    shouldFailoverToNextModel,
    type FailoverAttemptControl,
} from './model-failover';
import type { NormalizedModel } from './byok-config';
import { providerCircuit } from './provider-circuit';
import { createLogger } from '@libs/core/log/logger';

// The mocked createLogger always returns the same spy instance model-failover
// captured at module load, so this is that exact instance.
const mockLog = createLogger('test') as unknown as {
    warn: jest.Mock;
    log: jest.Mock;
    info: jest.Mock;
    error: jest.Mock;
    debug: jest.Mock;
};

const err = (cat: string, extra: Record<string, unknown> = {}) =>
    Object.assign(new Error(`boom:${cat}`), { __cat: cat, ...extra });

const slot = (id: string): NormalizedModel =>
    ({ model: id, byokModelId: id, provider: 'openai', apiKey: 'enc' }) as any;

describe('shouldFailoverToNextModel', () => {
    it('cascades on terminal model-specific failures', () => {
        for (const cat of [
            'AUTH_INVALID',
            'QUOTA_EXCEEDED',
            'MODEL_NOT_FOUND',
            'MODEL_ACCESS_DENIED',
        ]) {
            expect(shouldFailoverToNextModel(err(cat))).toBe(true);
        }
    });

    it('cascades on a persistent transient blip', () => {
        expect(shouldFailoverToNextModel(err('TRANSIENT'))).toBe(true);
    });

    it('does NOT cascade on rate-limit, context-overflow, or unknown', () => {
        expect(shouldFailoverToNextModel(err('RATE_LIMIT'))).toBe(false);
        expect(shouldFailoverToNextModel(err('CONTEXT_OVERFLOW'))).toBe(false);
        expect(shouldFailoverToNextModel(err('UNKNOWN'))).toBe(false);
    });

    it('DOES cascade on a rate-limit that outlived its same-model retries', () => {
        // The limiter owns a 429 it can still wait out; it does not own one that
        // already survived exponential backoff. See retriesWereExhausted.
        expect(
            shouldFailoverToNextModel(err('RATE_LIMIT', { __exhausted: true })),
        ).toBe(true);
    });

    it('keeps context-overflow and unknown refused even when exhausted', () => {
        // Exhaustion says the failure is PERSISTENT, not that a peer model can
        // fix it: the prompt is still too big, and unknown is still unknown.
        expect(
            shouldFailoverToNextModel(
                err('CONTEXT_OVERFLOW', { __exhausted: true }),
            ),
        ).toBe(false);
        expect(
            shouldFailoverToNextModel(err('UNKNOWN', { __exhausted: true })),
        ).toBe(false);
    });

    it('does NOT cascade on abort / hard-timeout even if otherwise transient', () => {
        expect(
            shouldFailoverToNextModel(err('TRANSIENT', { __abort: true })),
        ).toBe(false);
    });

    it('cascades on a provider timeout whatever its category reads', () => {
        // A 504/408/undici timeout is the route being down, not the budget.
        expect(
            shouldFailoverToNextModel(
                err('UNKNOWN', { __providerTimeout: true }),
            ),
        ).toBe(true);
    });

    it('lets a caller abort win over a provider timeout it wraps', () => {
        expect(
            shouldFailoverToNextModel(
                err('TRANSIENT', { __abort: true, __providerTimeout: true }),
            ),
        ).toBe(false);
    });

    it('does not read the prose gate while the cascade is on', () => {
        // The old regex matched "Gateway Timeout"; it no longer vetoes.
        expect(
            shouldFailoverToNextModel(
                err('TRANSIENT', { __legacyAbort: true }),
            ),
        ).toBe(true);
    });

    describe('LLM_PROVIDER_TIMEOUT_CASCADE=off', () => {
        const saved = process.env.LLM_PROVIDER_TIMEOUT_CASCADE;
        beforeEach(() => {
            process.env.LLM_PROVIDER_TIMEOUT_CASCADE = 'off';
        });
        afterEach(() => {
            if (saved === undefined) {
                delete process.env.LLM_PROVIDER_TIMEOUT_CASCADE;
            } else {
                process.env.LLM_PROVIDER_TIMEOUT_CASCADE = saved;
            }
        });

        it('restores the prose gate and ignores the provider-timeout predicate', () => {
            expect(
                shouldFailoverToNextModel(
                    err('TRANSIENT', { __legacyAbort: true }),
                ),
            ).toBe(false);
            expect(
                shouldFailoverToNextModel(
                    err('UNKNOWN', { __providerTimeout: true }),
                ),
            ).toBe(false);
            expect(shouldFailoverToNextModel(err('TRANSIENT'))).toBe(true);
        });
    });
});

describe('providerTimeoutCascadeEnabled', () => {
    it.each([
        [undefined, true],
        ['', true],
        ['on', true],
        ['true', true],
        ['off', false],
        [' OFF ', false],
        ['false', false],
        ['0', false],
    ])('LLM_PROVIDER_TIMEOUT_CASCADE=%p → %p', (value, expected) => {
        expect(
            providerTimeoutCascadeEnabled({
                LLM_PROVIDER_TIMEOUT_CASCADE: value,
            }),
        ).toBe(expected);
    });
});

describe('runWithModelFailover', () => {
    const opts = { runName: 'test' };

    it('runs the primary once and returns it on success (fallback untouched)', async () => {
        const runOne = jest.fn().mockResolvedValue('ok');
        const out = await runWithModelFailover(
            [slot('A'), slot('B')],
            runOne,
            opts,
        );
        expect(out).toBe('ok');
        expect(runOne).toHaveBeenCalledTimes(1);
        expect(runOne.mock.calls[0][0]).toMatchObject({ model: 'A' });
    });

    it('cascades to the fallback on a terminal failure and returns it', async () => {
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('AUTH_INVALID'))
            .mockResolvedValueOnce('from-fallback');
        const out = await runWithModelFailover(
            [slot('A'), slot('B')],
            runOne,
            opts,
        );
        expect(out).toBe('from-fallback');
        expect(runOne).toHaveBeenCalledTimes(2);
        expect(runOne.mock.calls[1][0]).toMatchObject({ model: 'B' });
    });

    it('does NOT cascade on a non-cascade-worthy error (rate-limit)', async () => {
        const runOne = jest.fn().mockRejectedValue(err('RATE_LIMIT'));
        await expect(
            runWithModelFailover([slot('A'), slot('B')], runOne, opts),
        ).rejects.toThrow('boom:RATE_LIMIT');
        expect(runOne).toHaveBeenCalledTimes(1);
    });

    it('does NOT cascade when the attempt marked itself unsafe to retry', async () => {
        const runOne = jest.fn(
            async (_s: unknown, c: FailoverAttemptControl) => {
                c.markUnsafeToRetry();
                throw err('AUTH_INVALID');
            },
        );
        await expect(
            runWithModelFailover([slot('A'), slot('B')], runOne, opts),
        ).rejects.toThrow('boom:AUTH_INVALID');
        expect(runOne).toHaveBeenCalledTimes(1);
    });

    it('tells each attempt whether a later entry will run (hasNext)', async () => {
        const seen: boolean[] = [];
        const runOne = jest.fn(
            async (_s: unknown, c: FailoverAttemptControl) => {
                seen.push(c.hasNext);
                throw err('TRANSIENT');
            },
        );
        await expect(
            runWithModelFailover([slot('A'), slot('B')], runOne, opts),
        ).rejects.toThrow('boom:TRANSIENT');
        expect(seen).toEqual([true, false]);

        // A fallback deduped away is no next entry.
        seen.length = 0;
        await expect(
            runWithModelFailover([slot('A'), slot('A')], runOne, opts),
        ).rejects.toThrow('boom:TRANSIENT');
        expect(seen).toEqual([false]);
    });

    it('re-throws the LAST error when both the primary and fallback fail', async () => {
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('TRANSIENT'))
            .mockRejectedValueOnce(err('AUTH_INVALID'));
        await expect(
            runWithModelFailover([slot('A'), slot('B')], runOne, opts),
        ).rejects.toThrow('boom:AUTH_INVALID');
        expect(runOne).toHaveBeenCalledTimes(2);
    });

    it('is a pass-through with a single slot (no fallback to cascade to)', async () => {
        const runOne = jest.fn().mockRejectedValue(err('AUTH_INVALID'));
        await expect(
            runWithModelFailover([slot('A')], runOne, opts),
        ).rejects.toThrow('boom:AUTH_INVALID');
        expect(runOne).toHaveBeenCalledTimes(1);
    });

    it('dedupes a fallback identical to the primary (one attempt)', async () => {
        const runOne = jest.fn().mockRejectedValue(err('AUTH_INVALID'));
        await expect(
            runWithModelFailover([slot('A'), slot('A')], runOne, opts),
        ).rejects.toThrow();
        expect(runOne).toHaveBeenCalledTimes(1);
    });

    it('dedupes by model name when a slot has no byokModelId', async () => {
        const named = (m: string): NormalizedModel =>
            ({ model: m, provider: 'openai', apiKey: 'enc' }) as any;
        const runOne = jest.fn().mockRejectedValue(err('AUTH_INVALID'));
        await expect(
            runWithModelFailover([named('X'), named('X')], runOne, opts),
        ).rejects.toThrow();
        expect(runOne).toHaveBeenCalledTimes(1);
    });

    it('runs the managed-default (undefined) attempt exactly once', async () => {
        const runOne = jest.fn().mockResolvedValue('managed');
        const out = await runWithModelFailover([undefined], runOne, opts);
        expect(out).toBe('managed');
        expect(runOne).toHaveBeenCalledTimes(1);
        expect(runOne.mock.calls[0][0]).toBeUndefined();
    });

    it('drops an `undefined` that trails a real slot (never queued as a fallback)', async () => {
        // distinctAttempts pushes the managed-default (undefined) ONLY as the sole
        // entry — an undefined AFTER a real slot must be dropped, so a terminal
        // primary failure does NOT cascade into a bogus managed-default attempt.
        const runOne = jest.fn().mockRejectedValue(err('AUTH_INVALID'));
        await expect(
            runWithModelFailover([slot('A'), undefined], runOne, opts),
        ).rejects.toThrow();
        expect(runOne).toHaveBeenCalledTimes(1);
        expect(runOne.mock.calls[0][0]).toMatchObject({ model: 'A' });
    });
});

describe('runWithModelFailover — an N-entry chain', () => {
    const opts = { runName: 'chain' };

    it('walks every entry in order until one answers', async () => {
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('AUTH_INVALID'))
            .mockRejectedValueOnce(err('TRANSIENT'))
            .mockRejectedValueOnce(err('QUOTA_EXCEEDED'))
            .mockResolvedValueOnce('from-D');
        const out = await runWithModelFailover(
            [slot('A'), slot('B'), slot('C'), slot('D')],
            runOne,
            opts,
        );
        expect(out).toBe('from-D');
        expect(runOne.mock.calls.map(([s]) => s.model)).toEqual([
            'A',
            'B',
            'C',
            'D',
        ]);
    });

    it('stops at the first error that must not cascade, mid-chain', async () => {
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('TRANSIENT'))
            .mockRejectedValueOnce(err('CONTEXT_OVERFLOW'));
        await expect(
            runWithModelFailover(
                [slot('A'), slot('B'), slot('C')],
                runOne,
                opts,
            ),
        ).rejects.toThrow('boom:CONTEXT_OVERFLOW');
        expect(runOne).toHaveBeenCalledTimes(2);
    });

    it('tells every entry but the last that a next one exists', async () => {
        const seen: boolean[] = [];
        const runOne = jest.fn(
            async (_s: unknown, c: FailoverAttemptControl) => {
                seen.push(c.hasNext);
                throw err('TRANSIENT');
            },
        );
        await expect(
            runWithModelFailover(
                [slot('A'), slot('B'), slot('C')],
                runOne,
                opts,
            ),
        ).rejects.toThrow();
        expect(seen).toEqual([true, true, false]);
    });

    it('lets markUnsafeToRetry veto the cascade on a middle entry, and isUnsafe reports it', async () => {
        const unsafe: boolean[] = [];
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('TRANSIENT'))
            .mockImplementationOnce(
                async (_s: unknown, c: FailoverAttemptControl) => {
                    unsafe.push(c.isUnsafe());
                    c.markUnsafeToRetry();
                    unsafe.push(c.isUnsafe());
                    throw err('TRANSIENT');
                },
            );
        await expect(
            runWithModelFailover(
                [slot('A'), slot('B'), slot('C')],
                runOne,
                opts,
            ),
        ).rejects.toThrow('boom:TRANSIENT');
        expect(runOne).toHaveBeenCalledTimes(2);
        expect(unsafe).toEqual([false, true]);
    });

    it('does not start the next entry once the caller signal has aborted', async () => {
        const controller = new AbortController();
        const runOne = jest.fn(async () => {
            controller.abort();
            throw err('TRANSIENT');
        });
        await expect(
            runWithModelFailover([slot('A'), slot('B'), slot('C')], runOne, {
                ...opts,
                signal: controller.signal,
            }),
        ).rejects.toThrow('boom:TRANSIENT');
        expect(runOne).toHaveBeenCalledTimes(1);
    });

    it('stops between later hops too when the caller aborts', async () => {
        const controller = new AbortController();
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('TRANSIENT'))
            .mockImplementationOnce(async () => {
                controller.abort();
                throw err('TRANSIENT');
            });
        const thrown = await runWithModelFailover(
            [slot('A'), slot('B'), slot('C')],
            runOne,
            { ...opts, signal: controller.signal },
        ).catch((e) => e);
        expect(runOne).toHaveBeenCalledTimes(2);
        expect(readAttemptedSlot(thrown)).toMatchObject({ model: 'B' });
    });

    it('records the hops before the entry that answered', async () => {
        const { failovers } = await recordModelFailovers(() =>
            runWithModelFailover(
                [slot('A'), slot('B'), slot('C')],
                jest
                    .fn()
                    .mockRejectedValueOnce(err('AUTH_INVALID'))
                    .mockRejectedValueOnce(
                        err('TRANSIENT', { __providerTimeout: true }),
                    )
                    .mockResolvedValueOnce('ok'),
                opts,
            ),
        );
        expect(failovers).toHaveLength(1);
        expect(failovers[0]).toMatchObject({
            failedModel: 'A',
            usedModel: 'C',
            chainIndex: 2,
        });
        expect(
            failovers[0].hops.map((h) => [h.model, h.chainIndex, h.reason]),
        ).toEqual([
            ['A', 0, 'AUTH_INVALID'],
            ['B', 1, 'provider-timeout'],
        ]);
    });

    it('names a round-trip deadline as its own hop reason', async () => {
        const deadline = new ProviderDeadlineError({
            deadlineMs: 1000,
            source: 'round-trip',
            phase: 'round-trip',
        });
        Object.assign(deadline, {
            __cat: 'TRANSIENT',
            __providerTimeout: true,
        });
        const { failovers } = await recordModelFailovers(() =>
            runWithModelFailover(
                [slot('A'), slot('B')],
                jest
                    .fn()
                    .mockRejectedValueOnce(deadline)
                    .mockResolvedValueOnce('ok'),
                opts,
            ),
        );
        expect(failovers[0].hops[0].reason).toBe('deadline');
    });

    it('stamps every hop on the error after a terminal failure, and lists them in the warn', async () => {
        mockLog.warn.mockClear();
        const thrown = await runWithModelFailover(
            [slot('A'), slot('B'), slot('C')],
            jest
                .fn()
                .mockRejectedValueOnce(err('AUTH_INVALID'))
                .mockRejectedValueOnce(err('TRANSIENT'))
                .mockRejectedValueOnce(err('QUOTA_EXCEEDED')),
            opts,
        ).catch((e) => e);
        expect(readFailoverHops(thrown)?.map((h) => h.model)).toEqual([
            'A',
            'B',
            'C',
        ]);
        expect(Object.keys(thrown)).not.toContain('hops');
        const terminal = mockLog.warn.mock.calls.at(-1)[0];
        expect(terminal.message).toContain(
            '[after: "A" failed (AUTH_INVALID), "B" failed (TRANSIENT)]',
        );
        expect(terminal.metadata.chainIndex).toBe(2);
        expect(terminal.metadata.hops).toHaveLength(3);
    });

    it('carries chainIndex and reason on each cascade warn', async () => {
        mockLog.warn.mockClear();
        await runWithModelFailover(
            [slot('A'), slot('B'), slot('C')],
            jest
                .fn()
                .mockRejectedValueOnce(err('AUTH_INVALID'))
                .mockRejectedValueOnce(err('TRANSIENT'))
                .mockResolvedValueOnce('ok'),
            opts,
        );
        expect(
            mockLog.warn.mock.calls.map(([c]) => [
                c.metadata.chainIndex,
                c.metadata.reason,
                c.metadata.toModelId,
            ]),
        ).toEqual([
            [0, 'AUTH_INVALID', 'B'],
            [1, 'TRANSIENT', 'C'],
        ]);
    });
});

describe('recordModelFailovers (#2066)', () => {
    const opts = { runName: 'code-review-bug' };

    it('records a call that only succeeded on the fallback', async () => {
        const { value, failovers } = await recordModelFailovers(() =>
            runWithModelFailover(
                [slot('A'), slot('B')],
                jest
                    .fn()
                    .mockRejectedValueOnce(err('AUTH_INVALID'))
                    .mockResolvedValueOnce('from-fallback'),
                opts,
            ),
        );

        expect(value).toBe('from-fallback');
        expect(failovers).toEqual([
            {
                runName: 'code-review-bug',
                failedModel: 'A',
                usedModel: 'B',
                chainIndex: 1,
                hops: [
                    {
                        model: 'A',
                        chainIndex: 0,
                        outcome: 'failed',
                        reason: 'AUTH_INVALID',
                        elapsedMs: expect.any(Number),
                    },
                ],
            },
        ]);
    });

    it('records nothing when the primary answered', async () => {
        const { failovers } = await recordModelFailovers(() =>
            runWithModelFailover(
                [slot('A'), slot('B')],
                jest.fn().mockResolvedValue('ok'),
                opts,
            ),
        );

        expect(failovers).toEqual([]);
    });

    it('records nothing when both models failed (the call threw)', async () => {
        const { failovers } = await recordModelFailovers(() =>
            runWithModelFailover(
                [slot('A'), slot('B')],
                jest.fn().mockRejectedValue(err('AUTH_INVALID')),
                opts,
            ).catch(() => 'handled'),
        );

        expect(failovers).toEqual([]);
    });

    it('keeps concurrent runs apart', async () => {
        const failingThenOk = () =>
            jest
                .fn()
                .mockRejectedValueOnce(err('AUTH_INVALID'))
                .mockResolvedValueOnce('ok');
        const [a, b] = await Promise.all([
            recordModelFailovers(() =>
                runWithModelFailover(
                    [slot('A'), slot('B')],
                    failingThenOk(),
                    opts,
                ),
            ),
            recordModelFailovers(() =>
                runWithModelFailover(
                    [slot('C'), slot('D')],
                    jest.fn().mockResolvedValue('ok'),
                    opts,
                ),
            ),
        ]);

        expect(a.failovers).toHaveLength(1);
        expect(b.failovers).toEqual([]);
    });

    it('is a no-op outside a recording scope', async () => {
        await expect(
            runWithModelFailover(
                [slot('A'), slot('B')],
                jest
                    .fn()
                    .mockRejectedValueOnce(err('AUTH_INVALID'))
                    .mockResolvedValueOnce('ok'),
                opts,
            ),
        ).resolves.toBe('ok');
    });
});

describe('runWithModelFailover — observability ([LLM-ERROR]/[LLM-SUCCESS])', () => {
    const opts = { runName: 'test', organizationId: 'org-1' };
    beforeEach(() => {
        mockLog.warn.mockClear();
        mockLog.debug.mockClear();
    });

    it('emits a [LLM-SUCCESS] debug line with the model and usedFallback=false on a primary win', async () => {
        const runOne = jest.fn().mockResolvedValue('ok');
        await runWithModelFailover([slot('A'), slot('B')], runOne, opts);
        expect(mockLog.debug).toHaveBeenCalledTimes(1);
        const call = mockLog.debug.mock.calls[0][0];
        expect(call.message).toContain('[LLM-SUCCESS]');
        expect(call.message).toContain('A');
        expect(call.metadata.usedFallback).toBe(false);
    });

    it('stamps usedFallback=true on the success line when the fallback saved the call', async () => {
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('AUTH_INVALID'))
            .mockResolvedValueOnce('ok');
        await runWithModelFailover([slot('A'), slot('B')], runOne, opts);
        const success = mockLog.debug.mock.calls[0][0];
        expect(success.message).toContain('[LLM-SUCCESS]');
        expect(success.metadata.usedFallback).toBe(true);
    });

    it('names from→to (the i+1 fallback) on the cascade [LLM-ERROR] warn', async () => {
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('AUTH_INVALID'))
            .mockResolvedValueOnce('ok');
        await runWithModelFailover([slot('A'), slot('B')], runOne, opts);
        const cascade = mockLog.warn.mock.calls[0][0];
        expect(cascade.message).toContain('[LLM-ERROR]');
        expect(cascade.message).toContain('A'); // from
        expect(cascade.message).toContain('B'); // to (attempts[i+1])
        expect(cascade.metadata.toModelId).toBe('B');
    });

    it('emits a terminal [LLM-ERROR] warn with the category and exhausted=true when all attempts fail', async () => {
        const runOne = jest.fn().mockRejectedValue(err('AUTH_INVALID'));
        await expect(
            runWithModelFailover([slot('A')], runOne, opts),
        ).rejects.toThrow();
        const terminal = mockLog.warn.mock.calls[0][0];
        expect(terminal.message).toContain('[LLM-ERROR]');
        expect(terminal.metadata.category).toBe('AUTH_INVALID');
        expect(terminal.metadata.exhausted).toBe(true);
    });
});

describe('the failed attempt is recorded ON the error', () => {
    const opts = { runName: 'test' };
    const providerSlot = (model: string, provider: string): NormalizedModel =>
        ({ model, byokModelId: model, provider, apiKey: 'enc' }) as any;

    it('names the FALLBACK when the fallback is what failed', async () => {
        // The whole point. Downstream reporting otherwise re-reads the slot
        // resolved before the run and names the primary — a model that did not
        // produce this failure, asserted as fact, in exactly the case a reader
        // needs the truth: both routes down.
        const runOne = jest
            .fn()
            .mockRejectedValueOnce(err('AUTH_INVALID'))
            .mockRejectedValueOnce(err('AUTH_INVALID'));

        await expect(
            runWithModelFailover(
                [
                    providerSlot('primary-model', 'openai'),
                    providerSlot('fallback-model', 'open_router'),
                ],
                runOne,
                opts,
            ),
        ).rejects.toMatchObject({ __cat: 'AUTH_INVALID' });

        const thrown = await runWithModelFailover(
            [
                providerSlot('primary-model', 'openai'),
                providerSlot('fallback-model', 'open_router'),
            ],
            jest
                .fn()
                .mockRejectedValueOnce(err('AUTH_INVALID'))
                .mockRejectedValueOnce(err('AUTH_INVALID')),
            opts,
        ).catch((e) => e);

        // Model AND provider come from the same attempt: a fallback model
        // reported beside the primary's provider is a pair that never existed.
        expect(readAttemptedSlot(thrown)).toEqual({
            model: 'fallback-model',
            provider: 'open_router',
        });
    });

    it('names the primary when there was no fallback to try', async () => {
        const thrown = await runWithModelFailover(
            [providerSlot('only-model', 'anthropic')],
            jest.fn().mockRejectedValue(err('AUTH_INVALID')),
            opts,
        ).catch((e) => e);

        expect(readAttemptedSlot(thrown)).toEqual({
            model: 'only-model',
            provider: 'anthropic',
        });
    });

    it('names the primary when the error must not cascade', async () => {
        // An unsafe-to-retry veto stops on the primary, so the primary is what
        // ran and what should be reported.
        const thrown = await runWithModelFailover(
            [
                providerSlot('primary-model', 'openai'),
                providerSlot('fallback-model', 'open_router'),
            ],
            async (_slot, control: FailoverAttemptControl) => {
                control.markUnsafeToRetry();
                throw err('AUTH_INVALID');
            },
            opts,
        ).catch((e) => e);

        expect(readAttemptedSlot(thrown)).toMatchObject({
            model: 'primary-model',
        });
    });

    it('stays invisible to JSON and span recording', async () => {
        // Non-enumerable, like the classification stamp — a diagnostic aid must
        // not turn into payload on every error that crosses a boundary.
        const thrown = await runWithModelFailover(
            [providerSlot('only-model', 'anthropic')],
            jest.fn().mockRejectedValue(err('AUTH_INVALID')),
            opts,
        ).catch((e) => e);

        expect(Object.keys(thrown)).not.toContain('attemptedSlot');
        expect(JSON.stringify(thrown)).not.toContain('only-model');
    });

    it('logs when it cannot stamp, instead of failing silently', async () => {
        // A frozen error degrades to the pre-run resolved slot — which is the
        // wrong-model report this stamp exists to prevent. Staying silent there
        // would make a WRONG diagnostic untraceable.
        mockLog.debug.mockClear();
        const frozen = Object.freeze(err('AUTH_INVALID'));

        const thrown = await runWithModelFailover(
            [providerSlot('only-model', 'anthropic')],
            jest.fn().mockRejectedValue(frozen),
            opts,
        ).catch((e) => e);

        expect(readAttemptedSlot(thrown)).toBeUndefined();
        expect(
            mockLog.debug.mock.calls.some(
                (c) =>
                    typeof c[0]?.message === 'string' &&
                    c[0].message.includes('could not stamp'),
            ),
        ).toBe(true);
    });

    it('says nothing for an error that never went through the failover', () => {
        expect(readAttemptedSlot(new Error('unrelated'))).toBeUndefined();
        expect(readAttemptedSlot(undefined)).toBeUndefined();
        expect(readAttemptedSlot('a string')).toBeUndefined();
    });
});

describe('runWithModelFailover — the circuit breaker', () => {
    // One counted failure opens a key, so each test can set its state up in a
    // single call. The mocked classifier counts TRANSIENT.
    const opts = { runName: 'test', organizationId: 'org-1' };
    const env = { ...process.env };
    beforeEach(() => {
        process.env.LLM_CIRCUIT_FAILURE_THRESHOLD = '1';
        delete process.env.LLM_CIRCUIT_MODE;
        providerCircuit.reset();
    });
    afterEach(() => {
        process.env = { ...env };
        providerCircuit.reset();
    });

    /** Open the breaker on `model` by failing it once as a non-last entry. */
    async function openOn(model: string): Promise<void> {
        await runWithModelFailover(
            [slot(model), slot('opener-fallback')],
            jest
                .fn()
                .mockRejectedValueOnce(err('TRANSIENT'))
                .mockResolvedValueOnce('ok'),
            opts,
        );
    }

    it('skips an open entry and records the skip as a hop', async () => {
        await openOn('A');
        const runOne = jest.fn().mockResolvedValue('ok');
        const { failovers } = await recordModelFailovers(() =>
            runWithModelFailover([slot('A'), slot('B')], runOne, opts),
        );
        expect(runOne).toHaveBeenCalledTimes(1);
        expect(runOne.mock.calls[0][0].model).toBe('B');
        expect(failovers[0].hops).toEqual([
            expect.objectContaining({
                model: 'A',
                outcome: 'skipped',
                reason: 'circuit-open',
            }),
        ]);
        // The attempt that ran knows why it is running.
        expect(runOne.mock.calls[0][1]).toMatchObject({
            chainIndex: 1,
            failoverReason: 'circuit-open',
        });
    });

    it('never skips the last entry', async () => {
        await openOn('A');
        const runOne = jest.fn().mockResolvedValue('ok');
        await expect(
            runWithModelFailover([slot('A')], runOne, opts),
        ).resolves.toBe('ok');
        expect(runOne).toHaveBeenCalledTimes(1);
        expect(runOne.mock.calls[0][1].circuitState).toBe('open');
    });

    it('runs the last entry forced when every entry is open', async () => {
        await openOn('A');
        await openOn('B');
        const runOne = jest.fn().mockResolvedValue('ok');
        const { failovers } = await recordModelFailovers(() =>
            runWithModelFailover([slot('A'), slot('B')], runOne, opts),
        );
        expect(runOne.mock.calls.map((c) => c[0].model)).toEqual(['B']);
        expect(failovers[0].hops.map((h) => h.outcome)).toEqual(['skipped']);
    });

    it('shadow mode runs everything', async () => {
        process.env.LLM_CIRCUIT_MODE = 'shadow';
        await openOn('A');
        mockLog.log.mockClear();
        const runOne = jest.fn().mockResolvedValue('ok');
        await runWithModelFailover([slot('A'), slot('B')], runOne, opts);
        expect(runOne.mock.calls.map((c) => c[0].model)).toEqual(['A']);
        expect(
            mockLog.log.mock.calls.some((c) =>
                String(c[0]?.message).includes('would-skip'),
            ),
        ).toBe(true);
    });

    it('does not count an uncounted category (rate limit, abort)', async () => {
        for (const e of [
            err('RATE_LIMIT'),
            err('TRANSIENT', { __abort: true }),
        ]) {
            await runWithModelFailover(
                [slot('A'), slot('B')],
                jest.fn().mockRejectedValueOnce(e).mockResolvedValue('ok'),
                opts,
            ).catch(() => undefined);
        }
        const runOne = jest.fn().mockResolvedValue('ok');
        await runWithModelFailover([slot('A'), slot('B')], runOne, opts);
        expect(runOne.mock.calls[0][0].model).toBe('A');
    });

    it('still counts a failure after markUnsafeToRetry', async () => {
        await runWithModelFailover(
            [slot('A'), slot('B')],
            async (_s, control: FailoverAttemptControl) => {
                control.markUnsafeToRetry();
                throw err('TRANSIENT');
            },
            opts,
        ).catch(() => undefined);
        const runOne = jest.fn().mockResolvedValue('ok');
        await runWithModelFailover([slot('A'), slot('B')], runOne, opts);
        expect(runOne.mock.calls[0][0].model).toBe('B');
    });

    it('applies no breaker without an org', async () => {
        await runWithModelFailover(
            [slot('A'), slot('B')],
            jest
                .fn()
                .mockRejectedValueOnce(err('TRANSIENT'))
                .mockResolvedValueOnce('ok'),
            { runName: 'test' },
        );
        const runOne = jest.fn().mockResolvedValue('ok');
        await runWithModelFailover([slot('A'), slot('B')], runOne, {
            runName: 'test',
        });
        expect(runOne.mock.calls[0][0].model).toBe('A');
        expect(runOne.mock.calls[0][1].circuitState).toBeUndefined();
    });
});
