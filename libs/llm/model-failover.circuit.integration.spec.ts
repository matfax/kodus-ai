/**
 * model-failover.circuit.integration.spec.ts — the cascade and the circuit
 * breaker together, against the REAL classifier and errors built by the SDK's
 * own request code (only the logger is mocked). `model-failover.spec.ts` pins
 * the loop with a mocked classifier; this pins that a real outage opens the
 * breaker and a real load spike does not.
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

import type { NormalizedModel } from './byok-config';
import {
    recordModelFailovers,
    runWithModelFailover,
    type FailoverAttemptControl,
} from './model-failover';
import {
    circuitKeyFor,
    DEFAULT_CIRCUIT_CONFIG,
    providerCircuit,
} from './provider-circuit';
import {
    HTML_504_BODY,
    afterSdkRetries,
    httpProviderError,
    jsonErrorBody,
} from './testing/provider-errors';

const ORG = 'org-circuit';
const opts = { runName: 'circuit.it', organizationId: ORG };

const slot = (model: string): NormalizedModel =>
    ({
        provider: 'openai',
        apiKey: `cipher-${model}`,
        model,
        byokModelId: model,
        credentialId: 'cred-main',
    }) as NormalizedModel;

const PRIMARY = slot('primary');
const FALLBACK = slot('fallback');
const chain = [PRIMARY, FALLBACK];
const primaryState = () =>
    providerCircuit.stateOf(circuitKeyFor(PRIMARY, ORG)!);

const html504 = () => httpProviderError(504, 'Gateway Timeout', HTML_504_BODY);

/** One call whose primary fails with `fail()` and whose fallback answers. */
async function callFailingPrimary(fail: () => Promise<unknown>) {
    const models: string[] = [];
    const value = await runWithModelFailover(
        chain,
        async (s) => {
            models.push(s!.model);
            if (s === PRIMARY) throw await fail();
            return 'ok';
        },
        opts,
    );
    return { value, models };
}

describe('the circuit breaker in the failover cascade (real errors)', () => {
    const env = { ...process.env };
    beforeEach(() => {
        delete process.env.LLM_CIRCUIT_MODE;
        delete process.env.LLM_CIRCUIT_FAILURE_THRESHOLD;
        providerCircuit.reset();
        jest.useFakeTimers({ now: Date.parse('2026-10-17T10:00:00Z') });
    });
    afterEach(() => {
        jest.useRealTimers();
        process.env = { ...env };
        providerCircuit.reset();
    });

    it('three HTML 504s open the primary, and the next call goes straight to the fallback', async () => {
        for (let i = 0; i < DEFAULT_CIRCUIT_CONFIG.failureThreshold; i++) {
            const { models } = await callFailingPrimary(html504);
            expect(models).toEqual(['primary', 'fallback']);
        }
        expect(primaryState()).toBe('open');

        const runOne = jest.fn().mockResolvedValue('ok');
        const { failovers } = await recordModelFailovers(() =>
            runWithModelFailover(chain, runOne, opts),
        );
        expect(runOne.mock.calls.map((c) => c[0].model)).toEqual(['fallback']);
        expect(failovers[0].hops[0]).toMatchObject({
            model: 'primary',
            outcome: 'skipped',
            reason: 'circuit-open',
        });
    });

    it('after openMs the probe re-admits the primary on its first answer', async () => {
        for (let i = 0; i < DEFAULT_CIRCUIT_CONFIG.failureThreshold; i++) {
            await callFailingPrimary(html504);
        }
        jest.advanceTimersByTime(DEFAULT_CIRCUIT_CONFIG.openMs);
        expect(primaryState()).toBe('half-open');

        // The probe is a loop: its first provider answer closes the key, and
        // a failure later in the same loop (after a step ran) changes nothing.
        const e504 = await html504();
        await runWithModelFailover(
            chain,
            async (s, control: FailoverAttemptControl) => {
                expect(s).toBe(PRIMARY);
                control.onAnswered();
                control.markUnsafeToRetry();
                throw e504;
            },
            opts,
        ).catch(() => undefined);
        expect(primaryState()).toBe('closed');

        const runOne = jest.fn().mockResolvedValue('ok');
        await runWithModelFailover(chain, runOne, opts);
        expect(runOne.mock.calls[0][0]).toBe(PRIMARY);
    });

    it('a late failure from a loop admitted before the outage does not change state', async () => {
        process.env.LLM_CIRCUIT_FAILURE_THRESHOLD = '1';
        let failOldLoop: (err: unknown) => void = () => undefined;
        const oldLoop = runWithModelFailover(
            chain,
            (s) =>
                s === PRIMARY
                    ? new Promise((_, reject) => {
                          failOldLoop = reject;
                      })
                    : Promise.resolve('ok'),
            opts,
        );

        // Meanwhile: the primary opens, cools off, and a probe closes it.
        await callFailingPrimary(html504);
        expect(primaryState()).toBe('open');
        jest.advanceTimersByTime(DEFAULT_CIRCUIT_CONFIG.openMs);
        await runWithModelFailover(chain, async () => 'ok', opts);
        expect(primaryState()).toBe('closed');

        // Forty minutes later the old loop fails with a counted error.
        jest.advanceTimersByTime(40 * 60_000);
        failOldLoop(await html504());
        await expect(oldLoop).resolves.toBe('ok');
        expect(primaryState()).toBe('closed');
    });

    it('concurrent exhausted 429s never divert later one-shot calls', async () => {
        // The SDK's retry loop waits on real timers, even at zero delay.
        jest.useRealTimers();
        const e429 = await httpProviderError(
            429,
            'Too Many Requests',
            jsonErrorBody('rate limited'),
        );
        const exhausted = await afterSdkRetries([e429, e429, e429]);
        // Each exhausted call cascades on its own…
        const results = await Promise.all(
            Array.from({ length: 10 }, () =>
                callFailingPrimary(async () => exhausted),
            ),
        );
        expect(
            results.every((r) => r.models.join() === 'primary,fallback'),
        ).toBe(true);
        // …but the breaker learned nothing from them.
        expect(primaryState()).toBe('closed');
        const runOne = jest.fn().mockResolvedValue('ok');
        await runWithModelFailover(chain, runOne, opts);
        expect(runOne.mock.calls[0][0]).toBe(PRIMARY);
    });

    it('a plain 429 neither cascades nor counts', async () => {
        const e429 = await httpProviderError(
            429,
            'Too Many Requests',
            jsonErrorBody('rate limited'),
        );
        for (let i = 0; i < 5; i++) {
            await expect(callFailingPrimary(async () => e429)).rejects.toBe(
                e429,
            );
        }
        expect(primaryState()).toBe('closed');
    });
});
