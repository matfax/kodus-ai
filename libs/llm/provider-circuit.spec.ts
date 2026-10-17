/**
 * provider-circuit.spec.ts — the breaker's state machine, its keys and what it
 * counts, on an injected clock. The failover loop that drives it is pinned in
 * `model-failover.spec.ts` and `model-failover.circuit.integration.spec.ts`.
 */
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

import { APICallError } from '@ai-sdk/provider';

import { createLogger } from '@libs/core/log/logger';
import type { NormalizedModel } from './byok-config';
import { ProviderDeadlineError } from './provider-deadline-error';
import {
    circuitConfigFromEnv,
    circuitCountsFailure,
    circuitKeyFor,
    circuitModeFromEnv,
    DEFAULT_CIRCUIT_CONFIG,
    ProviderCircuitBreaker,
    providerAccountKey,
    retryAfterMsOf,
    type CircuitConfig,
    type CircuitMode,
} from './provider-circuit';
import {
    HTML_504_BODY,
    afterSdkRetries,
    httpProviderError,
    jsonErrorBody,
} from './testing/provider-errors';

const mockLog = createLogger('test') as unknown as Record<string, jest.Mock>;

const KEY = 'org-1::openai::abc::::gpt';
const LEASE = 60_000;

function breaker(
    opts: { mode?: CircuitMode; config?: Partial<CircuitConfig> } = {},
) {
    let now = 1_000_000;
    const b = new ProviderCircuitBreaker({
        now: () => now,
        mode: () => opts.mode ?? 'enforce',
        config: () => ({ ...DEFAULT_CIRCUIT_CONFIG, ...opts.config }),
    });
    return {
        b,
        advance: (ms: number) => {
            now += ms;
        },
        admit: (isLast = false, extra: { cooling?: boolean } = {}) =>
            b.admit(KEY, {
                isLast,
                probeLeaseMs: LEASE,
                labels: { provider: 'openai', model: 'gpt' },
                isCoolingDown: () => !!extra.cooling,
            }),
    };
}

/** Admit and fail, `n` times, on a non-last entry. */
function failTimes(t: ReturnType<typeof breaker>, n: number): void {
    for (let i = 0; i < n; i++) {
        t.b.onFailure(t.admit().ticket);
    }
}

function openKey(t: ReturnType<typeof breaker>): void {
    failTimes(t, DEFAULT_CIRCUIT_CONFIG.failureThreshold);
    expect(t.b.stateOf(KEY)).toBe('open');
}

function halfOpenKey(t: ReturnType<typeof breaker>): void {
    openKey(t);
    t.advance(DEFAULT_CIRCUIT_CONFIG.openMs);
    expect(t.b.stateOf(KEY)).toBe('half-open');
}

describe('ProviderCircuitBreaker — CLOSED → OPEN', () => {
    it('opens after `failureThreshold` consecutive counted failures', () => {
        const t = breaker();
        failTimes(t, 2);
        expect(t.b.stateOf(KEY)).toBe('closed');
        failTimes(t, 1);
        expect(t.b.stateOf(KEY)).toBe('open');
    });

    it('restarts the count when the previous failure fell out of the window', () => {
        const t = breaker();
        failTimes(t, 2);
        t.advance(DEFAULT_CIRCUIT_CONFIG.windowMs + 1);
        failTimes(t, 2);
        expect(t.b.stateOf(KEY)).toBe('closed');
        failTimes(t, 1);
        expect(t.b.stateOf(KEY)).toBe('open');
    });

    it('a success resets the count', () => {
        const t = breaker();
        failTimes(t, 2);
        t.b.onSuccess(t.admit().ticket);
        failTimes(t, 2);
        expect(t.b.stateOf(KEY)).toBe('closed');
    });

    it('skips a non-last entry while OPEN, and runs the last one forced', () => {
        const t = breaker();
        openKey(t);
        expect(t.admit(false).decision).toBe('skip');
        expect(t.admit(false).ticket).toBeNull();
        const forced = t.admit(true);
        expect(forced.decision).toBe('forced');
        expect(forced.ticket?.forced).toBe(true);
    });

    it('logs the opening as a warn naming provider and model, never the key', () => {
        mockLog.warn.mockClear();
        const t = breaker();
        openKey(t);
        const line = mockLog.warn.mock.calls.at(-1)?.[0];
        expect(line.message).toContain('[LLM-CIRCUIT] open openai/gpt');
        expect(JSON.stringify(line)).not.toContain('abc');
    });
});

describe('ProviderCircuitBreaker — HALF_OPEN and the probe', () => {
    it('moves to HALF_OPEN once `openMs` has passed', () => {
        const t = breaker();
        openKey(t);
        t.advance(DEFAULT_CIRCUIT_CONFIG.openMs - 1);
        expect(t.b.stateOf(KEY)).toBe('open');
        t.advance(1);
        expect(t.b.stateOf(KEY)).toBe('half-open');
    });

    it('leases exactly one probe under concurrent admits', () => {
        const t = breaker();
        halfOpenKey(t);
        const decisions = [t.admit(), t.admit(), t.admit()].map(
            (a) => a.decision,
        );
        expect(decisions).toEqual(['probe', 'skip', 'skip']);
    });

    it('a probe that answers closes the key and resets openMs', () => {
        const t = breaker();
        halfOpenKey(t);
        const probe = t.admit();
        t.b.onSuccess(probe.ticket);
        expect(t.b.stateOf(KEY)).toBe('closed');
        // A fresh opening starts from the base period again.
        openKey(t);
        t.advance(DEFAULT_CIRCUIT_CONFIG.openMs);
        expect(t.b.stateOf(KEY)).toBe('half-open');
    });

    it('a probe that fails reopens with openMs doubled, capped at maxOpenMs', () => {
        const t = breaker({ config: { openMs: 1_000, maxOpenMs: 3_000 } });
        failTimes(t, 3);
        const periods: number[] = [];
        for (let round = 0; round < 3; round++) {
            let waited = 0;
            while (t.b.stateOf(KEY) === 'open') {
                t.advance(100);
                waited += 100;
            }
            periods.push(waited);
            t.b.onFailure(t.admit().ticket);
        }
        expect(periods).toEqual([1_000, 2_000, 3_000]);
    });

    it('a probe that ends uncounted hands the lease back and stays HALF_OPEN', () => {
        const t = breaker();
        halfOpenKey(t);
        t.b.release(t.admit().ticket);
        expect(t.b.stateOf(KEY)).toBe('half-open');
        expect(t.admit().decision).toBe('probe');
    });

    it('an expired lease frees the probe, and the old probe result is stale', () => {
        const t = breaker();
        halfOpenKey(t);
        const first = t.admit();
        expect(t.admit().decision).toBe('skip');
        t.advance(LEASE);
        const second = t.admit();
        expect(second.decision).toBe('probe');
        // The first probe's late answer no longer decides anything.
        t.b.onSuccess(first.ticket);
        expect(t.b.stateOf(KEY)).toBe('half-open');
        t.b.onSuccess(second.ticket);
        expect(t.b.stateOf(KEY)).toBe('closed');
    });

    it('hands out no probe while the slot limiter is cooling down', () => {
        const t = breaker();
        halfOpenKey(t);
        expect(t.admit(false, { cooling: true }).decision).toBe('skip');
        expect(t.admit(false, { cooling: false }).decision).toBe('probe');
    });
});

describe('ProviderCircuitBreaker — forced runs', () => {
    it('a forced success closes the key', () => {
        const t = breaker();
        openKey(t);
        t.b.onSuccess(t.admit(true).ticket);
        expect(t.b.stateOf(KEY)).toBe('closed');
    });

    it('a forced failure leaves OPEN as it is: no doubling, until unmoved', () => {
        const t = breaker();
        openKey(t);
        t.advance(DEFAULT_CIRCUIT_CONFIG.openMs - 10);
        t.b.onFailure(t.admit(true).ticket);
        expect(t.b.stateOf(KEY)).toBe('open');
        t.advance(10);
        expect(t.b.stateOf(KEY)).toBe('half-open');
    });

    it('says "not enforced" for a key only ever forced (single-model org)', () => {
        mockLog.warn.mockClear();
        const t = breaker();
        for (let i = 0; i < 3; i++) t.b.onFailure(t.admit(true).ticket);
        expect(t.b.stateOf(KEY)).toBe('open');
        expect(mockLog.warn.mock.calls.at(-1)?.[0].message).toContain(
            '(not enforced: last entry)',
        );
        // And it never skips anything: the only entry always runs.
        expect(t.admit(true).decision).toBe('forced');
    });
});

describe('ProviderCircuitBreaker — stale generations', () => {
    it('ignores a CLOSED-era failure once the key opened (no early reopen after the probe)', () => {
        const t = breaker();
        const longLoop = t.admit();
        openKey(t);
        halfOpenKeyFromOpen(t);
        t.b.onSuccess(t.admit().ticket);
        expect(t.b.stateOf(KEY)).toBe('closed');
        // The loop admitted before any of that ends now with a failure.
        t.b.onFailure(longLoop.ticket);
        t.b.onFailure(longLoop.ticket);
        t.b.onFailure(longLoop.ticket);
        expect(t.b.stateOf(KEY)).toBe('closed');
    });

    it('ignores a CLOSED-era success while OPEN', () => {
        const t = breaker();
        const early = t.admit();
        openKey(t);
        t.b.onSuccess(early.ticket);
        expect(t.b.stateOf(KEY)).toBe('open');
    });

    it('ignores an OPEN-era forced result once the key is HALF_OPEN', () => {
        const t = breaker();
        openKey(t);
        const forced = t.admit(true);
        t.advance(DEFAULT_CIRCUIT_CONFIG.openMs);
        expect(t.b.stateOf(KEY)).toBe('half-open');
        t.b.onSuccess(forced.ticket);
        expect(t.b.stateOf(KEY)).toBe('half-open');
    });

    it('ignores a probe failure reported after the probe already closed the key', () => {
        // The probe resolves on the first answer; the loop it belongs to may
        // still fail much later.
        const t = breaker();
        halfOpenKey(t);
        const probe = t.admit();
        t.b.onSuccess(probe.ticket);
        t.b.onFailure(probe.ticket);
        expect(t.b.stateOf(KEY)).toBe('closed');
        failTimes(t, 2);
        expect(t.b.stateOf(KEY)).toBe('closed');
    });
});

function halfOpenKeyFromOpen(t: ReturnType<typeof breaker>): void {
    t.advance(DEFAULT_CIRCUIT_CONFIG.openMs);
    expect(t.b.stateOf(KEY)).toBe('half-open');
}

describe('ProviderCircuitBreaker — retry-after', () => {
    it('extends the open period to the provider reset', () => {
        const t = breaker();
        failTimes(t, 2);
        t.b.onFailure(t.admit().ticket, { retryAfterMs: 120_000 });
        t.advance(DEFAULT_CIRCUIT_CONFIG.openMs);
        expect(t.b.stateOf(KEY)).toBe('open');
        t.advance(120_000 - DEFAULT_CIRCUIT_CONFIG.openMs);
        expect(t.b.stateOf(KEY)).toBe('half-open');
    });

    it('caps the extension at maxResetMs', () => {
        const t = breaker({ config: { maxResetMs: 60_000 } });
        failTimes(t, 2);
        t.b.onFailure(t.admit().ticket, { retryAfterMs: 24 * 3_600_000 });
        t.advance(60_000);
        expect(t.b.stateOf(KEY)).toBe('half-open');
    });

    it('reads retry-after (seconds, HTTP date) and retry-after-ms from the error', () => {
        const withHeaders = (responseHeaders: Record<string, string>) =>
            new APICallError({
                message: 'Service Unavailable',
                url: 'https://p.test',
                requestBodyValues: {},
                statusCode: 503,
                responseHeaders,
            });
        expect(retryAfterMsOf(withHeaders({ 'retry-after': '30' }))).toBe(
            30_000,
        );
        expect(retryAfterMsOf(withHeaders({ 'retry-after-ms': '1500' }))).toBe(
            1_500,
        );
        const now = Date.parse('2026-10-17T10:00:00Z');
        expect(
            retryAfterMsOf(
                withHeaders({ 'retry-after': 'Sat, 17 Oct 2026 10:01:00 GMT' }),
                now,
            ),
        ).toBe(60_000);
        expect(retryAfterMsOf(new Error('no headers'))).toBeUndefined();
    });
});

describe('ProviderCircuitBreaker — modes', () => {
    it('shadow keeps state but never skips, and flags the would-skip', () => {
        const t = breaker({ mode: 'shadow' });
        openKey(t);
        const a = t.admit(false);
        expect(a.decision).toBe('run');
        expect(a.wouldSkip).toBe(true);
        // A would-skip run does not change the state enforce would have kept.
        expect(a.ticket).toBeNull();
    });

    it('off keeps no state and always runs', () => {
        const t = breaker({ mode: 'off' });
        failTimes(t, 10);
        expect(t.b.size()).toBe(0);
        expect(t.admit(false)).toMatchObject({ decision: 'run', ticket: null });
    });

    it('defaults to enforce; reads off and shadow; ignores junk', () => {
        expect(circuitModeFromEnv({})).toBe('enforce');
        expect(circuitModeFromEnv({ LLM_CIRCUIT_MODE: 'shadow' })).toBe(
            'shadow',
        );
        expect(circuitModeFromEnv({ LLM_CIRCUIT_MODE: 'OFF' })).toBe('off');
        expect(circuitModeFromEnv({ LLM_CIRCUIT_MODE: 'maybe' })).toBe(
            'enforce',
        );
    });

    it('reads the knobs from env, keeping defaults for unset or invalid ones', () => {
        expect(
            circuitConfigFromEnv({
                LLM_CIRCUIT_FAILURE_THRESHOLD: '1',
                LLM_CIRCUIT_OPEN_MS: '5000',
                LLM_CIRCUIT_WINDOW_MS: 'soon',
            }),
        ).toMatchObject({
            failureThreshold: 1,
            openMs: 5_000,
            windowMs: DEFAULT_CIRCUIT_CONFIG.windowMs,
            maxResetMs: DEFAULT_CIRCUIT_CONFIG.maxResetMs,
        });
    });
});

describe('ProviderCircuitBreaker — memory', () => {
    it('evicts CLOSED keys idle past idleEvictMs', () => {
        const t = breaker();
        t.b.admit('idle', { isLast: false, probeLeaseMs: LEASE });
        t.advance(DEFAULT_CIRCUIT_CONFIG.idleEvictMs + 1);
        t.b.admit('fresh', { isLast: false, probeLeaseMs: LEASE });
        expect(t.b.stateOf('idle')).toBeUndefined();
        expect(t.b.stateOf('fresh')).toBe('closed');
    });

    it('keeps an idle OPEN key, and holds the hard cap', () => {
        const t = breaker({ config: { maxKeys: 3 } });
        openKey(t);
        t.advance(DEFAULT_CIRCUIT_CONFIG.idleEvictMs + 1);
        for (const k of ['a', 'b', 'c', 'd']) {
            t.b.admit(k, { isLast: false, probeLeaseMs: LEASE });
        }
        expect(t.b.size()).toBeLessThanOrEqual(3);
        expect(t.b.stateOf(KEY)).toBeDefined();
    });

    it('counts transitions per provider and state', () => {
        const t = breaker();
        halfOpenKey(t);
        t.b.onSuccess(t.admit().ticket);
        expect(t.b.transitionCounts()).toEqual({
            'openai:open': 1,
            'openai:half-open': 1,
            'openai:closed': 1,
        });
    });
});

describe('circuit keys', () => {
    const byok = (over: Partial<NormalizedModel> = {}): NormalizedModel =>
        ({
            provider: 'openai',
            apiKey: 'ciphertext-1',
            model: 'gpt-5',
            credentialId: 'cred-main',
            ...over,
        }) as NormalizedModel;

    it('never collides across orgs that share a credential id', () => {
        // Migrated orgs all carry `cred-main`; the key must not.
        expect(circuitKeyFor(byok(), 'org-a')).not.toBe(
            circuitKeyFor(byok(), 'org-b'),
        );
        expect(
            circuitKeyFor(byok({ apiKey: 'ciphertext-2' }), 'org-a'),
        ).not.toBe(circuitKeyFor(byok(), 'org-a'));
    });

    it('keys per model on one account, and never holds key material', () => {
        const a = circuitKeyFor(byok(), 'org-a')!;
        const b = circuitKeyFor(byok({ model: 'gpt-5-mini' }), 'org-a')!;
        expect(a).not.toBe(b);
        expect(providerAccountKey(byok(), 'org-a')).toBe(
            providerAccountKey(byok({ model: 'gpt-5-mini' }), 'org-a'),
        );
        expect(a).not.toContain('ciphertext-1');
    });

    it('separates base URLs on the same key', () => {
        expect(
            circuitKeyFor(byok({ baseURL: 'https://a.test/v1' }), 'org-a'),
        ).not.toBe(
            circuitKeyFor(byok({ baseURL: 'https://b.test/v1' }), 'org-a'),
        );
    });

    it('gives no breaker without an org or a slot', () => {
        expect(circuitKeyFor(byok(), undefined)).toBeNull();
        expect(circuitKeyFor(undefined, 'org-a')).toBeNull();
    });

    it('keys a key-less credential by its credential id', () => {
        const keyless = byok({ apiKey: '', credentialId: 'cred-iam' });
        expect(circuitKeyFor(keyless, 'org-a')).not.toBeNull();
        expect(circuitKeyFor(keyless, 'org-a')).not.toBe(
            circuitKeyFor(
                byok({ apiKey: '', credentialId: 'cred-other' }),
                'org-a',
            ),
        );
    });

    it('keys platform-funded slots org-blind, per upstream and model', () => {
        const kodus = (model: string) =>
            ({
                provider: 'kodus',
                apiKey: '',
                model,
            }) as unknown as NormalizedModel;
        const a = circuitKeyFor(kodus('anthropic/claude-sonnet-4-5'), 'org-a');
        expect(a).not.toBeNull();
        expect(a).toBe(
            circuitKeyFor(kodus('anthropic/claude-sonnet-4-5'), 'org-b'),
        );
        expect(a).toContain('platform');
        expect(a).not.toBe(circuitKeyFor(kodus('openai/gpt-5'), 'org-a'));
    });
});

describe('circuitCountsFailure — what trips the breaker', () => {
    it.each([
        [
            'an HTML 504',
            () => httpProviderError(504, 'Gateway Timeout', HTML_504_BODY),
        ],
        [
            'a JSON 503',
            () =>
                httpProviderError(
                    503,
                    'Service Unavailable',
                    jsonErrorBody('overloaded'),
                ),
        ],
        [
            'a 402 (quota)',
            () =>
                httpProviderError(
                    402,
                    'Payment Required',
                    jsonErrorBody('insufficient credit'),
                ),
        ],
    ])('counts %s', async (_label, make) => {
        expect(circuitCountsFailure(await make())).toBe(true);
    });

    it('counts a 504 that exhausted the SDK retries', async () => {
        const e504 = await httpProviderError(504, 'Gateway Timeout');
        expect(circuitCountsFailure(await afterSdkRetries([e504, e504]))).toBe(
            true,
        );
    });

    it.each([
        [
            'a 429',
            () =>
                httpProviderError(
                    429,
                    'Too Many Requests',
                    jsonErrorBody('rate limited'),
                ),
        ],
        [
            'a 401',
            () =>
                httpProviderError(
                    401,
                    'Unauthorized',
                    jsonErrorBody('bad key'),
                ),
        ],
        [
            'a 404 model',
            () =>
                httpProviderError(
                    404,
                    'Not Found',
                    jsonErrorBody('model not found'),
                ),
        ],
        [
            'a context overflow',
            () =>
                httpProviderError(
                    400,
                    'Bad Request',
                    jsonErrorBody('maximum context length exceeded'),
                ),
        ],
    ])('does not count %s', async (_label, make) => {
        expect(circuitCountsFailure(await make())).toBe(false);
    });

    it('never counts an exhausted 429 either', async () => {
        const e429 = await httpProviderError(
            429,
            'Too Many Requests',
            jsonErrorBody('rate limited'),
        );
        expect(
            circuitCountsFailure(await afterSdkRetries([e429, e429, e429])),
        ).toBe(false);
    });

    it('does not count a caller abort or a Kodus budget', () => {
        expect(
            circuitCountsFailure(
                new DOMException('This operation was aborted', 'AbortError'),
            ),
        ).toBe(false);
        expect(
            circuitCountsFailure(new Error('[BYOK-QUEUE-TIMEOUT] waited 60s')),
        ).toBe(false);
        expect(
            circuitCountsFailure(new Error('[HARD-TIMEOUT] x exceeded 90s')),
        ).toBe(false);
    });

    it('counts a full-length round-trip deadline, not a caller-budget one', () => {
        const deadline = (source: 'round-trip' | 'caller-budget') =>
            new ProviderDeadlineError({
                deadlineMs: 1_000,
                source,
                phase: 'round-trip',
            });
        expect(circuitCountsFailure(deadline('round-trip'))).toBe(true);
        expect(circuitCountsFailure(deadline('caller-budget'))).toBe(false);
    });
});
