/**
 * model-failover.integration.spec.ts — `shouldFailoverToNextModel` against the
 * REAL error classifier (only the logger is mocked). `model-failover.spec.ts`
 * mocks the classifier to pin the loop logic; this pins the OTHER half — that a
 * real provider error actually classifies to the category that (does not) cascade,
 * so a regression in the status→category mapping is caught here, not silently.
 */
jest.mock('@libs/core/log/logger', () => ({
    createLogger: () => ({ warn: jest.fn() }),
}));

import { RetryError } from 'ai';

import { shouldFailoverToNextModel } from './model-failover';
import {
    HTML_504_BODY,
    afterSdkRetries,
    httpProviderError,
    jsonErrorBody,
    networkProviderError,
    undiciError,
} from './testing/provider-errors';

const withStatus = (status: number, message = 'boom'): Error =>
    Object.assign(new Error(message), { status });

describe('shouldFailoverToNextModel — real classifier composition', () => {
    it.each([401, 403])('cascades on an auth failure (%i)', (status) => {
        expect(shouldFailoverToNextModel(withStatus(status, 'not authorized'))).toBe(
            true,
        );
    });

    it.each([500, 502, 503, 504])(
        'cascades on a persistent upstream error (%i)',
        (status) => {
            expect(
                shouldFailoverToNextModel(withStatus(status, 'upstream error')),
            ).toBe(true);
        },
    );

    it('does NOT cascade on a rate-limit (429) — the limiter owns backoff', () => {
        expect(shouldFailoverToNextModel(withStatus(429, 'rate limited'))).toBe(
            false,
        );
    });

    it('does NOT cascade on an abort / hard-timeout', () => {
        expect(
            shouldFailoverToNextModel(
                Object.assign(new Error('The operation was aborted'), {
                    name: 'AbortError',
                }),
            ),
        ).toBe(false);
        expect(
            shouldFailoverToNextModel(new Error('[HARD-TIMEOUT] exceeded budget')),
        ).toBe(false);
    });

    describe('provider timeouts, as the SDK really builds them', () => {
        // The regression: the SDK gives an empty or HTML 504 the message
        // "Gateway Timeout", the prose gate read that as a cancel, and the
        // org's fallback never ran although the 504 classifies as TRANSIENT.
        it('cascades on a 504 "Gateway Timeout" with an empty body', async () => {
            const err = await httpProviderError(504, 'Gateway Timeout');
            expect((err as Error).message).toBe('Gateway Timeout');
            expect(shouldFailoverToNextModel(err)).toBe(true);
        });

        it.each([
            [
                'an HTML 504',
                () => httpProviderError(504, 'Gateway Timeout', HTML_504_BODY),
            ],
            [
                'a JSON 504 ("The request timed out.")',
                () =>
                    httpProviderError(
                        504,
                        'Gateway Timeout',
                        jsonErrorBody('The request timed out.'),
                    ),
            ],
            [
                'a 408 "Request Timeout"',
                () => httpProviderError(408, 'Request Timeout'),
            ],
            ['a Cloudflare 524', () => httpProviderError(524, '')],
            [
                'an undici "Connect Timeout Error"',
                () =>
                    networkProviderError(
                        undiciError(
                            'ConnectTimeoutError',
                            'Connect Timeout Error',
                            'UND_ERR_CONNECT_TIMEOUT',
                        ),
                    ),
            ],
            [
                'an undici "Headers Timeout Error"',
                () =>
                    networkProviderError(
                        undiciError(
                            'HeadersTimeoutError',
                            'Headers Timeout Error',
                            'UND_ERR_HEADERS_TIMEOUT',
                        ),
                    ),
            ],
            [
                'a RetryError "…Last error: Gateway Timeout"',
                async () =>
                    afterSdkRetries([
                        await httpProviderError(504, 'Gateway Timeout'),
                        await httpProviderError(504, 'Gateway Timeout'),
                        await httpProviderError(504, 'Gateway Timeout'),
                    ]),
            ],
        ])('cascades on %s', async (_label, make) => {
            expect(shouldFailoverToNextModel(await make())).toBe(true);
        });

        it.each([
            [
                'a [BYOK-QUEUE-TIMEOUT]',
                async () =>
                    new Error(
                        '[BYOK-QUEUE-TIMEOUT] review waited more than 60s for an LLM concurrency slot',
                    ),
            ],
            [
                'a [BYOK-QUEUE-TIMEOUT] that ended a retried 504',
                async () =>
                    afterSdkRetries(
                        [
                            await httpProviderError(504, 'Gateway Timeout'),
                            new Error(
                                '[BYOK-QUEUE-TIMEOUT] review waited more than 60s for an LLM concurrency slot',
                            ),
                        ],
                        2,
                    ),
            ],
            [
                'a DOMException TimeoutError from the caller budget',
                async () =>
                    new DOMException(
                        'The operation was aborted due to timeout',
                        'TimeoutError',
                    ),
            ],
            [
                'an AbortError that ended a retried 504',
                async () =>
                    new RetryError({
                        message:
                            'Failed after 2 attempts. Last error: This operation was aborted',
                        reason: 'maxRetriesExceeded',
                        errors: [
                            await httpProviderError(504, 'Gateway Timeout'),
                            Object.assign(
                                new Error('This operation was aborted'),
                                {
                                    name: 'AbortError',
                                },
                            ),
                        ],
                    }),
            ],
        ])(
            'does NOT cascade on %s (a spent budget, not the model)',
            async (_label, make) => {
                expect(shouldFailoverToNextModel(await make())).toBe(false);
            },
        );
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

        it('restores the prose gate: a "Gateway Timeout" 504 does not cascade', async () => {
            expect(
                shouldFailoverToNextModel(
                    await httpProviderError(504, 'Gateway Timeout'),
                ),
            ).toBe(false);
            expect(
                shouldFailoverToNextModel(
                    await httpProviderError(408, 'Request Timeout'),
                ),
            ).toBe(false);
        });

        it('leaves the rest of the gate as it was', async () => {
            expect(
                shouldFailoverToNextModel(
                    await httpProviderError(503, 'Service Unavailable'),
                ),
            ).toBe(true);
            expect(
                shouldFailoverToNextModel(
                    new Error('[HARD-TIMEOUT] exceeded budget'),
                ),
            ).toBe(false);
        });
    });
});
