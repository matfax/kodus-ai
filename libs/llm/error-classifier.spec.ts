import {
    LlmErrorCategory,
    attachClassification,
    classifyLLMError,
    failureChain,
    getClassification,
    isAbortOrHardTimeout,
    isCallerAbortOrBudget,
    isProviderTimeout,
    isTerminalCategory,
    llmErrorLogLevel,
} from './error-classifier';
import { APICallError } from '@ai-sdk/provider';
import { RetryError } from 'ai';
import { ProviderDeadlineError } from './provider-deadline-error';
import {
    HTML_504_BODY,
    afterSdkRetries,
    httpProviderError,
    jsonErrorBody,
    networkProviderError,
    undiciError,
} from './testing/provider-errors';

import {
    AgentContextWindowTooSmallError,
    AgentPromptTooLargeError,
} from './errors';

function errorWithStatus(message: string, status: number): Error {
    const err = new Error(message) as Error & { status: number };
    err.status = status;
    return err;
}

describe('classifyLLMError', () => {
    it('does not throw on a malformed Error with a null message', () => {
        // Regression: "Cannot read properties of null (reading 'toLowerCase')".
        // err.message is typed string but can be null at runtime; the
        // rawMessage fallback must not blow up the whole error path.
        const err = new Error('x') as Error;
        (err as any).message = null;
        expect(() => classifyLLMError(err)).not.toThrow();
        expect(classifyLLMError(err).category).toBeDefined();
    });

    describe('HTTP status mapping (most reliable signal)', () => {
        it.each([401, 403])('maps %i → AUTH_INVALID', (status) => {
            const err = errorWithStatus('not authorized', status);
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.AUTH_INVALID,
            );
        });

        it('maps 402 → QUOTA_EXCEEDED', () => {
            const err = errorWithStatus('payment required', 402);
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });

        it('maps 404 → MODEL_NOT_FOUND', () => {
            const err = errorWithStatus('not found', 404);
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.MODEL_NOT_FOUND,
            );
        });

        it('reads the access-denied detail from responseBody when message is just "Not Found"', () => {
            // Mirrors @ai-sdk APICallError: terse message, detail in body.
            const err = Object.assign(new Error('Not Found'), {
                statusCode: 404,
                responseBody: JSON.stringify({
                    error: {
                        code: 404,
                        message:
                            'Publisher Model `.../publishers/anthropic/models/claude-sonnet-4-6` was not found or your project does not have access to it.',
                        status: 'NOT_FOUND',
                    },
                }),
            });
            expect(classifyLLMError(err, 'google_vertex').category).toBe(
                LlmErrorCategory.MODEL_ACCESS_DENIED,
            );
        });

        it('maps Vertex 404 "does not have access" → MODEL_ACCESS_DENIED with Model Garden guidance', () => {
            const err = errorWithStatus(
                'Publisher Model `projects/p/locations/global/publishers/anthropic/models/claude-sonnet-4-6` was not found or your project does not have access to it.',
                404,
            );
            const info = classifyLLMError(err, 'google_vertex');
            expect(info.category).toBe(LlmErrorCategory.MODEL_ACCESS_DENIED);
            expect(info.friendlyMessage).toMatch(/Model Garden/i);
        });

        it('disambiguates 429 with quota-ish wording → QUOTA_EXCEEDED', () => {
            const err = errorWithStatus(
                'You exceeded your current quota, please check your plan and billing details.',
                429,
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });

        // Read off production: Z.AI/GLM returns this exact prose with a 429.
        // "balance" matched none of the quota vocabulary, so it landed on
        // RATE_LIMIT -- retry-worthy -- and we kept retrying accounts that
        // could only be fixed by paying. Four GLM scopes were stuck this way.
        it.each([
            'Insufficient balance or no resource package. Please recharge.',
            'insufficient balance',
            'Please recharge your account',
        ])('reads a spent balance as quota, not rate limit: %s', (msg) => {
            expect(classifyLLMError(errorWithStatus(msg, 429)).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });

        it('treats plain 429 → RATE_LIMIT', () => {
            const err = errorWithStatus('too many requests', 429);
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.RATE_LIMIT,
            );
        });

        it.each([500, 502, 503, 504])('maps %i → TRANSIENT', (status) => {
            const err = errorWithStatus('upstream error', status);
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.TRANSIENT,
            );
        });

        it('reads status from nested response.status', () => {
            const err = Object.assign(new Error('boom'), {
                response: { status: 401 },
            });
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.AUTH_INVALID,
            );
        });

        it('reads status from cause chain', () => {
            const err = Object.assign(new Error('wrapped'), {
                cause: { statusCode: 429 },
            });
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.RATE_LIMIT,
            );
        });
    });

    // Production 2026-09-28: every provider failure that outlived the SDK's four
    // attempts reached the customer as "Unexpected error". The SDK wraps the
    // last attempt in a RetryError that carries no status of its own — the 503
    // (or 404, or 429) lives on `lastError`.
    describe('retries exhausted (AI_RetryError wrapping the provider error)', () => {
        function exhausted(lastError: APICallError): RetryError {
            return new RetryError({
                message: `Failed after 4 attempts. Last error: ${lastError.message}`,
                reason: 'maxRetriesExceeded',
                errors: [lastError, lastError, lastError, lastError],
            });
        }

        function apiError(
            statusCode: number,
            message: string,
            responseBody?: string,
        ) {
            return new APICallError({
                message,
                url: 'https://llm.example.test/v1/chat/completions',
                requestBodyValues: {},
                statusCode,
                responseBody,
            });
        }

        it('a 503 on every attempt → TRANSIENT', () => {
            const err = exhausted(apiError(503, 'Service Unavailable'));
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.TRANSIENT,
            );
        });

        it('reads the HTTP status from the last attempt', () => {
            const err = exhausted(apiError(503, 'Service Unavailable'));
            expect(classifyLLMError(err).httpStatus).toBe(503);
        });

        it('a 404 on every attempt → MODEL_NOT_FOUND', () => {
            const err = exhausted(apiError(404, 'Not Found'));
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.MODEL_NOT_FOUND,
            );
        });

        it("classifies from the last attempt's response body", () => {
            const err = exhausted(
                apiError(
                    429,
                    'Too Many Requests',
                    '{"error":{"message":"Insufficient balance or no resource package. Please recharge."}}',
                ),
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });
    });

    describe('error chains: cause and lastError', () => {
        it('classifies from a cause when the outer message says nothing', () => {
            const err = new Error('Bad request', {
                cause: new Error('error code: insufficient_quota'),
            });
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });

        it('terminates on a cause cycle', () => {
            const a = new Error('first') as Error & { cause?: unknown };
            const b = new Error('second') as Error & { cause?: unknown };
            a.cause = b;
            b.cause = a;
            expect(() => classifyLLMError(a)).not.toThrow();
        });

        it('terminates on a lastError cycle', () => {
            const a = new Error('first') as Error & { lastError?: unknown };
            const b = new Error('second') as Error & { lastError?: unknown };
            a.lastError = b;
            b.lastError = a;
            expect(() => classifyLLMError(a)).not.toThrow();
        });

        it('terminates on an error that is its own lastError', () => {
            const a = new Error('Service Unavailable') as Error & {
                lastError?: unknown;
            };
            a.lastError = a;
            expect(classifyLLMError(a).category).toBe(
                LlmErrorCategory.TRANSIENT,
            );
        });
    });

    describe('message-string fallback (no HTTP status)', () => {
        it.each([
            'model_not_found',
            'No such model: x',
            'The model x does not exist',
        ])('"%s" → MODEL_NOT_FOUND', (message) => {
            expect(classifyLLMError(new Error(message)).category).toBe(
                LlmErrorCategory.MODEL_NOT_FOUND,
            );
        });

        // The review path rebuilds the error from the harness trace as a plain
        // Error, so the status can be gone and only the SDK's text survives.
        it('"Service Unavailable" in a retry message → TRANSIENT', () => {
            const err = new Error(
                'Failed after 4 attempts. Last error: AI_APICallError: Service Unavailable',
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.TRANSIENT,
            );
        });

        it('"Bad Gateway" / "Gateway Timeout" → TRANSIENT', () => {
            expect(
                classifyLLMError(new Error('AI_APICallError: Bad Gateway'))
                    .category,
            ).toBe(LlmErrorCategory.TRANSIENT);
            expect(
                classifyLLMError(new Error('AI_APICallError: Gateway Timeout'))
                    .category,
            ).toBe(LlmErrorCategory.TRANSIENT);
        });

        it('OpenAI "exceeds the context window" → CONTEXT_OVERFLOW', () => {
            const err = new Error(
                'Your input exceeds the context window of this model. Please adjust your input and try again.',
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.CONTEXT_OVERFLOW,
            );
        });

        it('"Model does not support this protocol" → MODEL_NOT_FOUND', () => {
            const err = new Error('Model does not support this protocol.');
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.MODEL_NOT_FOUND,
            );
        });

        it('OpenAI insufficient_quota → QUOTA_EXCEEDED', () => {
            const err = new Error(
                'You exceeded your current quota. error code: insufficient_quota',
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });

        it('Anthropic credit_balance_too_low → QUOTA_EXCEEDED', () => {
            const err = new Error(
                'Your credit balance is too low (code: credit_balance_too_low)',
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });

        it('Moonshot account-suspended prose (no status) → QUOTA_EXCEEDED', () => {
            const err = new Error(
                'Account gabriel-xxx is suspended, possibly due to reaching the monthly spending limit or failure to pay past invoices.',
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });

        it('"account is suspended due to insufficient credit" → QUOTA_EXCEEDED', () => {
            const err = new Error(
                'Your account org-abc is suspended due to insufficient credit balance.',
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.QUOTA_EXCEEDED,
            );
        });

        it('invalid_api_key → AUTH_INVALID', () => {
            const err = new Error('Error code: invalid_api_key');
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.AUTH_INVALID,
            );
        });

        it('permission_denied → AUTH_INVALID', () => {
            const err = new Error('permission_denied: cannot access resource');
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.AUTH_INVALID,
            );
        });

        it('rate limit phrasing → RATE_LIMIT', () => {
            const err = new Error('Rate limit reached, retry after 5s');
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.RATE_LIMIT,
            );
        });

        it('model_not_found → MODEL_NOT_FOUND', () => {
            const err = new Error(
                'The model `gpt-5-turbo` does not exist or you do not have access to it',
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.MODEL_NOT_FOUND,
            );
        });

        it('context length errors → CONTEXT_OVERFLOW', () => {
            const err = new Error(
                'This model has a maximum context length of 128000 tokens',
            );
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.CONTEXT_OVERFLOW,
            );
        });

        it.each([
            'ECONNRESET',
            'ETIMEDOUT',
            'socket hang up',
            'fetch failed',
            'request aborted',
        ])('network/timeout pattern "%s" → TRANSIENT', (msg) => {
            const err = new Error(msg);
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.TRANSIENT,
            );
        });

        it.each([
            // An AI_RetryError (SDK retries exhausted) embeds the last attempt
            // as text with NO status field — the default RetryError is a plain
            // Error. These must still classify as transient so the configured
            // BYOK fallback cascades instead of failing the whole review (#1875).
            'Failed after 4 attempts. Last error: AI_APICallError: Bad Gateway',
            'Last error: AI_APICallError: 502 Bad Gateway',
            'Upstream error: service unavailable',
            'Proxy returned 530 (Bad Gateway) from Cloudflare',
            'An upstream error occurred: internal server error',
            'gateway timeout while waiting for the model to respond',
        ])('5xx phrasing without a status (%s) → TRANSIENT', (msg) => {
            // No `status`/`statusCode` on the error — message-string fallback only.
            const err = new Error(msg);
            const info = classifyLLMError(err);
            expect(info.category).toBe(LlmErrorCategory.TRANSIENT);
        });

        it.each([
            // A 5xx-like number embedded in a larger number is not a status:
            // substring checks used to read "5032" as a 503 and flip a permanent
            // failure to TRANSIENT, defeating the purpose of the fallback.
            'The input token count (5032) exceeds the maximum allowed for this model',
            'Request id req_5301 rejected by the gateway',
            // A 5xx glued to arbitrary letters is a request id / hash / base64
            // blob, not a status. Only an explicit status keyword (http/err/
            // error/status/code) counts as adjacency; anything else must stay
            // UNKNOWN so a permanent failure does not wrongly cascade to the
            // paid fallback (#1898 review).
            'Failed: request id req_a503b on the model call',
            'the deployment 05d503ee45x is not reachable',
            'blob payload ...d504e2f... failed to parse',
            // Same class with a `_`, `-`, `.` or `/` in front: `req_503ab`,
            // `trace-503x`, `blob_503f` and a URL path segment are id/hash
            // fragments, not statuses. Only the explicit-keyword alternative
            // (http/err/error/status/code) may read a status across a
            // separator (#1898 review).
            'Failed: request id req_503ab on the model call',
            'trace-503x rejected the call',
            'blob_503f could not be parsed',
            'docs https://cdn.example.com/releases/503 failed to parse',
            // Punctuation that only ever introduces a NUMBER or a base64 window:
            // `1,503 tokens` is a count, and a standard-base64 blob carries
            // `+`/`/`. Reading the triple as a status there would send a
            // permanent failure down the paid fallback (#1898 review).
            'the response mentioned 1,503 tokens before failing',
            'payload QmFz+503 was rejected by the provider',
            // `cloudflare: upstream_530` used to match only because the first
            // alternative ignored the underscore. "upstream" is not a status
            // keyword, so this must stay UNKNOWN rather than cascade.
            'cloudflare: upstream_530 reported by the edge',
            // Annotation punctuation in front is a number / id / issue, not a
            // status: every one of these introduces a value, not an HTTP code,
            // so the bare alternative must refuse them just like a separator
            // (#1898 review).
            'issue #503 is still open in the tracker',
            'the ledger entry $503 was rejected by the parser',
            'branch {503 is unreachable during the walk',
            'vlan (503 on the edge switch is down',
            'selector value =503 was not found',
            'connection on port :503 refused',
            // The #1898 review examples: a *standalone* 5xx in a permanent
            // body is a value/count/limit, not a status — these must not bill
            // the BYOK fallback.
            'The request failed: max_tokens must be <= 503',
            'retry in 503 ms and try again',
            'you have 530 credits remaining',
            'the model returned "limit": 504 in the payload',
            'line 530 col 2 of the template',
            'Request failed: id 503-abc is not a valid model',
        ])('bare digit inside a larger number (%s) → not TRANSIENT', (msg) => {
            const err = new Error(msg);
            expect(classifyLLMError(err).category).not.toBe(
                LlmErrorCategory.TRANSIENT,
            );
        });

        it.each([
            // A 5xx in message text tied to a keyword or another word is no
            // longer read as a status: the real status now rides in via
            // lastError (#1875), and a message body can mention a 5xx number
            // for other reasons. Only a genuine upstream *phrase* (bad gateway
            // / service unavailable / …) or a real HTTP status reaches
            // TRANSIENT — these must stay UNKNOWN so a permanent failure does
            // not wrongly bill the BYOK fallback (#1898 review).
            'upstream responded HTTP_503',
            'proxy hop failed: ERR_502',
            'gateway returned http504',
            'connect failed with 502badgateway via the mesh proxy',
            'the upstream returned code:503',
            'the upstream returned status=503',
            'a JSON body {"code":503} came back from the proxy',
            'a JSON body {"statusCode":503} came back from the proxy',
        ])('status keyword/word-glued 5xx without a real status (%s) → not TRANSIENT', (msg) => {
            const err = new Error(msg);
            expect(classifyLLMError(err).category).not.toBe(
                LlmErrorCategory.TRANSIENT,
            );
        });

        it('unrecognized message → UNKNOWN', () => {
            const err = new Error('something weird happened in the SDK');
            expect(classifyLLMError(err).category).toBe(
                LlmErrorCategory.UNKNOWN,
            );
        });

        it('non-Error inputs do not throw', () => {
            expect(classifyLLMError(null).category).toBe(
                LlmErrorCategory.UNKNOWN,
            );
            expect(classifyLLMError(undefined).category).toBe(
                LlmErrorCategory.UNKNOWN,
            );
            expect(classifyLLMError('plain string error').category).toBe(
                LlmErrorCategory.UNKNOWN,
            );
        });
    });

    describe('classification metadata', () => {
        it('echoes the provider when supplied', () => {
            const out = classifyLLMError(new Error('boom'), 'openai');
            expect(out.provider).toBe('openai');
        });

        it('renders a non-empty friendly message for every category', () => {
            for (const category of Object.values(LlmErrorCategory)) {
                const message = classifyLLMError(
                    Object.assign(new Error(`raw for ${category}`), {
                        status: 0,
                    }),
                ).friendlyMessage;
                expect(message.length).toBeGreaterThan(0);
            }
        });
    });

    describe('context-overflow friendly message — actionable variants', () => {
        it('AgentContextWindowTooSmallError surfaces the exact numbers and 3 options', () => {
            const err = new AgentContextWindowTooSmallError({
                contextWindow: 12_288,
                overheadTokens: 17_192,
                modelName: 'google_gemini:gemini-2.5-flash',
            });
            const msg = classifyLLMError(err).friendlyMessage;
            // The specific diagnosis with model and numbers.
            expect(msg).toContain('google_gemini:gemini-2.5-flash');
            expect(msg).toContain('12,288');
            expect(msg).toContain('17,192');
            // All three action options.
            expect(msg).toContain('Switch to a recommended model');
            expect(msg).toContain('Split the PR');
            expect(msg).toContain("Raise the model's Max input tokens");
            // Names of curated models (the ones shown as cards in BYOK
            // settings — admins should recognize these).
            expect(msg).toContain('Claude Sonnet 4.6');
            expect(msg).toContain('Gemini 3.1 Pro');
        });

        it('AgentPromptTooLargeError surfaces the estimated prompt size', () => {
            const err = new AgentPromptTooLargeError({
                estimatedTokens: 71_110,
                contextWindowTokens: 12_288,
                modelName: 'llama-12k',
            });
            const msg = classifyLLMError(err).friendlyMessage;
            expect(msg).toContain('llama-12k');
            expect(msg).toContain('12,288');
            expect(msg).toContain('71,110');
            expect(msg).toContain('Switch to a recommended model');
        });

        it('raw provider context-overflow (no typed error) omits BYOK-limit option but keeps the others', () => {
            const err = new Error(
                'This model has a maximum context length of 128000 tokens',
            );
            const msg = classifyLLMError(err).friendlyMessage;
            // Diagnosis is generic (no specific numbers — we don't have them).
            expect(msg).toContain('exceeded the maximum context size');
            // Switch-model + split-PR options are present.
            expect(msg).toContain('Switch to a recommended model');
            expect(msg).toContain('Split the PR');
            // BYOK-limit option is omitted (no specific window to compare).
            expect(msg).not.toContain("Raise the model's Max input tokens");
        });

        it('renders as GitHub-flavored Markdown (bold + bullets) so the PR comment formats correctly', () => {
            const err = new AgentContextWindowTooSmallError({
                contextWindow: 12_288,
                overheadTokens: 17_192,
                modelName: 'x',
            });
            const msg = classifyLLMError(err).friendlyMessage;
            expect(msg).toContain('**To resolve, choose one:**');
            expect(
                msg.split('\n').filter((l) => l.startsWith('- ')).length,
            ).toBe(3);
        });
    });
});

describe('isTerminalCategory', () => {
    it.each([
        LlmErrorCategory.AUTH_INVALID,
        LlmErrorCategory.QUOTA_EXCEEDED,
        LlmErrorCategory.MODEL_NOT_FOUND,
        LlmErrorCategory.MODEL_ACCESS_DENIED,
    ])('%s is terminal', (cat) => {
        expect(isTerminalCategory(cat)).toBe(true);
    });

    it.each([
        LlmErrorCategory.RATE_LIMIT,
        LlmErrorCategory.CONTEXT_OVERFLOW,
        LlmErrorCategory.TRANSIENT,
        LlmErrorCategory.UNKNOWN,
    ])('%s is not terminal', (cat) => {
        expect(isTerminalCategory(cat)).toBe(false);
    });
});

describe('llmErrorLogLevel', () => {
    it('terminal BYOK billing (suspended account) → warn', () => {
        const err = new Error(
            'Account xxx is suspended, spending limit reached',
        );
        expect(llmErrorLogLevel(err)).toBe('warn');
    });
    it('auth-invalid → warn', () => {
        expect(llmErrorLogLevel(errorWithStatus('nope', 401))).toBe('warn');
    });
    it('unknown / genuine fault → error', () => {
        expect(
            llmErrorLogLevel(new Error('TypeError: x is not a function')),
        ).toBe('error');
    });
});

describe('attachClassification / getClassification', () => {
    it('round-trips classification info on an Error', () => {
        const err = new Error('boom');
        const info = classifyLLMError(
            errorWithStatus('out of credits', 402),
            'anthropic',
        );
        attachClassification(err, info);

        const retrieved = getClassification(err);
        expect(retrieved).toBeDefined();
        expect(retrieved?.category).toBe(LlmErrorCategory.QUOTA_EXCEEDED);
        expect(retrieved?.provider).toBe('anthropic');
    });

    it('returns undefined when no classification is attached', () => {
        expect(getClassification(new Error('untagged'))).toBeUndefined();
        expect(getClassification(null)).toBeUndefined();
        expect(getClassification('not an object')).toBeUndefined();
    });

    it('classification is not enumerable (does not leak into JSON.stringify)', () => {
        const err = new Error('boom');
        attachClassification(err, classifyLLMError(err, 'openai'));
        const serialized = JSON.stringify({
            message: err.message,
            ...Object.fromEntries(Object.entries(err)),
        });
        expect(serialized).not.toContain('AUTH_INVALID');
        expect(serialized).not.toContain('reviewErrorClassification');
    });
});

describe('the timeout gate: provider timeout vs caller abort', () => {
    // Real SDK errors (see testing/provider-errors.ts). The prose gate read the
    // SDK's "Gateway Timeout" status phrase as a cancel, so a 504 never reached
    // the org's fallback; these pin the split on the fields the errors carry.
    const queueTimeout = () =>
        new Error(
            '[BYOK-QUEUE-TIMEOUT] review waited more than 60s for an LLM concurrency slot',
        );
    const abortError = () =>
        Object.assign(new Error('This operation was aborted'), {
            name: 'AbortError',
        });
    const deadline = () =>
        new ProviderDeadlineError({
            deadlineMs: 90_000,
            source: 'round-trip',
            phase: 'round-trip',
            provider: 'openai',
            model: 'gpt-x',
        });
    const undiciTimeout = (name: string, message: string, code: string) => () =>
        networkProviderError(undiciError(name, message, code));

    describe('isProviderTimeout', () => {
        it.each([
            [
                'an empty-bodied 504',
                () => httpProviderError(504, 'Gateway Timeout'),
            ],
            [
                'an HTML 504',
                () => httpProviderError(504, 'Gateway Timeout', HTML_504_BODY),
            ],
            [
                'a JSON 504',
                () =>
                    httpProviderError(
                        504,
                        'Gateway Timeout',
                        jsonErrorBody('The request timed out.'),
                    ),
            ],
            ['a 408', () => httpProviderError(408, 'Request Timeout')],
            ['a Cloudflare 524', () => httpProviderError(524, '')],
            [
                'an undici connect timeout',
                undiciTimeout(
                    'ConnectTimeoutError',
                    'Connect Timeout Error',
                    'UND_ERR_CONNECT_TIMEOUT',
                ),
            ],
            [
                'an undici headers timeout',
                undiciTimeout(
                    'HeadersTimeoutError',
                    'Headers Timeout Error',
                    'UND_ERR_HEADERS_TIMEOUT',
                ),
            ],
            [
                'an undici body timeout',
                undiciTimeout(
                    'BodyTimeoutError',
                    'Body Timeout Error',
                    'UND_ERR_BODY_TIMEOUT',
                ),
            ],
            [
                'a socket ETIMEDOUT',
                undiciTimeout(
                    'Error',
                    'connect ETIMEDOUT 10.0.0.1:443',
                    'ETIMEDOUT',
                ),
            ],
            [
                'a 504 after the SDK spent its retries',
                async () =>
                    afterSdkRetries([
                        await httpProviderError(504, 'Gateway Timeout'),
                        await httpProviderError(504, 'Gateway Timeout'),
                        await httpProviderError(504, 'Gateway Timeout'),
                    ]),
            ],
            [
                'a ProviderDeadlineError after a retried 503',
                async () =>
                    afterSdkRetries(
                        [
                            await httpProviderError(503, 'Service Unavailable'),
                            deadline(),
                        ],
                        2,
                    ),
            ],
            ['a bare ProviderDeadlineError', async () => deadline()],
        ])('is true for %s', async (_label, make) => {
            const err = await make();
            expect(isProviderTimeout(err)).toBe(true);
            expect(isCallerAbortOrBudget(err)).toBe(false);
        });

        it('the empty and HTML 504s carry only the status phrase (the bug)', async () => {
            // Why the prose gate missed them: this message is all there is.
            for (const body of ['', HTML_504_BODY]) {
                const err = (await httpProviderError(
                    504,
                    'Gateway Timeout',
                    body,
                )) as Error;
                expect(APICallError.isInstance(err)).toBe(true);
                expect(err.message).toBe('Gateway Timeout');
                expect(isAbortOrHardTimeout(err)).toBe(true);
                expect(classifyLLMError(err).category).toBe(
                    LlmErrorCategory.TRANSIENT,
                );
            }
        });

        it('sees the undici timeout the SDK wrapped as "Cannot connect to API"', async () => {
            const err = (await undiciTimeout(
                'ConnectTimeoutError',
                'Connect Timeout Error',
                'UND_ERR_CONNECT_TIMEOUT',
            )()) as Error;
            expect(err.message).toBe(
                'Cannot connect to API: Connect Timeout Error',
            );
            expect(isProviderTimeout(err)).toBe(true);
        });

        it.each([
            ['a 503', () => httpProviderError(503, 'Service Unavailable')],
            ['a 500', () => httpProviderError(500, 'Internal Server Error')],
            ['a 429', () => httpProviderError(429, 'Too Many Requests')],
            [
                'a 400 that mentions a timeout in its body',
                () =>
                    httpProviderError(
                        400,
                        'Bad Request',
                        jsonErrorBody('timeout must be <= 600'),
                    ),
            ],
            [
                'a closed socket',
                undiciTimeout(
                    'SocketError',
                    'other side closed',
                    'UND_ERR_SOCKET',
                ),
            ],
            ['an AbortError', async () => abortError()],
            ['a plain Error', async () => new Error('boom')],
            ['undefined', async () => undefined],
        ])('is false for %s', async (_label, make) => {
            expect(isProviderTimeout(await make())).toBe(false);
        });
    });

    describe('isCallerAbortOrBudget', () => {
        it.each([
            ['an AbortError', async () => abortError()],
            [
                'a DOMException AbortError',
                async () =>
                    new DOMException(
                        'This operation was aborted',
                        'AbortError',
                    ),
            ],
            [
                'a DOMException TimeoutError (AbortSignal.timeout)',
                async () =>
                    new DOMException(
                        'The operation was aborted due to timeout',
                        'TimeoutError',
                    ),
            ],
            [
                'a Next.js ResponseAborted',
                async () =>
                    Object.assign(new Error('response aborted'), {
                        name: 'ResponseAborted',
                    }),
            ],
            [
                'the [HARD-TIMEOUT] race',
                async () => new Error('[HARD-TIMEOUT] review exceeded 600s'),
            ],
            ['a [BYOK-QUEUE-TIMEOUT]', async () => queueTimeout()],
            [
                'a [BYOK-QUEUE-ABORTED]',
                async () =>
                    new Error(
                        '[BYOK-QUEUE-ABORTED] review was cancelled before acquiring an LLM concurrency slot',
                    ),
            ],
            [
                'a [BYOK-QUEUE-TIMEOUT] after a retried 504',
                async () =>
                    afterSdkRetries(
                        [
                            await httpProviderError(504, 'Gateway Timeout'),
                            queueTimeout(),
                        ],
                        2,
                    ),
            ],
            [
                'an AbortError wrapped in an exhausted RetryError',
                async () =>
                    new RetryError({
                        message:
                            'Failed after 2 attempts. Last error: This operation was aborted',
                        reason: 'maxRetriesExceeded',
                        errors: [
                            await httpProviderError(504, 'Gateway Timeout'),
                            abortError(),
                        ],
                    }),
            ],
            [
                'a cancel whose cause is a provider timeout',
                async () =>
                    Object.assign(abortError(), {
                        cause: await httpProviderError(504, 'Gateway Timeout'),
                    }),
            ],
            [
                'a marker passed as a string',
                async () => '[HARD-TIMEOUT] x exceeded 90s',
            ],
        ])('is true for %s', async (_label, make) => {
            expect(isCallerAbortOrBudget(await make())).toBe(true);
        });

        it.each([
            [
                'an empty-bodied 504',
                () => httpProviderError(504, 'Gateway Timeout'),
            ],
            ['a 408', () => httpProviderError(408, 'Request Timeout')],
            [
                'a Vertex "timed out" message',
                () =>
                    httpProviderError(
                        504,
                        'Gateway Timeout',
                        jsonErrorBody('The request timed out.'),
                    ),
            ],
            [
                'a 500 whose text merely says "aborted"',
                () =>
                    httpProviderError(
                        500,
                        'Internal Server Error',
                        jsonErrorBody('upstream request aborted'),
                    ),
            ],
            ['a bare ProviderDeadlineError', async () => deadline()],
            ['undefined', async () => undefined],
        ])('is false for %s', async (_label, make) => {
            expect(isCallerAbortOrBudget(await make())).toBe(false);
        });
    });

    describe('failureChain', () => {
        it('walks lastError, errors.at(-1) and cause, outermost first', async () => {
            const inner = (await undiciTimeout(
                'HeadersTimeoutError',
                'Headers Timeout Error',
                'UND_ERR_HEADERS_TIMEOUT',
            )()) as Error;
            const retry = await afterSdkRetries([inner, inner]);
            const chain = failureChain(retry);
            expect(chain[0]).toBe(retry);
            expect(chain[1]).toBe(inner);
            expect(chain[2]).toBe(inner.cause);
        });

        it('stops on a cycle and at the depth cap', () => {
            const a: any = new Error('a');
            const b: any = new Error('b');
            a.cause = b;
            b.cause = a;
            expect(failureChain(a)).toEqual([a, b]);

            let deep: any = new Error('0');
            for (let i = 1; i < 10; i++) {
                deep = Object.assign(new Error(String(i)), { cause: deep });
            }
            expect(failureChain(deep)).toHaveLength(4);
        });
    });

    it('classifies a ProviderDeadlineError as TRANSIENT, bare or inside a RetryError', async () => {
        expect(classifyLLMError(deadline()).category).toBe(
            LlmErrorCategory.TRANSIENT,
        );
        const wrapped = await afterSdkRetries(
            [await httpProviderError(503, 'Service Unavailable'), deadline()],
            2,
        );
        expect((wrapped as RetryError).reason).toBe('errorNotRetryable');
        expect(classifyLLMError(wrapped).category).toBe(
            LlmErrorCategory.TRANSIENT,
        );
    });
});
