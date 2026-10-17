/**
 * Provider errors as the AI SDK really builds them — for the timeout-gate specs.
 *
 * The 504 bug hid behind hand-built errors: a test that writes
 * `Object.assign(new Error('upstream error'), { status: 504 })` never sees the
 * message the SDK actually puts on an empty or HTML 504, which is the status
 * phrase "Gateway Timeout". These helpers run the SDK's own request and retry
 * code (`postJsonToApi`, the JSON error handler, `handleFetchError`,
 * `retryWithExponentialBackoff`) against a stubbed fetch, so the error a spec
 * asserts on is the one production throws.
 */
import { APICallError } from '@ai-sdk/provider';
import {
    createJsonErrorResponseHandler,
    createJsonResponseHandler,
    postJsonToApi,
    retryWithExponentialBackoff,
} from '@ai-sdk/provider-utils';
import { RetryError } from 'ai';
import { z } from 'zod';

const providerErrorSchema = z.object({
    error: z.object({ message: z.string() }),
});

async function failedCall(fetch: typeof globalThis.fetch): Promise<unknown> {
    try {
        await postJsonToApi({
            url: 'https://provider.test/v1/chat/completions',
            body: { model: 'm' },
            failedResponseHandler: createJsonErrorResponseHandler({
                errorSchema: providerErrorSchema,
                errorToMessage: (data) => data.error.message,
            }),
            successfulResponseHandler: createJsonResponseHandler(z.any()),
            fetch,
        });
    } catch (err) {
        return err;
    }
    throw new Error('expected the stubbed provider call to fail');
}

/** The APICallError the SDK builds for an HTTP error response. */
export function httpProviderError(
    status: number,
    statusText: string,
    body = '',
): Promise<unknown> {
    return failedCall(
        async () =>
            new Response(body, {
                status,
                statusText,
                headers: {
                    'content-type': body.trim().startsWith('{')
                        ? 'application/json'
                        : 'text/html',
                },
            }),
    );
}

/** A JSON error body in the OpenAI-compatible `{error:{message}}` shape. */
export function jsonErrorBody(message: string): string {
    return JSON.stringify({ error: { message } });
}

export const HTML_504_BODY =
    '<html><head><title>504 Gateway Time-out</title></head><body><center><h1>504 Gateway Time-out</h1></center></body></html>';

/** What `fetch` throws when undici fails below HTTP (`TypeError('fetch
 *  failed', {cause})`), after the SDK's `handleFetchError` wrapped it into
 *  "Cannot connect to API: …". */
export function networkProviderError(cause: Error): Promise<unknown> {
    return failedCall(async () => {
        throw new TypeError('fetch failed', { cause });
    });
}

/** undici's own error shape: a name, the default message, and a `code`. */
export function undiciError(
    name: string,
    message: string,
    code: string,
): Error {
    return Object.assign(new Error(message), { name, code });
}

/**
 * Run `attempts` through the SDK's real retry loop, with the same retry rule
 * `generateText` uses (retry an APICallError marked retryable). Each attempt
 * throws the next error. Returns what the loop finally throws: a RetryError
 * once it retried at least once. With the default `maxRetries` the last
 * attempt ends the budget (`maxRetriesExceeded`); pass `attempts.length` to
 * let a non-retryable last attempt stop it instead (`errorNotRetryable`).
 */
export async function afterSdkRetries(
    attempts: unknown[],
    maxRetries = attempts.length - 1,
): Promise<unknown> {
    let call = 0;
    try {
        await retryWithExponentialBackoff({
            maxRetries,
            initialDelayInMs: 0,
            shouldRetry: async (error) =>
                APICallError.isInstance(error) && error.isRetryable === true,
            createRetryError: ({ message, reason, errors }) =>
                new RetryError({ message, reason, errors }),
        })(async () => {
            throw attempts[Math.min(call++, attempts.length - 1)];
        });
    } catch (err) {
        return err;
    }
    throw new Error('expected the retried call to fail');
}
