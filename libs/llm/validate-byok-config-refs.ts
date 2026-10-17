/**
 * Write-time referential integrity for the BYOK config blob (RFC §13.8).
 *
 * The write DTO is untyped (`POST /create-or-update` accepts `configValue: any`
 * and casts to NormalizedByokConfig — controller:137), so nothing else gates the shape:
 * the front-end fully drives the blob. This validator is the ONLY server-side
 * schema gate. It asserts that every `model.credentialId` resolves to a
 * `credentials[]` entry and every routing ref
 * (`defaultModelId` / `fallbackModelId` / `taskOverrides[*]`) resolves to a
 * `models[]` id — a dangling ref is REJECTED (not silently dropped) before
 * persist. The fallback chain (`fallbackModelIds`) is gated the same way and
 * capped at MAX_FALLBACKS; the write path prunes STORED tail entries that went
 * stale before it validates (byok-fallback-chain.ts), so what fails here is
 * only what the request itself supplied.
 *
 * Secret hygiene: error strings name ids ONLY (model id, credentialId, routing
 * key) — never key material.
 *
 * `findModelReferences` is exported for reuse by the 04-06 model-delete guard
 * (which must reject a delete that would orphan a routing ref).
 */
import { isByokConfig, type LlmTask } from './byok-config';
import {
    fallbackChainOf,
    MAX_FALLBACKS,
    requestedFallbackChain,
} from './byok-fallback-chain';

export interface ByokRefValidationResult {
    valid: boolean;
    /** Human-readable messages naming the offending ids (no secret material). */
    errors: string[];
}

/**
 * Validate the referential integrity of a BYOK config. Legacy `{main,fallback}`
 * (and any non-config blob) is a no-op PASS — only the shape carries refs to gate.
 */
export function validateByokConfigRefs(config: unknown): ByokRefValidationResult {
    if (!isByokConfig(config)) {
        return { valid: true, errors: [] };
    }

    const errors: string[] = [];

    const credentialIds = new Set(
        (config.credentials ?? [])
            .filter((c) => c && typeof c.id === 'string' && c.id)
            .map((c) => c.id),
    );
    const modelIds = new Set(
        (config.models ?? [])
            .filter((m) => m && typeof m.id === 'string' && m.id)
            .map((m) => m.id),
    );

    // Every model must reference an existing credential.
    for (const model of config.models ?? []) {
        if (!model) continue;
        if (!model.credentialId || !credentialIds.has(model.credentialId)) {
            errors.push(
                `Model "${model.id ?? '(missing id)'}" references credentialId ` +
                    `"${model.credentialId ?? '(none)'}" which does not resolve ` +
                    `to any credential`,
            );
        }
    }

    // Every routing ref must point at an existing model.
    const routing = config.routing;
    if (routing) {
        if (
            routing.defaultModelId &&
            !modelIds.has(routing.defaultModelId)
        ) {
            errors.push(
                `routing.defaultModelId "${routing.defaultModelId}" does not ` +
                    `resolve to any model`,
            );
        }
        if (
            routing.fallbackModelId &&
            !modelIds.has(routing.fallbackModelId)
        ) {
            errors.push(
                `routing.fallbackModelId "${routing.fallbackModelId}" does not ` +
                    `resolve to any model`,
            );
        }
        const chain = requestedFallbackChain(routing);
        if (chain.length > MAX_FALLBACKS) {
            errors.push(
                `routing.fallbackModelIds lists ${chain.length} fallback ` +
                    `models; at most ${MAX_FALLBACKS} are allowed`,
            );
        }
        for (const modelId of chain) {
            // `fallbackModelId` itself was already checked above.
            if (modelId !== routing.fallbackModelId && !modelIds.has(modelId)) {
                errors.push(
                    `routing.fallbackModelIds "${modelId}" does not ` +
                        `resolve to any model`,
                );
            }
        }
        for (const [task, modelId] of Object.entries(
            routing.taskOverrides ?? {},
        )) {
            if (modelId && !modelIds.has(modelId)) {
                errors.push(
                    `routing.taskOverrides.${task} "${modelId}" does not ` +
                        `resolve to any model`,
                );
            }
        }
    }

    return { valid: errors.length === 0, errors };
}

/** Human labels for the routing tasks — the delete guard's message is shown to
 *  the user, so it must read in product terms, not raw config paths. */
const TASK_LABEL: Record<LlmTask, string> = {
    codeReview: 'the Code Review model',
    kodyRulesReview: 'the Kody Rules review model',
    ruleGeneration: 'the Kody Rules generation model',
    businessValidation: 'the Business Rules validation model',
    prSummary: 'the PR Summary model',
    conversation: 'the Chat model',
};

/**
 * Which routing refs point at `modelId`, as HUMAN labels (e.g. "your
 * organization default model", "the Code Review model"). Used by the 04-06
 * delete guard to reject a delete that would orphan a ref — and its message is
 * user-facing, so it must not leak dotted config paths. A non-config (or an
 * empty modelId) yields `[]`.
 */
export function findModelReferences(
    config: unknown,
    modelId: string,
): string[] {
    if (!isByokConfig(config) || !modelId) {
        return [];
    }

    const refs: string[] = [];
    const routing = config.routing;
    if (!routing) {
        return refs;
    }

    if (routing.defaultModelId === modelId) {
        refs.push('your organization default model');
    }
    // Only the chain HEAD is a reference that blocks a delete: it is the one
    // the user picked on screen. A tail entry is pruned by the delete instead
    // (removeFromFallbackTail), since no UI shows it to be reassigned.
    if (fallbackChainOf(routing)[0] === modelId) {
        refs.push('your fallback model');
    }
    for (const [task, id] of Object.entries(routing.taskOverrides ?? {})) {
        if (id === modelId) {
            refs.push(TASK_LABEL[task] ?? `the ${task} model`);
        }
    }

    return refs;
}
