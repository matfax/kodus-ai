/**
 * resolveTaskSlot — the SINGLE task→slot routing primitive (slice 04b).
 *
 * "Decide which model + creds for THIS task": resolve the routed model id via
 * `StaticTaskStrategy` over the org's config, materialize that model's slot
 * (ciphertext-bearing), or degrade to `{ slot: undefined }` when there's no BYOK.
 * NO model is BUILT here — `buildModelFromSlot` and everything that RUNS a model
 * live behind `LLM.run`; a consumer passes this slot to `LLM.run`.
 *
 * Degradation contract (never throws):
 *  - no BYOK (config null/undefined/non-v2) → undefined slot → managed default.
 *  - BLOCKED verdict (modelId null) → undefined slot → managed default.
 *  - an unresolvable routed slot (managed/incomplete) → undefined slot.
 *
 * Secret hygiene: the slot carries ENCRYPTED apiKey ciphertext; `resolveTaskSlot`
 * never decrypts (only `buildModelFromSlot`, elsewhere, touches plaintext).
 */

import {
    isByokConfig,
    type BYOKConfig,
    type FallbackSlot,
    type LlmTask,
    type NormalizedModel,
} from './byok-config';
import { fallbackChainOf } from './byok-fallback-chain';
import type { RequestContext, RoutingVerdict } from './routing-strategy';
import { StaticTaskStrategy } from './static-task-strategy';
import { resolveModelSlot } from './resolve-model-slot';
import { asServedSlot } from './served-slot';

// Manual routing policy (Phase 4). Stateless + dependency-free — instantiated once.
const strategy = new StaticTaskStrategy();

/**
 * Resolve `task` over the org's config to a slot (ciphertext-bearing) WITHOUT
 * building a model — the "decide which model + creds" step. BYOK when the org
 * routes one for the task; otherwise `{ slot: null }` → the caller degrades to
 * the managed/env default. Consumers that need to RUN the model pass this slot
 * to `LLM.run` (which builds + wraps it); consumers that only need the routing
 * decision read `.slot` / `.verdict` directly.
 *
 * Degrade contract (never throws): non-v2 / BLOCKED verdict / unresolvable slot
 * → `{ slot: undefined }`.
 *
 * Secret hygiene: the slot carries ENCRYPTED apiKey ciphertext
 * (`resolveModelSlot` never decrypts); only `buildModelFromSlot` touches
 * plaintext.
 */
export function resolveTaskSlot(
    config: BYOKConfig | null | undefined,
    task: LlmTask,
    options: { ctx?: RequestContext } = {},
): { slot: NormalizedModel | undefined; verdict: RoutingVerdict | undefined } {
    if (!isByokConfig(config)) {
        return { slot: undefined, verdict: undefined };
    }

    const verdict = strategy.resolve(task, options.ctx ?? {}, config);

    if (!verdict.modelId) {
        return { slot: undefined, verdict };
    }
    const routed = resolveModelSlot(config, verdict.modelId);
    // id-THEN-name: a legacy NAME override applies onto the resolved slot's
    // `.model` (the slot still supplies the credential/ciphertext). Stamp the
    // routing provenance onto the slot in the SAME step, so it travels to
    // whichever span records the call — the single door where a routed slot is
    // born, so every consumer stays untouched. `route` is the TASK (the argument
    // above), giving the usage span its per-task dimension; `usedFallback` is the
    // one boolean the verdict adds (primary tiers all gated out).
    const slot = routed
        ? asServedSlot({
              ...routed,
              ...(verdict.modelName ? { model: verdict.modelName } : {}),
              route: task,
              usedFallback: verdict.usedFallback,
          })
        : routed;

    // Runtime failover chain: stamp the org's configured fallbacks
    // (gate-checked) onto the primary slot so they ride to `LLM.run` with no
    // per-consumer threading. When the primary that WON is itself a chain
    // entry (`usedFallback`), only the entries after it remain to try. Each
    // entry is a PLAIN slot (no nested `.fallbacks`).
    const slotWithFallbacks = slot
        ? attachFallbacks(config, task, slot, verdict)
        : slot;

    return { slot: slotWithFallbacks, verdict };
}

/**
 * Resolve + gate the org's fallback chain and attach the distinct, eligible
 * entries to `primarySlot.fallbacks`, or return `primarySlot` unchanged when
 * none qualifies. Entries are deduped against the primary and each other, by
 * `models[]` id and by what is actually served (provider + model + baseURL +
 * credential), so two configured ids that call the same thing are tried once.
 */
function attachFallbacks(
    config: BYOKConfig,
    task: LlmTask,
    primarySlot: NormalizedModel,
    verdict: RoutingVerdict,
): NormalizedModel {
    // Fast path (the common case): no fallback configured → skip the extra gate
    // pass entirely. Routing runs on every LLM call, so don't rebuild the lookup
    // maps for a chain that isn't there.
    const chain = fallbackChainOf(config.routing);
    if (chain.length === 0) {
        return primarySlot;
    }
    // A gate-time winner from the chain: only the entries after it are a tail.
    // One that cannot be placed in the chain gets no tail, as before the chain.
    let afterModelId: string | undefined;
    if (verdict.usedFallback) {
        if (!verdict.modelId || !chain.includes(verdict.modelId)) {
            return primarySlot;
        }
        afterModelId = verdict.modelId;
    }

    const { fallbacks: verdicts } = strategy.resolveFallbacks(
        task,
        config,
        afterModelId,
    );
    const seenIds = new Set<string>([verdict.modelId ?? '']);
    const seenServed = new Set<string>([servedKey(primarySlot)]);
    const fallbacks: FallbackSlot[] = [];
    for (const entry of verdicts) {
        if (seenIds.has(entry.modelId)) {
            continue;
        }
        seenIds.add(entry.modelId);
        const routed = resolveModelSlot(config, entry.modelId);
        if (!routed) {
            continue;
        }
        const served = asServedSlot({
            ...routed,
            route: task,
            usedFallback: true,
            chainIndex: entry.chainIndex,
        });
        const key = servedKey(served);
        if (seenServed.has(key)) {
            continue;
        }
        seenServed.add(key);
        fallbacks.push(served);
    }
    return fallbacks.length > 0 ? { ...primarySlot, fallbacks } : primarySlot;
}

/** What a slot actually calls: the same provider + model + endpoint +
 *  credential is the same route, whatever `models[]` id names it. */
function servedKey(slot: NormalizedModel): string {
    return [
        slot.provider,
        slot.model,
        slot.baseURL ?? '',
        slot.credentialId ?? '',
    ].join('::');
}
