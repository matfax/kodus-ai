/**
 * The org's fallback CHAIN: the ordered `models[]` ids a failed call cascades
 * through, after the model that ran.
 *
 * Two stored fields carry it. `routing.fallbackModelId` is the head, the one
 * field every reader that predates the chain knows (the enterprise routing
 * tab, the audit log). `routing.fallbackModelIds` is the whole chain, head
 * first, and is stored only when there is a tail. Every reader goes through
 * `fallbackChainOf`, so a config that only has `fallbackModelId` resolves to
 * `[fallbackModelId]`, which is exactly the one-hop behaviour it had before.
 *
 * The write side exists because the enterprise routing tab rebuilds `routing`
 * from its four fields on every save and cannot see the tail: without the
 * server-side merge below, any routing-tab save would silently drop it.
 *
 * Pure, ids only: nothing here reads or logs key material.
 */
import type { BYOKConfig, BYOKRouting } from './byok-config';
import { isPlatformFundedProvider } from './platform-funded-provider';

/** How many fallbacks one call can cascade through after its primary. */
export const MAX_FALLBACKS = 3;

const isModelId = (value: unknown): value is string =>
    typeof value === 'string' && value.trim().length > 0;

function dedupe(ids: readonly string[]): string[] {
    return [...new Set(ids)];
}

/**
 * The chain as the routing block states it, head first, deduped, NOT capped.
 * The write validator needs the uncapped list to reject an over-long request;
 * every runtime reader uses `fallbackChainOf`.
 */
export function requestedFallbackChain(
    routing: BYOKRouting | null | undefined,
): string[] {
    const head = isModelId(routing?.fallbackModelId)
        ? routing.fallbackModelId
        : undefined;
    const tail = Array.isArray(routing?.fallbackModelIds)
        ? routing.fallbackModelIds.filter(isModelId)
        : [];
    return dedupe(head ? [head, ...tail] : tail);
}

/**
 * The normalized chain every reader uses: `fallbackModelId` first when it is
 * set (the head wins over a disagreeing list), then the rest of
 * `fallbackModelIds`, deduped and capped at MAX_FALLBACKS.
 */
export function fallbackChainOf(
    routing: BYOKRouting | null | undefined,
): string[] {
    return requestedFallbackChain(routing).slice(0, MAX_FALLBACKS);
}

/**
 * `routing` with `chain` written in the stored shape: `fallbackModelId` is the
 * head, `fallbackModelIds` the whole chain when it has a tail, and both fields
 * are removed for an empty chain.
 */
export function withFallbackChain(
    routing: BYOKRouting,
    chain: readonly string[],
): BYOKRouting {
    const {
        fallbackModelId: _head,
        fallbackModelIds: _chain,
        ...rest
    } = routing;
    if (chain.length === 0) {
        return rest;
    }
    return {
        ...rest,
        fallbackModelId: chain[0],
        ...(chain.length > 1 ? { fallbackModelIds: [...chain] } : {}),
    };
}

/** A tail entry the write dropped, and why. Ids only. */
export interface PrunedFallbackEntry {
    modelId: string;
    reason: 'unknown-model' | 'platform-funded';
}

export interface FallbackChainWrite {
    /** The incoming config with the merged (and pruned) chain on its routing. */
    config: BYOKConfig;
    /** Tail entries carried over from the stored config that no longer qualify. */
    pruned: PrunedFallbackEntry[];
    /** The stored tail before this write, and the tail it persists. */
    previousTail: string[];
    tail: string[];
}

/**
 * Merge the fallback chain of an incoming BYOK write with the stored one.
 *
 *  - The incoming routing has a `fallbackModelIds` key: the request states the
 *    chain, so it is used as given (head first, deduped). It is NOT capped
 *    here, so the validator can reject an over-long request instead of
 *    silently trimming it.
 *  - The key is absent, as on an enterprise routing-tab save: an unset
 *    `fallbackModelId` clears the whole chain ("no fallback" means none);
 *    otherwise the incoming head is kept in front of the stored tail.
 *
 * Tail entries the stored config already had are PRUNED when they no longer
 * resolve to a model or now draw on a platform-funded credential: the
 * enterprise UI cannot see the tail, so rejecting would block every save with
 * an error the user cannot fix. The same "validate only what changed" rule as
 * the baseURL guard. Entries the request adds are left for the validator to
 * reject. The head is never pruned; it is validated as it always was.
 */
export function mergeFallbackChainOnWrite(
    incoming: BYOKConfig,
    existing?: BYOKConfig | null,
): FallbackChainWrite {
    const storedChain = fallbackChainOf(existing?.routing);
    const routing: BYOKRouting = incoming.routing ?? {};

    let chain: string[];
    if (Array.isArray(routing.fallbackModelIds)) {
        chain = requestedFallbackChain(routing);
    } else if (isModelId(routing.fallbackModelId)) {
        const head = routing.fallbackModelId;
        chain = [head, ...storedChain.slice(1).filter((id) => id !== head)];
    } else {
        chain = [];
    }

    const carried = new Set(storedChain.slice(1));
    const pruned: PrunedFallbackEntry[] = [];
    const kept = chain.filter((id, index) => {
        if (index === 0 || !carried.has(id)) {
            return true;
        }
        const reason = disqualification(incoming, id);
        if (reason) {
            pruned.push({ modelId: id, reason });
            return false;
        }
        return true;
    });

    // No routing block means no fallback field to merge (and an empty chain).
    const config: BYOKConfig = incoming.routing
        ? { ...incoming, routing: withFallbackChain(routing, kept) }
        : incoming;

    return {
        config,
        pruned,
        previousTail: storedChain.slice(1),
        tail: fallbackChainOf(config.routing).slice(1),
    };
}

/**
 * Tail entries whose credential is platform-funded (`kodus`). Refused this
 * round: the enterprise UI's credit-balance alarm cannot see the tail, so such
 * an entry would spend prepaid credit with nothing on screen saying so. The
 * head is the one the UI shows, so it stays allowed.
 */
export function platformFundedTailEntries(config: BYOKConfig): string[] {
    return requestedFallbackChain(config.routing)
        .slice(1)
        .filter((id) => disqualification(config, id) === 'platform-funded');
}

/**
 * `routing` with `modelId` removed from the TAIL of its chain, for a model
 * delete. Returns `routing` itself when the model is not a tail entry, so an
 * unrelated delete leaves the stored blob byte-identical. The head is not
 * touched: deleting it is refused by the reference guard instead.
 */
export function removeFromFallbackTail(
    routing: BYOKRouting,
    modelId: string,
): BYOKRouting {
    const chain = fallbackChainOf(routing);
    if (!chain.slice(1).includes(modelId)) {
        return routing;
    }
    return withFallbackChain(
        routing,
        chain.filter((id) => id !== modelId),
    );
}

function disqualification(
    config: BYOKConfig,
    modelId: string,
): PrunedFallbackEntry['reason'] | undefined {
    const model = (config.models ?? []).find((m) => m?.id === modelId);
    if (!model) {
        return 'unknown-model';
    }
    const credential = (config.credentials ?? []).find(
        (c) => c?.id === model.credentialId,
    );
    return isPlatformFundedProvider(credential?.provider)
        ? 'platform-funded'
        : undefined;
}
