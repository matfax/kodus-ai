import { axiosAuthorized } from "src/core/utils/axios";
import type { BYOKConfig, BYOKRouting } from "src/features/ee/byok/_types";

import { ORGANIZATION_PARAMETERS_PATHS } from ".";

/**
 * The org's BYOK fallback chain: the ordered `models[]` ids a failed call
 * cascades through, head first. Web mirror of libs/llm/byok-fallback-chain.ts
 * (the web bundle cannot import a value from libs/llm), kept to the read-side
 * normalization and the limits the server enforces on a write.
 */

/** How many fallbacks one call can cascade through. Mirror of MAX_FALLBACKS. */
export const MAX_FALLBACKS = 3;

/** The routing block with the chain field the enterprise web mirror lacks. */
type RoutingWithChain = BYOKRouting & { fallbackModelIds?: string[] };

const isModelId = (value: unknown): value is string =>
    typeof value === "string" && value.trim().length > 0;

/**
 * The chain as every runtime reader sees it: `fallbackModelId` first when set
 * (the head wins over a disagreeing list), then the rest of
 * `fallbackModelIds`, deduped and capped. Mirror of `fallbackChainOf`.
 */
export const fallbackChainOf = (
    config: BYOKConfig | null | undefined,
): string[] => {
    const routing = config?.routing as RoutingWithChain | undefined;
    const head = isModelId(routing?.fallbackModelId)
        ? routing.fallbackModelId
        : undefined;
    const tail = Array.isArray(routing?.fallbackModelIds)
        ? routing.fallbackModelIds.filter(isModelId)
        : [];
    return [...new Set(head ? [head, ...tail] : tail)].slice(0, MAX_FALLBACKS);
};

/**
 * Replace the chain, and nothing else: the server rewrites only the routing's
 * fallback fields, so no credential travels with this call. An empty list
 * clears the chain. Resolves to the chain as stored.
 */
export const setByokFallbackChain = async (
    modelIds: string[],
): Promise<string[]> => {
    const envelope = await axiosAuthorized.put<{
        data: { fallbackModelIds: string[] };
    }>(ORGANIZATION_PARAMETERS_PATHS.SET_BYOK_FALLBACK_CHAIN, { modelIds });
    return envelope.data?.fallbackModelIds ?? [];
};
