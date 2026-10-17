import { MAX_FALLBACKS } from "@services/organizationParameters/fallback-chain";
import type { BYOKConfig } from "src/features/ee/byok/_types";

/** The `kodus` provider: Kodus routes the model and bills the org's credits. */
const PLATFORM_FUNDED_PROVIDER = "kodus";

/** A configured model the chain can point at. */
export type FallbackPoolModel = {
    id: string;
    model: string;
    provider: string;
    /** Billed to the org's Kodus credits; the server allows it only first. */
    platformFunded: boolean;
};

/**
 * The models a chain entry can name: every `models[]` entry on a credential the
 * org connected. Models on a managed credential (the env default) are left out,
 * as everywhere else in the UI.
 */
export const fallbackPool = (
    config: BYOKConfig | null | undefined,
): FallbackPoolModel[] => {
    const credentials = new Map(
        (config?.credentials ?? [])
            .filter((credential) => !credential.managed)
            .map((credential) => [credential.id, credential]),
    );
    return (config?.models ?? []).flatMap((model) => {
        const credential = credentials.get(model.credentialId);
        if (!credential) return [];
        return [
            {
                id: model.id,
                model: model.model || model.id,
                provider: credential.provider,
                platformFunded:
                    credential.provider === PLATFORM_FUNDED_PROVIDER,
            },
        ];
    });
};

/** `chain` with the entry at `index` moved by `offset`, clamped to the ends. */
export const moveEntry = (
    chain: readonly string[],
    index: number,
    offset: number,
): string[] => {
    const target = Math.min(Math.max(index + offset, 0), chain.length - 1);
    const next = [...chain];
    if (index < 0 || index >= chain.length || target === index) return next;
    const [entry] = next.splice(index, 1);
    next.splice(target, 0, entry);
    return next;
};

/**
 * Why the server would refuse `chain`, in the words shown under the list, or
 * undefined when it can be saved. The same rules the endpoint enforces, checked
 * here so the user sees which entry to fix instead of a generic 400.
 */
export const chainProblem = (
    chain: readonly string[],
    pool: readonly FallbackPoolModel[],
): string | undefined => {
    if (chain.length > MAX_FALLBACKS) {
        return `At most ${MAX_FALLBACKS} fallback models.`;
    }
    const byId = new Map(pool.map((model) => [model.id, model]));
    const unknown = chain.find((id) => !byId.has(id));
    if (unknown) {
        return `"${unknown}" is no longer a connected model. Remove it to save.`;
    }
    if (chain.slice(1).some((id) => byId.get(id)?.platformFunded)) {
        return "A Kodus-funded model can only be the first fallback.";
    }
    return undefined;
};
