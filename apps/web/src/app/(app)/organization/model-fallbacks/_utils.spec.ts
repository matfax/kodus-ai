import { fallbackChainOf } from "@services/organizationParameters/fallback-chain";
import type { BYOKConfig } from "src/features/ee/byok/_types";

import { chainProblem, fallbackPool, moveEntry } from "./_utils";

const config = (routing: Record<string, unknown> = {}): BYOKConfig => ({
    version: 2,
    credentials: [
        { id: "cred-openai", provider: "openai", apiKey: "sk...abc" },
        { id: "cred-kodus", provider: "kodus" },
        { id: "cred-env", provider: "anthropic", managed: true },
    ],
    models: [
        { id: "model-a", credentialId: "cred-openai", model: "gpt-5" },
        { id: "model-b", credentialId: "cred-openai", model: "gpt-5-mini" },
        { id: "model-k", credentialId: "cred-kodus", model: "kodus-1" },
        { id: "model-env", credentialId: "cred-env", model: "claude-x" },
    ],
    routing: { defaultModelId: "model-a", ...routing },
});

describe("fallbackChainOf (web mirror)", () => {
    it("reads a legacy single fallback as a one-entry chain", () => {
        expect(fallbackChainOf(config({ fallbackModelId: "model-b" }))).toEqual(
            ["model-b"],
        );
    });

    it("puts the head first, dedupes and caps at three", () => {
        expect(
            fallbackChainOf(
                config({
                    fallbackModelId: "model-b",
                    fallbackModelIds: ["model-a", "model-b", "x", "y", "z"],
                }),
            ),
        ).toEqual(["model-b", "model-a", "x"]);
    });

    it("is empty without a config or a fallback", () => {
        expect(fallbackChainOf(undefined)).toEqual([]);
        expect(fallbackChainOf(config())).toEqual([]);
    });
});

describe("fallbackPool", () => {
    it("lists connected models, flags Kodus-funded ones and hides managed ones", () => {
        expect(fallbackPool(config())).toEqual([
            {
                id: "model-a",
                model: "gpt-5",
                provider: "openai",
                platformFunded: false,
            },
            {
                id: "model-b",
                model: "gpt-5-mini",
                provider: "openai",
                platformFunded: false,
            },
            {
                id: "model-k",
                model: "kodus-1",
                provider: "kodus",
                platformFunded: true,
            },
        ]);
    });

    it("is empty without a config", () => {
        expect(fallbackPool(undefined)).toEqual([]);
    });
});

describe("moveEntry", () => {
    it("moves an entry up and down", () => {
        expect(moveEntry(["a", "b", "c"], 2, -1)).toEqual(["a", "c", "b"]);
        expect(moveEntry(["a", "b", "c"], 0, 1)).toEqual(["b", "a", "c"]);
    });

    it("leaves the chain as it is past either end", () => {
        expect(moveEntry(["a", "b"], 0, -1)).toEqual(["a", "b"]);
        expect(moveEntry(["a", "b"], 1, 1)).toEqual(["a", "b"]);
        expect(moveEntry([], 0, 1)).toEqual([]);
    });
});

describe("chainProblem", () => {
    const pool = fallbackPool(config());

    it("accepts a valid chain, including a Kodus-funded head", () => {
        expect(chainProblem([], pool)).toBeUndefined();
        expect(chainProblem(["model-k", "model-a"], pool)).toBeUndefined();
    });

    it("rejects more than three entries", () => {
        expect(
            chainProblem(["model-a", "model-b", "model-k", "x"], pool),
        ).toMatch(/At most 3/);
    });

    it("names an entry that is no longer connected", () => {
        expect(chainProblem(["model-a", "model-gone"], pool)).toMatch(
            /model-gone/,
        );
    });

    it("rejects a Kodus-funded model after the head", () => {
        expect(chainProblem(["model-a", "model-k"], pool)).toMatch(
            /Kodus-funded/,
        );
    });
});
