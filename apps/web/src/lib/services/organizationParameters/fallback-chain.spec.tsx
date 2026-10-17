/** @jest-environment jsdom */
/**
 * The save path of the model-fallbacks page below the page itself: the PUT
 * the service sends, the envelope it unwraps, and the cache key the mutation
 * invalidates, which must be the one the BYOK config is read under, or a save
 * never re-seeds the editor.
 */
import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { axiosAuthorized } from "src/core/utils/axios";

import { ORGANIZATION_PARAMETERS_PATHS } from ".";
import { setByokFallbackChain } from "./fallback-chain";
import { useSetByokFallbackChain, useSuspenseGetBYOK } from "./hooks";

// fetch.ts pulls the authorized-fetch stack (next-auth, ESM), which jest
// cannot load as CommonJS; nothing here exercises it.
jest.mock("@services/fetch", () => ({ authorizedFetch: jest.fn() }));
jest.mock("src/core/providers/auth.provider", () => ({
    useAuth: () => ({ accessToken: "token" }),
}));
jest.mock("src/core/utils/axios", () => ({
    axiosAuthorized: { put: jest.fn() },
}));

/** The keys `useSuspenseFetch` was asked to cache under. */
const mockSuspenseKeys: unknown[] = [];
jest.mock("src/core/utils/reactQuery", () => {
    const actual = jest.requireActual("src/core/utils/reactQuery");
    return {
        ...actual,
        useSuspenseFetch: (url: string, params?: never) => {
            mockSuspenseKeys.push(actual.generateQueryKey(url, params));
            return null;
        },
    };
});

const put = axiosAuthorized.put as jest.Mock;

beforeEach(() => {
    put.mockReset();
    mockSuspenseKeys.length = 0;
});

describe("setByokFallbackChain", () => {
    it("PUTs the ids and resolves to the chain as stored", async () => {
        put.mockResolvedValue({ data: { fallbackModelIds: ["model-b"] } });

        await expect(
            setByokFallbackChain(["model-b", "model-a"]),
        ).resolves.toEqual(["model-b"]);
        expect(put).toHaveBeenCalledWith(
            ORGANIZATION_PARAMETERS_PATHS.SET_BYOK_FALLBACK_CHAIN,
            { modelIds: ["model-b", "model-a"] },
        );
    });

    it("resolves to an empty chain when the envelope carries none", async () => {
        put.mockResolvedValue({});

        await expect(setByokFallbackChain([])).resolves.toEqual([]);
    });
});

describe("useSetByokFallbackChain", () => {
    const setup = () => {
        const queryClient = new QueryClient({
            defaultOptions: { mutations: { retry: false } },
        });
        const invalidate = jest.spyOn(queryClient, "invalidateQueries");
        const wrapper = ({ children }: { children: ReactNode }) => (
            <QueryClientProvider client={queryClient}>
                {children}
            </QueryClientProvider>
        );
        return { invalidate, wrapper };
    };

    it("invalidates the key the BYOK config is cached under", async () => {
        const { invalidate, wrapper } = setup();
        renderHook(() => useSuspenseGetBYOK(), { wrapper });
        expect(mockSuspenseKeys).toHaveLength(1);

        put.mockResolvedValue({ data: { fallbackModelIds: ["model-a"] } });
        const { result } = renderHook(() => useSetByokFallbackChain(), {
            wrapper,
        });
        await act(async () => {
            await result.current.mutateAsync(["model-a"]);
        });

        expect(invalidate).toHaveBeenCalledWith({
            queryKey: mockSuspenseKeys[0],
        });
    });

    it("leaves the cache alone when the save fails", async () => {
        const { invalidate, wrapper } = setup();
        put.mockRejectedValue(new Error("400"));
        const { result } = renderHook(() => useSetByokFallbackChain(), {
            wrapper,
        });

        await act(async () => {
            await expect(result.current.mutateAsync(["x"])).rejects.toThrow(
                "400",
            );
        });
        expect(invalidate).not.toHaveBeenCalled();
    });
});
