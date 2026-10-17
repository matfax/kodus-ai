/** @jest-environment jsdom */
import "@testing-library/jest-dom";

import type { ReactNode } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { AxiosError, AxiosHeaders } from "axios";
import type { BYOKConfig } from "src/features/ee/byok/_types";

import { ModelFallbacksPage } from "./_page-component";

const mutateAsync = jest.fn();
let stored: BYOKConfig;
let mockCanEdit = true;

jest.mock("@services/organizationParameters/hooks", () => ({
    useSuspenseGetBYOK: () => ({ configValue: stored }),
    useSetByokFallbackChain: () => ({ mutateAsync, isPending: false }),
}));
jest.mock("@services/permissions/hooks", () => ({
    usePermission: () => mockCanEdit,
}));
const mockToast = jest.fn();
jest.mock("@components/ui/toaster/use-toast", () => ({
    toast: (...args: unknown[]) => mockToast(...args),
}));
jest.mock("src/core/utils/revalidate-server-side", () => ({
    revalidateServerSidePath: jest.fn(() => Promise.resolve()),
}));
jest.mock("@components/ui/link", () => ({
    Link: ({ href, children }: { href: string; children: ReactNode }) => (
        <a href={href}>{children}</a>
    ),
}));

const config = (routing: Record<string, unknown>): BYOKConfig => ({
    version: 2,
    credentials: [
        { id: "cred-openai", provider: "openai", apiKey: "sk...abc" },
        { id: "cred-kodus", provider: "kodus" },
    ],
    models: [
        { id: "model-a", credentialId: "cred-openai", model: "gpt-5" },
        { id: "model-b", credentialId: "cred-openai", model: "gpt-5-mini" },
        { id: "model-k", credentialId: "cred-kodus", model: "kodus-1" },
    ],
    routing: { defaultModelId: "model-a", ...routing },
});

const saveButton = () =>
    screen.getByRole("button", { name: /save fallbacks/i });
const entries = () =>
    screen.getAllByRole("listitem").map((item) => item.textContent ?? "");

beforeEach(() => {
    mutateAsync.mockReset().mockResolvedValue(["model-b"]);
    mockToast.mockClear();
    mockCanEdit = true;
});

/** An API 400 as axios delivers it, with the use-case's message. */
const apiError = (message: string) => {
    const config = { headers: new AxiosHeaders() };
    return new AxiosError(
        "Request failed with status code 400",
        "ERR_BAD_REQUEST",
        config,
        null,
        {
            status: 400,
            statusText: "Bad Request",
            headers: {},
            config,
            data: { message },
        },
    );
};

describe("ModelFallbacksPage", () => {
    it("lists the stored chain in order, with nothing to save yet", () => {
        stored = config({
            fallbackModelId: "model-b",
            fallbackModelIds: ["model-b", "model-a"],
        });
        render(<ModelFallbacksPage />);

        expect(entries()).toEqual([
            expect.stringContaining("gpt-5-mini"),
            expect.stringContaining("gpt-5"),
        ]);
        expect(saveButton()).toBeDisabled();
    });

    it("saves a reordered chain", async () => {
        stored = config({
            fallbackModelId: "model-b",
            fallbackModelIds: ["model-b", "model-a"],
        });
        render(<ModelFallbacksPage />);

        fireEvent.click(screen.getByRole("button", { name: "Move gpt-5 up" }));
        expect(saveButton()).toBeEnabled();
        await act(async () => {
            fireEvent.click(saveButton());
        });

        expect(mutateAsync).toHaveBeenCalledWith(["model-a", "model-b"]);
    });

    it("saves an empty chain when the last entry is removed", async () => {
        stored = config({ fallbackModelId: "model-b" });
        render(<ModelFallbacksPage />);

        fireEvent.click(
            screen.getByRole("button", { name: "Remove gpt-5-mini" }),
        );
        expect(screen.getByText(/No fallback models/i)).toBeInTheDocument();
        await act(async () => {
            fireEvent.click(saveButton());
        });

        expect(mutateAsync).toHaveBeenCalledWith([]);
    });

    it("shows the API's message when the save is refused", async () => {
        stored = config({ fallbackModelId: "model-b" });
        mutateAsync.mockRejectedValue(
            apiError("Fallback model model-gone is not connected"),
        );
        render(<ModelFallbacksPage />);

        fireEvent.click(
            screen.getByRole("button", { name: "Remove gpt-5-mini" }),
        );
        await act(async () => {
            fireEvent.click(saveButton());
        });

        expect(mockToast).toHaveBeenCalledWith(
            expect.objectContaining({
                variant: "danger",
                description: "Fallback model model-gone is not connected",
            }),
        );
    });

    it("falls back to a generic message when the error carries none", async () => {
        stored = config({ fallbackModelId: "model-b" });
        mutateAsync.mockRejectedValue(new Error("Network Error"));
        render(<ModelFallbacksPage />);

        fireEvent.click(
            screen.getByRole("button", { name: "Remove gpt-5-mini" }),
        );
        await act(async () => {
            fireEvent.click(saveButton());
        });

        expect(mockToast).toHaveBeenCalledWith(
            expect.objectContaining({
                variant: "danger",
                description: "Please try again.",
            }),
        );
    });

    it("lets a member without the permission see the chain but not change it", () => {
        mockCanEdit = false;
        stored = config({
            fallbackModelId: "model-b",
            fallbackModelIds: ["model-b", "model-a"],
        });
        render(<ModelFallbacksPage />);

        expect(
            screen.getByRole("button", { name: "Move gpt-5 up" }),
        ).toBeDisabled();
        expect(
            screen.getByRole("button", { name: "Remove gpt-5-mini" }),
        ).toBeDisabled();
        expect(saveButton()).toBeDisabled();
    });

    it("blocks the save while a Kodus-funded model is not first", () => {
        stored = config({
            fallbackModelId: "model-k",
            fallbackModelIds: ["model-k", "model-a"],
        });
        render(<ModelFallbacksPage />);

        fireEvent.click(
            screen.getByRole("button", { name: "Move kodus-1 down" }),
        );

        expect(
            screen.getByText(/Kodus-funded model can only be the first/i),
        ).toBeInTheDocument();
        expect(saveButton()).toBeDisabled();
    });

    it("flags a stored entry that is no longer connected, so it can be removed", () => {
        stored = config({
            fallbackModelId: "model-b",
            fallbackModelIds: ["model-b", "model-gone"],
        });
        render(<ModelFallbacksPage />);

        expect(screen.getByText("No longer connected")).toBeInTheDocument();
        fireEvent.click(
            screen.getByRole("button", { name: "Move gpt-5-mini down" }),
        );
        expect(saveButton()).toBeDisabled();

        fireEvent.click(
            screen.getByRole("button", { name: "Remove model-gone" }),
        );
        expect(saveButton()).toBeEnabled();
    });
});
