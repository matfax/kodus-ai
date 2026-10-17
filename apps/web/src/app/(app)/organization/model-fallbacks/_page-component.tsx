"use client";

import { useMemo, useState, type Dispatch } from "react";
import { Badge } from "@components/ui/badge";
import { Button } from "@components/ui/button";
import { Card, CardContent } from "@components/ui/card";
import { Link } from "@components/ui/link";
import { Page } from "@components/ui/page";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@components/ui/select";
import { toast } from "@components/ui/toaster/use-toast";
import {
    fallbackChainOf,
    MAX_FALLBACKS,
} from "@services/organizationParameters/fallback-chain";
import {
    useSetByokFallbackChain,
    useSuspenseGetBYOK,
} from "@services/organizationParameters/hooks";
import { usePermission } from "@services/permissions/hooks";
import { Action, ResourceType } from "@services/permissions/types";
import { isAxiosError } from "axios";
import {
    ArrowDownIcon,
    ArrowUpIcon,
    Layers3Icon,
    TriangleAlertIcon,
    XIcon,
} from "lucide-react";
import { useUnsavedChangesGuard } from "src/core/hooks/use-unsaved-changes-guard";
import { revalidateServerSidePath } from "src/core/utils/revalidate-server-side";
import type { BYOKConfig } from "src/features/ee/byok/_types";

import {
    chainProblem,
    fallbackPool,
    moveEntry,
    type FallbackPoolModel,
} from "./_utils";

/** The API's error body carries a readable `message`; anything else is generic. */
const saveErrorMessage = (error: unknown): string => {
    const message = isAxiosError<{ message?: unknown } | undefined>(error)
        ? error.response?.data?.message
        : undefined;
    return typeof message === "string" && message
        ? message
        : "Please try again.";
};

export const ModelFallbacksPage = () => {
    const config = useSuspenseGetBYOK()?.configValue;
    const stored = useMemo(() => fallbackChainOf(config), [config]);

    // Keyed on the stored chain, so a save (which refetches the config) or a
    // change made on the AI providers page re-seeds the editor from the server.
    return (
        <ModelFallbacksEditor
            key={stored.join("\n")}
            config={config}
            stored={stored}
        />
    );
};

const ModelFallbacksEditor = ({
    config,
    stored,
}: {
    config?: BYOKConfig;
    stored: string[];
}) => {
    // The same gate as the endpoint (and the full BYOK save).
    const canEdit = usePermission(
        Action.Create,
        ResourceType.OrganizationSettings,
    );
    const pool = useMemo(() => fallbackPool(config), [config]);
    const [chain, setChain] = useState<string[]>(stored);
    const { save, isSaving } = useSaveFallbackChain(chain);

    const isDirty =
        chain.length !== stored.length ||
        chain.some((id, index) => id !== stored.at(index));
    const problem = chainProblem(chain, pool);

    useUnsavedChangesGuard({
        id: "model-fallbacks",
        isDirty,
        onBlock: () =>
            toast({
                variant: "warning",
                title: "Unsaved fallback changes",
                description:
                    "Save the fallback models you changed, or reset them, before leaving this page.",
            }),
    });

    return (
        <Page.Root>
            <Page.Header sticky>
                <Page.TitleContainer>
                    <Page.Title>Model fallbacks</Page.Title>
                    <Page.Description>
                        When a model call fails with an outage, a timeout or an
                        exhausted rate limit, Kody runs it again on these
                        models, in this order. Up to {MAX_FALLBACKS}.
                    </Page.Description>
                </Page.TitleContainer>
                <Page.SaveActions
                    isDirty={isDirty}
                    isSaving={isSaving}
                    canSave={canEdit && !problem}
                    onReset={() => {
                        setChain(stored);
                    }}
                    onSave={save}
                    saveLabel="Save fallbacks"
                />
            </Page.Header>

            <Page.Content>
                <div className="flex w-full max-w-3xl flex-col gap-4">
                    {pool.length === 0 ? (
                        <NoModels />
                    ) : (
                        <ChainCard
                            chain={chain}
                            pool={pool}
                            problem={problem}
                            disabled={!canEdit}
                            onChange={setChain}
                        />
                    )}

                    <p className="text-text-tertiary text-xs">
                        The routing tab on{" "}
                        <Link href="/byok?tab=routing">AI providers</Link> shows
                        the first fallback. Turning the fallback off there
                        clears this whole list.
                    </p>
                </div>
            </Page.Content>
        </Page.Root>
    );
};

/** Saves `chain` and reports the outcome in a toast. */
const useSaveFallbackChain = (chain: string[]) => {
    const setFallbackChain = useSetByokFallbackChain();

    const save = async () => {
        try {
            await setFallbackChain.mutateAsync(chain);
            // The AI providers routing tab shows the first entry. Best effort:
            // the chain is saved, so a failed revalidation is not a failed save.
            revalidateServerSidePath("/byok").catch(() => undefined);
            toast({
                variant: "success",
                description: chain.length
                    ? "Fallback models saved"
                    : "Fallback models cleared",
            });
        } catch (error) {
            toast({
                variant: "danger",
                title: "Couldn't save the fallback models",
                description: saveErrorMessage(error),
            });
        }
    };

    return { save, isSaving: setFallbackChain.isPending };
};

const ChainCard = ({
    chain,
    pool,
    problem,
    disabled,
    onChange,
}: {
    chain: string[];
    pool: FallbackPoolModel[];
    problem: string | null | undefined;
    disabled: boolean;
    onChange: Dispatch<string[]>;
}) => (
    <Card color="lv1">
        <CardContent className="flex flex-col gap-4 pt-6">
            {chain.length === 0 ? (
                <p className="text-text-secondary text-sm">
                    No fallback models. A call that fails is not retried on
                    another model.
                </p>
            ) : (
                <ol className="flex flex-col gap-2">
                    {chain.map((id, index) => (
                        <ChainEntry
                            key={id}
                            id={id}
                            index={index}
                            count={chain.length}
                            model={pool.find((m) => m.id === id)}
                            disabled={disabled}
                            onMove={(offset) => {
                                onChange(moveEntry(chain, index, offset));
                            }}
                            onRemove={() => {
                                onChange(chain.filter((entry) => entry !== id));
                            }}
                        />
                    ))}
                </ol>
            )}

            {problem && (
                <p className="text-danger flex items-center gap-2 text-sm">
                    <TriangleAlertIcon className="size-4 shrink-0" />
                    {problem}
                </p>
            )}

            <AddEntry
                options={pool.filter((model) => !chain.includes(model.id))}
                chainLength={chain.length}
                disabled={disabled}
                onAdd={(id) => {
                    onChange([...chain, id]);
                }}
            />
        </CardContent>
    </Card>
);

const ChainEntry = ({
    id,
    index,
    count,
    model,
    disabled,
    onMove,
    onRemove,
}: {
    id: string;
    index: number;
    count: number;
    model: FallbackPoolModel | undefined;
    disabled: boolean;
    onMove: Dispatch<number>;
    onRemove: () => void;
}) => {
    const label = model?.model ?? id;

    return (
        <li className="bg-card-lv2 flex items-center gap-3 rounded-xl px-4 py-2">
            <span className="text-text-tertiary w-4 text-sm font-semibold">
                {index + 1}
            </span>
            <div className="flex min-w-0 flex-1 flex-col">
                <span className="truncate font-mono text-sm">{label}</span>
                <span className="text-text-tertiary text-xs">
                    {model ? model.provider : "No longer connected"}
                </span>
            </div>
            {model?.platformFunded && <Badge>Kodus credits</Badge>}
            <Button
                type="button"
                size="icon-sm"
                variant="cancel"
                aria-label={`Move ${label} up`}
                disabled={disabled || index === 0}
                onClick={() => {
                    onMove(-1);
                }}>
                <ArrowUpIcon />
            </Button>
            <Button
                type="button"
                size="icon-sm"
                variant="cancel"
                aria-label={`Move ${label} down`}
                disabled={disabled || index === count - 1}
                onClick={() => {
                    onMove(1);
                }}>
                <ArrowDownIcon />
            </Button>
            <Button
                type="button"
                size="icon-sm"
                variant="cancel"
                aria-label={`Remove ${label}`}
                disabled={disabled}
                onClick={onRemove}>
                <XIcon />
            </Button>
        </li>
    );
};

const AddEntry = ({
    options,
    chainLength,
    disabled,
    onAdd,
}: {
    options: FallbackPoolModel[];
    chainLength: number;
    disabled: boolean;
    onAdd: Dispatch<string>;
}) => {
    const full = chainLength >= MAX_FALLBACKS;

    return (
        // Controlled with an empty value, so the trigger always reads as the
        // prompt and picking a model only appends it.
        <Select value="" onValueChange={onAdd}>
            <SelectTrigger
                size="md"
                className="w-full sm:w-80"
                disabled={disabled || full || options.length === 0}>
                <SelectValue
                    placeholder={
                        full
                            ? `${MAX_FALLBACKS} fallbacks is the limit`
                            : options.length === 0
                              ? "Every connected model is in the list"
                              : "Add a fallback model"
                    }
                />
            </SelectTrigger>
            <SelectContent>
                {options.map((model) => (
                    // Kodus credits may only pay for the first fallback.
                    <SelectItem
                        key={model.id}
                        value={model.id}
                        disabled={model.platformFunded && chainLength > 0}>
                        {model.model}{" "}
                        <span className="text-text-tertiary">
                            · {model.provider}
                        </span>
                    </SelectItem>
                ))}
            </SelectContent>
        </Select>
    );
};

const NoModels = () => (
    <Card color="lv1">
        <CardContent className="flex flex-col items-center gap-3 py-10 text-center">
            <span className="text-text-tertiary">
                <Layers3Icon size={24} />
            </span>
            <p className="text-text-secondary max-w-md text-sm text-balance">
                Connect a model on AI providers first. A fallback is one of the
                models you have connected.
            </p>
            <Link href="/byok" noHoverUnderline>
                <Button decorative variant="primary" size="md">
                    Go to AI providers
                </Button>
            </Link>
        </CardContent>
    </Card>
);
