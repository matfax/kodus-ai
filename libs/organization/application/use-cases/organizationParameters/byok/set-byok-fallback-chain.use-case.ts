import { isByokConfig, type BYOKConfig } from '@libs/llm/byok-config';
import {
    fallbackChainOf,
    platformFundedTailEntries,
    requestedFallbackChain,
    withFallbackChain,
} from '@libs/llm/byok-fallback-chain';
import { validateByokConfigRefs } from '@libs/llm/validate-byok-config-refs';
import { OrganizationParametersKey } from '@libs/core/domain/enums';
import { IUseCase } from '@libs/core/domain/interfaces/use-case.interface';
import { OrganizationAndTeamData } from '@libs/core/infrastructure/config/types/general/organizationAndTeamData';
import { ActionType } from '@libs/core/infrastructure/config/types/general/codeReviewSettingsLog.type';
import { UserRequest } from '@libs/core/infrastructure/config/types/http/user-request.type';
import { createLogger } from '@libs/core/log/logger';
import { AuditLogEvents } from '@libs/ee/codeReviewSettingsLog/events/audit-log.events';
import {
    IOrganizationParametersService,
    ORGANIZATION_PARAMETERS_SERVICE_TOKEN,
} from '@libs/organization/domain/organizationParameters/contracts/organizationParameters.service.contract';
import { BadRequestException, Inject, Injectable } from '@nestjs/common';
import { REQUEST } from '@nestjs/core';
import { EventEmitter2 } from '@nestjs/event-emitter';

export interface SetByokFallbackChainResult {
    /** The chain as stored and as every runtime reader now sees it. */
    fallbackModelIds: string[];
}

/**
 * Replace the org's BYOK fallback chain, and nothing else.
 *
 * The one write surface for the chain outside the enterprise UI. The full BYOK
 * save (`POST /create-or-update`) takes the whole blob, masked secrets
 * included, so setting a chain through it means rebuilding every credential;
 * here the stored config is read, only `routing.fallbackModelId` (the head) and
 * `routing.fallbackModelIds` are replaced, and the credentials are written back
 * exactly as they were stored (same ciphertext, never decrypted or re-encrypted).
 *
 * Every id is one the request states, so there is no carried-over tail to
 * prune: an id that does not resolve, an over-long chain and a Kodus-funded
 * entry after the head are all rejected, with the same validator and the same
 * rules as the full save. An empty list clears the chain.
 */
@Injectable()
export class SetByokFallbackChainUseCase implements IUseCase {
    private readonly logger = createLogger(SetByokFallbackChainUseCase.name);

    constructor(
        @Inject(ORGANIZATION_PARAMETERS_SERVICE_TOKEN)
        private readonly organizationParametersService: IOrganizationParametersService,

        @Inject(REQUEST)
        private readonly request: UserRequest,

        private readonly eventEmitter: EventEmitter2,
    ) {}

    async execute(
        organizationAndTeamData: OrganizationAndTeamData,
        modelIds: unknown,
    ): Promise<SetByokFallbackChainResult> {
        const requested = this.parseModelIds(modelIds);

        const stored = await this.organizationParametersService.findByKey(
            OrganizationParametersKey.BYOK_CONFIG,
            organizationAndTeamData,
        );
        const existing = stored?.configValue as BYOKConfig | undefined;
        if (!isByokConfig(existing)) {
            throw new BadRequestException(
                'A fallback chain needs a BYOK configuration with models to choose from',
            );
        }

        // Head first, deduped, NOT capped: an over-long request is rejected by
        // the validator below instead of being silently trimmed.
        const chain = requestedFallbackChain({ fallbackModelIds: requested });
        const next: BYOKConfig = {
            ...existing,
            routing: withFallbackChain(existing.routing ?? {}, chain),
        };

        const refCheck = validateByokConfigRefs(next);
        if (!refCheck.valid) {
            throw new BadRequestException({
                message:
                    'Invalid BYOK configuration: unresolved model/routing references',
                errors: refCheck.errors,
            });
        }
        this.assertNoPlatformFundedTail(next);

        await this.organizationParametersService.createOrUpdateConfig(
            OrganizationParametersKey.BYOK_CONFIG,
            next,
            organizationAndTeamData,
        );

        const previousChain = fallbackChainOf(existing.routing);
        const nextChain = fallbackChainOf(next.routing);

        // The enterprise audit log records only the head (`fallbackModelId`),
        // so the tail's server-side trail is this warn, as on the full save.
        // Ids only.
        if (
            previousChain.length !== nextChain.length ||
            previousChain.some((id, i) => id !== nextChain[i])
        ) {
            this.logger.warn({
                message: nextChain.length
                    ? 'BYOK fallback chain changed'
                    : 'BYOK fallback chain cleared',
                context: SetByokFallbackChainUseCase.name,
                metadata: {
                    organizationId: organizationAndTeamData.organizationId,
                    userId: this.request.user?.uuid,
                    previousChain,
                    chain: nextChain,
                },
            });
        }

        this.eventEmitter.emit(AuditLogEvents.ORG_SETTINGS, {
            organizationAndTeamData,
            userInfo: {
                userId: this.request.user?.uuid,
                userEmail: this.request.user?.email,
            },
            actionType: ActionType.EDIT,
            settingKey: OrganizationParametersKey.BYOK_CONFIG,
            previousValue: existing,
            currentValue: next,
        });

        return { fallbackModelIds: nextChain };
    }

    /** The body is untyped JSON: accept only a list of non-blank id strings. */
    private parseModelIds(modelIds: unknown): string[] {
        if (
            !Array.isArray(modelIds) ||
            modelIds.some((id) => typeof id !== 'string' || !id.trim())
        ) {
            throw new BadRequestException(
                'modelIds must be a list of BYOK model ids',
            );
        }
        return modelIds.map((id: string) => id.trim());
    }

    /**
     * Same rule as the full save: a Kodus-funded model may only be the head,
     * the one entry the enterprise credit-balance alarm can see.
     */
    private assertNoPlatformFundedTail(config: BYOKConfig): void {
        const offending = platformFundedTailEntries(config);
        if (offending.length > 0) {
            throw new BadRequestException({
                message:
                    'Invalid BYOK configuration: a Kodus-funded model can only be the first fallback',
                errors: offending.map(
                    (id) =>
                        `routing.fallbackModelIds "${id}" uses the Kodus provider, which is not allowed after the first fallback`,
                ),
            });
        }
    }
}
