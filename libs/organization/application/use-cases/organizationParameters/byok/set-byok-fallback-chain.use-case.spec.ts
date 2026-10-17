import { BadRequestException } from '@nestjs/common';
import { encrypt } from '@libs/common/utils/crypto';
import { OrganizationParametersKey } from '@libs/core/domain/enums';
import type { BYOKConfig } from '@libs/llm/byok-config';
import { MAX_FALLBACKS } from '@libs/llm/byok-fallback-chain';
import { AuditLogEvents } from '@libs/ee/codeReviewSettingsLog/events/audit-log.events';
import { SetByokFallbackChainUseCase } from './set-byok-fallback-chain.use-case';

const orgAndTeam = { organizationId: 'org-1' };

/**
 * `existing` is what findByKey returns as the stored configValue (undefined →
 * no row). `persisted` captures what createOrUpdateConfig was asked to write.
 */
function buildUseCase(existing?: unknown) {
    const persisted: { value?: any } = {};
    const organizationParametersService = {
        findByKey: jest
            .fn()
            .mockResolvedValue(
                existing === undefined ? null : { configValue: existing },
            ),
        createOrUpdateConfig: jest.fn(async (_key, value: any) => {
            persisted.value = value;
            return true;
        }),
    };
    const request = { user: { uuid: 'user-1', email: 'u@k.io' } } as any;
    const eventEmitter = { emit: jest.fn() };

    const useCase = new SetByokFallbackChainUseCase(
        organizationParametersService as any,
        request,
        eventEmitter as any,
    );
    return { useCase, persisted, organizationParametersService, eventEmitter };
}

const openaiCipher = encrypt('sk-real-openai-key');
const anthropicCipher = encrypt('sk-ant-real-key');

/** Four BYOK models on two credentials, one Kodus-funded model. */
const stored = (routing: BYOKConfig['routing'] = {}): BYOKConfig => ({
    version: 2,
    credentials: [
        { id: 'cred-openai', provider: 'openai', apiKey: openaiCipher },
        {
            id: 'cred-anthropic',
            provider: 'anthropic',
            apiKey: anthropicCipher,
            settings: { baseURL: 'https://api.anthropic.com' },
        },
        { id: 'cred-kodus', provider: 'kodus' as any },
    ],
    models: [
        { id: 'model-a', credentialId: 'cred-openai', model: 'gpt-5' },
        { id: 'model-b', credentialId: 'cred-anthropic', model: 'claude-x' },
        { id: 'model-c', credentialId: 'cred-openai', model: 'gpt-5-mini' },
        { id: 'model-d', credentialId: 'cred-anthropic', model: 'claude-y' },
        { id: 'model-k', credentialId: 'cred-kodus', model: 'kodus-1' },
    ],
    routing: { defaultModelId: 'model-a', ...routing },
});

describe('SetByokFallbackChainUseCase', () => {
    it('writes the chain: the head to fallbackModelId, the whole list to fallbackModelIds', async () => {
        const { useCase, persisted, organizationParametersService } =
            buildUseCase(stored({ fallbackModelId: 'model-b' }));

        await expect(
            useCase.execute(orgAndTeam, ['model-c', 'model-b', 'model-d']),
        ).resolves.toEqual({
            fallbackModelIds: ['model-c', 'model-b', 'model-d'],
        });

        expect(
            organizationParametersService.createOrUpdateConfig,
        ).toHaveBeenCalledWith(
            OrganizationParametersKey.BYOK_CONFIG,
            expect.anything(),
            orgAndTeam,
        );
        expect(persisted.value.routing).toEqual({
            defaultModelId: 'model-a',
            fallbackModelId: 'model-c',
            fallbackModelIds: ['model-c', 'model-b', 'model-d'],
        });
    });

    it('stores a single entry in the legacy shape (head only, no list)', async () => {
        const { useCase, persisted } = buildUseCase(
            stored({
                fallbackModelId: 'model-b',
                fallbackModelIds: ['model-b', 'model-c'],
            }),
        );

        await useCase.execute(orgAndTeam, ['model-d']);

        expect(persisted.value.routing).toEqual({
            defaultModelId: 'model-a',
            fallbackModelId: 'model-d',
        });
    });

    it('clears both fallback fields on an empty list', async () => {
        const { useCase, persisted } = buildUseCase(
            stored({
                fallbackModelId: 'model-b',
                fallbackModelIds: ['model-b', 'model-c'],
            }),
        );

        await expect(useCase.execute(orgAndTeam, [])).resolves.toEqual({
            fallbackModelIds: [],
        });
        expect(persisted.value.routing).toEqual({ defaultModelId: 'model-a' });
    });

    it('leaves the credentials, models and the rest of routing byte-identical', async () => {
        const existing = stored({
            fallbackModelId: 'model-b',
            taskOverrides: { prSummary: 'model-c' },
        });
        const before = JSON.parse(JSON.stringify(existing));
        const { useCase, persisted } = buildUseCase(existing);

        await useCase.execute(orgAndTeam, ['model-c', 'model-d']);

        expect(JSON.stringify(persisted.value.credentials)).toBe(
            JSON.stringify(before.credentials),
        );
        expect(persisted.value.credentials[0].apiKey).toBe(openaiCipher);
        expect(persisted.value.credentials[1].apiKey).toBe(anthropicCipher);
        expect(persisted.value.models).toEqual(before.models);
        expect(persisted.value.version).toBe(2);
        expect(persisted.value.routing.taskOverrides).toEqual({
            prSummary: 'model-c',
        });
        // The stored object itself is not mutated.
        expect(existing).toEqual(before);
    });

    it('dedupes a repeated id, keeping the first position', async () => {
        const { useCase, persisted } = buildUseCase(stored());

        await useCase.execute(orgAndTeam, ['model-b', 'model-c', 'model-b']);

        expect(persisted.value.routing.fallbackModelIds).toEqual([
            'model-b',
            'model-c',
        ]);
    });

    it('rejects an id that names no model, and writes nothing', async () => {
        const { useCase, organizationParametersService, eventEmitter } =
            buildUseCase(stored({ fallbackModelId: 'model-b' }));

        const error = await useCase
            .execute(orgAndTeam, ['model-b', 'model-gone'])
            .catch((e) => e);

        expect(error).toBeInstanceOf(BadRequestException);
        expect(JSON.stringify(error.getResponse())).toContain('model-gone');
        expect(
            organizationParametersService.createOrUpdateConfig,
        ).not.toHaveBeenCalled();
        expect(eventEmitter.emit).not.toHaveBeenCalled();
    });

    it('rejects a stored tail entry that went stale instead of pruning it, since the request names it', async () => {
        const { useCase, organizationParametersService } = buildUseCase(
            stored({
                fallbackModelId: 'model-b',
                fallbackModelIds: ['model-b', 'model-gone'],
            }),
        );

        await expect(
            useCase.execute(orgAndTeam, ['model-b', 'model-gone']),
        ).rejects.toBeInstanceOf(BadRequestException);
        expect(
            organizationParametersService.createOrUpdateConfig,
        ).not.toHaveBeenCalled();
    });

    it(`rejects more than MAX_FALLBACKS (${MAX_FALLBACKS}) entries instead of trimming`, async () => {
        const { useCase, organizationParametersService } =
            buildUseCase(stored());

        const error = await useCase
            .execute(orgAndTeam, ['model-a', 'model-b', 'model-c', 'model-d'])
            .catch((e) => e);

        expect(error).toBeInstanceOf(BadRequestException);
        expect(JSON.stringify(error.getResponse())).toContain(
            `at most ${MAX_FALLBACKS}`,
        );
        expect(
            organizationParametersService.createOrUpdateConfig,
        ).not.toHaveBeenCalled();
    });

    it('rejects a Kodus-funded model after the head', async () => {
        const { useCase, organizationParametersService } =
            buildUseCase(stored());

        const error = await useCase
            .execute(orgAndTeam, ['model-b', 'model-k'])
            .catch((e) => e);

        expect(error).toBeInstanceOf(BadRequestException);
        expect(JSON.stringify(error.getResponse())).toContain('model-k');
        expect(
            organizationParametersService.createOrUpdateConfig,
        ).not.toHaveBeenCalled();
    });

    it('allows a Kodus-funded model as the head, as the routing tab does', async () => {
        const { useCase, persisted } = buildUseCase(stored());

        await useCase.execute(orgAndTeam, ['model-k', 'model-b']);

        expect(persisted.value.routing.fallbackModelId).toBe('model-k');
    });

    it.each([
        ['a non-array', 'model-b'],
        ['a missing body field', undefined],
        ['a non-string entry', ['model-b', 3]],
        ['a blank entry', ['model-b', '  ']],
    ])('rejects %s before reading the config', async (_label, modelIds) => {
        const { useCase, organizationParametersService } =
            buildUseCase(stored());

        await expect(
            useCase.execute(orgAndTeam, modelIds),
        ).rejects.toBeInstanceOf(BadRequestException);
        expect(organizationParametersService.findByKey).not.toHaveBeenCalled();
    });

    it('rejects an org without a BYOK config', async () => {
        const { useCase, organizationParametersService } = buildUseCase();

        await expect(
            useCase.execute(orgAndTeam, ['model-b']),
        ).rejects.toBeInstanceOf(BadRequestException);
        expect(
            organizationParametersService.createOrUpdateConfig,
        ).not.toHaveBeenCalled();
    });

    it('emits the ORG_SETTINGS audit event with the previous and new config', async () => {
        const existing = stored({ fallbackModelId: 'model-b' });
        const { useCase, persisted, eventEmitter } = buildUseCase(existing);

        await useCase.execute(orgAndTeam, ['model-c', 'model-b']);

        expect(eventEmitter.emit).toHaveBeenCalledTimes(1);
        expect(eventEmitter.emit).toHaveBeenCalledWith(
            AuditLogEvents.ORG_SETTINGS,
            expect.objectContaining({
                organizationAndTeamData: orgAndTeam,
                userInfo: { userId: 'user-1', userEmail: 'u@k.io' },
                settingKey: OrganizationParametersKey.BYOK_CONFIG,
                previousValue: existing,
                currentValue: persisted.value,
            }),
        );
    });
});
