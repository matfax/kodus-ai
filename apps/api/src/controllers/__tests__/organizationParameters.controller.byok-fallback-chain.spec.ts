import { BadRequestException } from '@nestjs/common';
import { GUARDS_METADATA } from '@nestjs/common/constants';

import { PolicyGuard } from '@libs/identity/infrastructure/adapters/services/permissions/policy.guard';

import { OrganizationParametersController } from '../organizationParameters.controller';
import { buildMatrix } from './rbac-matrix.shared';

/**
 * PUT /organization-parameters/byok-fallback-chain: the HTTP layer only. The
 * body's `modelIds` reach SetByokFallbackChainUseCase scoped to the caller's
 * org, and the route carries the same policy as the full BYOK save.
 */
describe('OrganizationParametersController.setByokFallbackChain', () => {
    let setByokFallbackChainUseCase: { execute: jest.Mock };

    function buildController(orgId: string | undefined) {
        setByokFallbackChainUseCase = {
            execute: jest
                .fn()
                .mockResolvedValue({ fallbackModelIds: ['model-b'] }),
        };
        return new OrganizationParametersController(
            undefined as any, // createOrUpdateOrganizationParametersUseCase
            undefined as any, // findByKeyOrganizationParametersUseCase
            undefined as any, // getModelsByProviderUseCase
            undefined as any, // getModelCapabilitiesUseCase
            undefined as any, // providerService
            undefined as any, // deleteByokConfigUseCase
            undefined as any, // getLLMConfigStatusUseCase
            undefined as any, // getByokProvidersUseCase
            undefined as any, // testByokConnectionUseCase
            undefined as any, // testByokModelUseCase
            undefined as any, // listModelOverridesUseCase
            undefined as any, // clearModelOverridesUseCase
            undefined as any, // getCockpitMetricsVisibilityUseCase
            undefined as any, // ignoreBotsUseCase
            undefined as any, // updateAutoLicenseAllowedUsersUseCase
            { user: { organization: { uuid: orgId } } } as any, // request
            setByokFallbackChainUseCase as any,
        );
    }

    it('passes the body modelIds to the use-case, scoped to the org', async () => {
        const controller = buildController('org-1');

        await expect(
            controller.setByokFallbackChain({ modelIds: ['model-b'] }),
        ).resolves.toEqual({ fallbackModelIds: ['model-b'] });
        expect(setByokFallbackChainUseCase.execute).toHaveBeenCalledWith(
            { organizationId: 'org-1' },
            ['model-b'],
        );
    });

    it('400s without an organization, without touching the use-case', async () => {
        const controller = buildController(undefined);

        await expect(
            controller.setByokFallbackChain({ modelIds: ['model-b'] }),
        ).rejects.toBeInstanceOf(BadRequestException);
        expect(setByokFallbackChainUseCase.execute).not.toHaveBeenCalled();
    });

    it('runs behind PolicyGuard', () => {
        const guards = Reflect.getMetadata(
            GUARDS_METADATA,
            OrganizationParametersController.prototype.setByokFallbackChain,
        );
        expect(guards).toContain(PolicyGuard);
    });

    it('lets the same roles in as the full BYOK save: the owner only', async () => {
        const matrix = await buildMatrix();
        const row =
            matrix['organizationParameters.controller.ts#setByokFallbackChain'];

        expect(row).toEqual(
            matrix['organizationParameters.controller.ts#createOrUpdate'],
        );
        expect(row).toEqual({
            owner: 'allow',
            billing_manager: 'deny',
            repo_admin: 'deny',
            contributor: 'deny',
        });
    });
});
