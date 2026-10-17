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
        // Only the request (16th) and the chain use-case (17th, last) are
        // touched by this route; every other dependency stays unset.
        const unused = Array.from({ length: 15 }, () => undefined);
        const deps = [
            ...unused,
            { user: { organization: { uuid: orgId } } },
            setByokFallbackChainUseCase,
        ] as unknown as ConstructorParameters<
            typeof OrganizationParametersController
        >;
        return new OrganizationParametersController(...deps);
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
        // The decorators store their metadata on the method's function object.
        const handler = Object.getOwnPropertyDescriptor(
            OrganizationParametersController.prototype,
            'setByokFallbackChain',
        )?.value as object;
        const guards = Reflect.getMetadata(GUARDS_METADATA, handler);
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
