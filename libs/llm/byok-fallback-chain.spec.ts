import type { BYOKConfig, BYOKRouting } from './byok-config';
import {
    fallbackChainOf,
    MAX_FALLBACKS,
    mergeFallbackChainOnWrite,
    platformFundedTailEntries,
    removeFromFallbackTail,
    requestedFallbackChain,
    withFallbackChain,
} from './byok-fallback-chain';

const v2 = (routing?: BYOKRouting, extraModels: string[] = []): BYOKConfig => ({
    version: 2,
    credentials: [
        { id: 'cred-openai', provider: 'openai', apiKey: 'enc(k)' },
        { id: 'cred-kodus', provider: 'kodus' },
    ],
    models: [
        ...['a', 'b', 'c', 'd', 'e', ...extraModels].map((id) => ({
            id,
            credentialId: 'cred-openai',
            model: `gpt-${id}`,
        })),
        { id: 'k', credentialId: 'cred-kodus', model: 'kodus-model' },
    ],
    ...(routing ? { routing } : {}),
});

describe('fallbackChainOf — read-time normalization', () => {
    it('resolves a legacy config with only fallbackModelId to that one entry', () => {
        expect(fallbackChainOf({ fallbackModelId: 'a' })).toEqual(['a']);
    });

    it('is empty when no fallback is configured', () => {
        expect(fallbackChainOf(undefined)).toEqual([]);
        expect(fallbackChainOf({})).toEqual([]);
        expect(fallbackChainOf({ fallbackModelId: '' })).toEqual([]);
    });

    it('puts fallbackModelId first even when the list disagrees (head wins)', () => {
        expect(
            fallbackChainOf({
                fallbackModelId: 'c',
                fallbackModelIds: ['a', 'b', 'c'],
            }),
        ).toEqual(['c', 'a', 'b']);
    });

    it('uses the list alone when no head is set', () => {
        expect(fallbackChainOf({ fallbackModelIds: ['b', 'a'] })).toEqual([
            'b',
            'a',
        ]);
    });

    it('dedupes and drops blank or non-string entries', () => {
        expect(
            fallbackChainOf({
                fallbackModelId: 'a',
                fallbackModelIds: ['a', 'b', '', 'b', 7 as unknown as string],
            }),
        ).toEqual(['a', 'b']);
    });

    it(`caps the chain at MAX_FALLBACKS (${MAX_FALLBACKS})`, () => {
        const routing = { fallbackModelIds: ['a', 'b', 'c', 'd', 'e'] };
        expect(fallbackChainOf(routing)).toEqual(['a', 'b', 'c']);
        // The uncapped view is what the write validator rejects.
        expect(requestedFallbackChain(routing)).toHaveLength(5);
    });
});

describe('withFallbackChain — stored shape', () => {
    it('stores only the head for a one-entry chain (legacy shape kept)', () => {
        expect(withFallbackChain({ mode: 'manual' }, ['a'])).toEqual({
            mode: 'manual',
            fallbackModelId: 'a',
        });
    });

    it('stores the head and the whole chain when there is a tail', () => {
        expect(withFallbackChain({}, ['a', 'b'])).toEqual({
            fallbackModelId: 'a',
            fallbackModelIds: ['a', 'b'],
        });
    });

    it('removes both fields for an empty chain', () => {
        expect(
            withFallbackChain(
                {
                    defaultModelId: 'a',
                    fallbackModelId: 'b',
                    fallbackModelIds: ['b', 'c'],
                },
                [],
            ),
        ).toEqual({ defaultModelId: 'a' });
    });
});

describe('mergeFallbackChainOnWrite', () => {
    const stored = v2({
        defaultModelId: 'a',
        fallbackModelId: 'b',
        fallbackModelIds: ['b', 'c', 'd'],
    });

    it('a routing-tab save (no fallbackModelIds key) keeps the stored tail', () => {
        const incoming = v2({ defaultModelId: 'e', fallbackModelId: 'b' });
        const { config, previousTail, tail } = mergeFallbackChainOnWrite(
            incoming,
            stored,
        );
        expect(config.routing).toEqual({
            defaultModelId: 'e',
            fallbackModelId: 'b',
            fallbackModelIds: ['b', 'c', 'd'],
        });
        expect(previousTail).toEqual(['c', 'd']);
        expect(tail).toEqual(['c', 'd']);
    });

    it('a routing-tab save that changes the head keeps the tail behind the new head', () => {
        const { config } = mergeFallbackChainOnWrite(
            v2({ fallbackModelId: 'e' }),
            stored,
        );
        expect(fallbackChainOf(config.routing)).toEqual(['e', 'c', 'd']);
        expect(config.routing?.fallbackModelId).toBe('e');
    });

    it('a new head that was in the tail moves to the front without a duplicate', () => {
        const { config } = mergeFallbackChainOnWrite(
            v2({ fallbackModelId: 'c' }),
            stored,
        );
        expect(fallbackChainOf(config.routing)).toEqual(['c', 'd']);
        // The persisted shape, not just the normalized read, has no duplicate.
        expect(config.routing?.fallbackModelId).toBe('c');
        expect(config.routing?.fallbackModelIds).toEqual(['c', 'd']);
    });

    it('"no fallback" on a routing-tab save clears the whole chain', () => {
        const { config, tail } = mergeFallbackChainOnWrite(
            v2({ defaultModelId: 'a' }),
            stored,
        );
        expect(config.routing).toEqual({ defaultModelId: 'a' });
        expect(tail).toEqual([]);
    });

    it('an explicit fallbackModelIds is used as given and fallbackModelId follows it', () => {
        const { config } = mergeFallbackChainOnWrite(
            v2({ fallbackModelIds: ['e', 'a'] }),
            stored,
        );
        expect(config.routing).toEqual({
            fallbackModelId: 'e',
            fallbackModelIds: ['e', 'a'],
        });
    });

    it('an explicit empty list with no head clears the chain', () => {
        const { config } = mergeFallbackChainOnWrite(
            v2({ fallbackModelIds: [] }),
            stored,
        );
        expect(fallbackChainOf(config.routing)).toEqual([]);
    });

    it('does not cap an explicit list, so the validator can reject it', () => {
        const { config } = mergeFallbackChainOnWrite(
            v2({ fallbackModelIds: ['a', 'b', 'c', 'd', 'e'] }),
            undefined,
        );
        expect(requestedFallbackChain(config.routing)).toHaveLength(5);
    });

    it('prunes a stored tail entry whose model no longer exists', () => {
        // The incoming blob dropped model "d" (removed in the UI) while the
        // stored chain still lists it.
        const incoming = v2({ fallbackModelId: 'b' });
        incoming.models = incoming.models.filter((m) => m.id !== 'd');
        const { config, pruned, tail } = mergeFallbackChainOnWrite(
            incoming,
            stored,
        );
        expect(fallbackChainOf(config.routing)).toEqual(['b', 'c']);
        expect(pruned).toEqual([{ modelId: 'd', reason: 'unknown-model' }]);
        expect(tail).toEqual(['c']);
    });

    it('prunes a stale stored entry even when the request echoes it back', () => {
        const incoming = v2({ fallbackModelIds: ['b', 'c', 'd'] });
        incoming.models = incoming.models.filter((m) => m.id !== 'c');
        const { config, pruned } = mergeFallbackChainOnWrite(incoming, stored);
        expect(fallbackChainOf(config.routing)).toEqual(['b', 'd']);
        expect(pruned).toEqual([{ modelId: 'c', reason: 'unknown-model' }]);
    });

    it('leaves an unknown entry the request ADDED for the validator', () => {
        const { config, pruned } = mergeFallbackChainOnWrite(
            v2({ fallbackModelIds: ['b', 'ghost'] }),
            stored,
        );
        expect(requestedFallbackChain(config.routing)).toEqual(['b', 'ghost']);
        expect(pruned).toEqual([]);
    });

    it('never prunes the head', () => {
        const incoming = v2({ fallbackModelId: 'ghost' });
        const { config } = mergeFallbackChainOnWrite(incoming, stored);
        expect(config.routing?.fallbackModelId).toBe('ghost');
    });

    it('leaves a blob without a routing block untouched', () => {
        const incoming = v2();
        expect(mergeFallbackChainOnWrite(incoming, stored).config).toBe(
            incoming,
        );
    });
});

describe('platformFundedTailEntries', () => {
    it('lists a kodus entry in the tail', () => {
        expect(
            platformFundedTailEntries(v2({ fallbackModelIds: ['a', 'k'] })),
        ).toEqual(['k']);
    });

    it('allows a kodus head', () => {
        expect(
            platformFundedTailEntries(v2({ fallbackModelIds: ['k', 'a'] })),
        ).toEqual([]);
    });
});

describe('removeFromFallbackTail', () => {
    it('prunes a tail entry', () => {
        expect(
            removeFromFallbackTail(
                { fallbackModelId: 'a', fallbackModelIds: ['a', 'b', 'c'] },
                'b',
            ),
        ).toEqual({ fallbackModelId: 'a', fallbackModelIds: ['a', 'c'] });
    });

    it('collapses to the legacy shape when only the head is left', () => {
        expect(
            removeFromFallbackTail(
                { fallbackModelId: 'a', fallbackModelIds: ['a', 'b'] },
                'b',
            ),
        ).toEqual({ fallbackModelId: 'a' });
    });

    it('returns the same routing when the model is not a tail entry', () => {
        const routing = { fallbackModelId: 'a', fallbackModelIds: ['a', 'b'] };
        expect(removeFromFallbackTail(routing, 'a')).toBe(routing);
        expect(removeFromFallbackTail(routing, 'z')).toBe(routing);
    });
});
