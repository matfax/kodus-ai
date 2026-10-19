import base from './jest.config.ts';

/**
 * Every spec except those jest.integration.config.ts runs: none of these
 * touches a shared database, so they run in parallel workers. Separators
 * match both `/` and `\` so the split holds on Windows too.
 */
const SEP = '[/\\\\]';

export default {
    ...base,
    testPathIgnorePatterns: [
        ...base.testPathIgnorePatterns,
        '\\.integration\\.spec\\.ts$',
        `${SEP}test${SEP}integration${SEP}`,
        '\\.e2e-spec\\.ts$',
    ],
};
