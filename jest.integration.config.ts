import base from './jest.config.ts';

/**
 * The specs that share the test Postgres and Mongo (docker-compose.test.yml):
 * run one at a time, in band. Everything else runs in parallel workers
 * (jest.unit.config.ts); the two configs split jest.config.ts's specs between
 * them with no overlap and no gap.
 */
export const INTEGRATION_SPEC_GLOBS = [
    '**/*.integration.spec.ts',
    '**/test/integration/**/*.spec.ts',
    '**/*.e2e-spec.ts',
];

export default {
    ...base,
    testMatch: INTEGRATION_SPEC_GLOBS,
};
