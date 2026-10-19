import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        include: ['src/**/__tests__/**/*.test.ts'],
        // A few tests use `vi.doMock('os', …)` to point `homedir()` at a
        // tempdir. With Vitest's default file-level parallelism that mock
        // can leak across workers and flake unrelated FS-touching tests
        // (config/device/credentials utils). Running test files serially
        // costs ~7s on the full suite and removes the flake — worth it.
        fileParallelism: false,
        // The e2e tests commit in temp repos, and git reads the developer's
        // global config: with commit/tag signing on, every test commit asks
        // for a signature (and fails when it is declined). These env-level
        // config entries outrank ~/.gitconfig and reach every spawned git.
        env: {
            GIT_CONFIG_COUNT: '2',
            GIT_CONFIG_KEY_0: 'commit.gpgsign',
            GIT_CONFIG_VALUE_0: 'false',
            GIT_CONFIG_KEY_1: 'tag.gpgsign',
            GIT_CONFIG_VALUE_1: 'false',
        },
    },
});
