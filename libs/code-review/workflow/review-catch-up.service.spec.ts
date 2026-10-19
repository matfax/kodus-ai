import { AutomationStatus } from '@libs/automation/domain/automation/enum/automation-status';
import { reviewCatchUpEnabled } from '@libs/code-review/domain/review-catch-up';

import {
    readCatchUpState,
    ReviewCatchUpService,
    type ReviewCatchUpState,
} from './review-catch-up.service';

const STATE: ReviewCatchUpState = {
    target: {
        organizationAndTeamData: { organizationId: 'org-1', teamId: 'team-1' },
        repository: { id: 'repo-1', name: 'repo' },
        pullRequest: { number: 42 },
        platformType: 'github' as any,
    },
    teamAutomationId: 'ta-1',
    pushHeadSha: 'sha-2',
};

const commits = (...shas: string[]) => shas.map((sha) => ({ sha }));

describe('ReviewCatchUpService.isCovered', () => {
    let executions: { findLatestExecutionByFilters: jest.Mock };
    let codeManagement: { getCommitsForPullRequestForCodeReview: jest.Mock };
    let service: ReviewCatchUpService;

    const analyzed = (lastAnalyzedCommit: unknown) =>
        executions.findLatestExecutionByFilters.mockResolvedValue({
            dataExecution: { lastAnalyzedCommit },
        });

    beforeEach(() => {
        executions = { findLatestExecutionByFilters: jest.fn() };
        codeManagement = {
            getCommitsForPullRequestForCodeReview: jest
                .fn()
                .mockResolvedValue(commits('sha-1', 'sha-2', 'sha-3')),
        };
        service = new ReviewCatchUpService(
            executions as any,
            codeManagement as any,
        );
    });

    it("reads the PR's latest successful review", async () => {
        analyzed('sha-3');
        await service.isCovered(STATE);
        expect(executions.findLatestExecutionByFilters).toHaveBeenCalledWith({
            status: AutomationStatus.SUCCESS,
            teamAutomation: { uuid: 'ta-1' },
            pullRequestNumber: 42,
            repositoryId: 'repo-1',
        });
    });

    it('is covered when that review analyzed the push head, without a commits call', async () => {
        analyzed({ sha: 'sha-2' });
        await expect(service.isCovered(STATE)).resolves.toBe(true);
        expect(
            codeManagement.getCommitsForPullRequestForCodeReview,
        ).not.toHaveBeenCalled();
    });

    it('is covered when that review analyzed a later commit', async () => {
        analyzed('sha-3');
        await expect(service.isCovered(STATE)).resolves.toBe(true);
    });

    it('is not covered when that review stopped at an earlier commit', async () => {
        analyzed('sha-1');
        await expect(service.isCovered(STATE)).resolves.toBe(false);
    });

    it.each([
        ['the push carried no head', { ...STATE, pushHeadSha: undefined }, 'sha-3'],
        ['no review analyzed a commit yet', STATE, undefined],
        ['the push head left the PR (force-push)', { ...STATE, pushHeadSha: 'gone' }, 'sha-3'],
    ])('is not covered when %s', async (_label, state, last) => {
        analyzed(last);
        await expect(service.isCovered(state)).resolves.toBe(false);
    });

    it('runs the catch-up when a lookup fails', async () => {
        executions.findLatestExecutionByFilters.mockRejectedValue(
            new Error('db down'),
        );
        await expect(service.isCovered(STATE)).resolves.toBe(false);
    });
});

describe('readCatchUpState', () => {
    it('reads the state a deferred catch-up kept', () => {
        expect(
            readCatchUpState({ metadata: { prReviewCatchUp: STATE } } as any),
        ).toEqual(STATE);
    });

    it('is null for any other job', () => {
        expect(readCatchUpState({ metadata: {} } as any)).toBeNull();
        expect(readCatchUpState({} as any)).toBeNull();
    });
});

describe('reviewCatchUpEnabled', () => {
    it.each([
        [undefined, true],
        ['on', true],
        ['off', false],
        [' OFF ', false],
        ['false', false],
        ['0', false],
    ])('CODE_REVIEW_CATCH_UP=%p → %p', (value, expected) => {
        expect(reviewCatchUpEnabled({ CODE_REVIEW_CATCH_UP: value })).toBe(
            expected,
        );
    });
});
