import { createLogger } from '@libs/core/log/logger';
import { Inject, Injectable } from '@nestjs/common';

import {
    AUTOMATION_EXECUTION_SERVICE_TOKEN,
    IAutomationExecutionService,
} from '@libs/automation/domain/automationExecution/contracts/automation-execution.service';
import { AutomationStatus } from '@libs/automation/domain/automation/enum/automation-status';
import type { CommandReviewFeedbackTarget } from '@libs/code-review/infrastructure/adapters/services/codeReviewHandlerService.service';
import type { IWorkflowJob } from '@libs/core/workflow/domain/interfaces/workflow-job.interface';
import { CodeManagementService } from '@libs/platform/infrastructure/adapters/services/codeManagement.service';

/** Where a deferred catch-up keeps what it needs on wake-up (job metadata). */
export const REVIEW_CATCH_UP_METADATA_KEY = 'prReviewCatchUp';

/** A push refused while a review of the same PR ran, waiting to catch up. */
export interface ReviewCatchUpState {
    target: CommandReviewFeedbackTarget;
    teamAutomationId: string;
    /** The head commit the push carried, when the platform payload had one. */
    pushHeadSha?: string;
}

/** The catch-up state a deferred job carries, if it is one. */
export function readCatchUpState(job: IWorkflowJob): ReviewCatchUpState | null {
    const state = (job.metadata as Record<string, unknown> | null)?.[
        REVIEW_CATCH_UP_METADATA_KEY
    ] as ReviewCatchUpState | undefined;
    return state?.target && state.teamAutomationId ? state : null;
}

/**
 * Decides, when a catch-up wakes up, whether it still has anything to
 * review. Several pushes refused during one review each wait to catch up;
 * the first to run reviews the PR up to its live head, and the others find
 * their push already covered and finish without starting a review.
 */
@Injectable()
export class ReviewCatchUpService {
    private readonly logger = createLogger(ReviewCatchUpService.name);

    constructor(
        @Inject(AUTOMATION_EXECUTION_SERVICE_TOKEN)
        private readonly automationExecutionService: IAutomationExecutionService,
        private readonly codeManagement: CodeManagementService,
    ) {}

    /**
     * True when the latest successful review of the PR analyzed the push's
     * head or a later commit. False whenever that cannot be shown (no head
     * on the push, no earlier review, a lookup failure): running a review
     * that turns out to have no new commits costs a skip, missing one costs
     * the push.
     */
    async isCovered(state: ReviewCatchUpState): Promise<boolean> {
        const { target, teamAutomationId, pushHeadSha } = state;
        if (!pushHeadSha) return false;

        try {
            const last =
                await this.automationExecutionService.findLatestExecutionByFilters(
                    {
                        status: AutomationStatus.SUCCESS,
                        teamAutomation: { uuid: teamAutomationId },
                        pullRequestNumber: target.pullRequest.number,
                        repositoryId: target.repository.id,
                    },
                );
            const analyzed = last?.dataExecution?.lastAnalyzedCommit as
                | string
                | { sha?: string }
                | undefined;
            const analyzedSha =
                typeof analyzed === 'string' ? analyzed : analyzed?.sha;
            if (!analyzedSha) return false;
            if (analyzedSha === pushHeadSha) return true;

            const commits =
                (await this.codeManagement.getCommitsForPullRequestForCodeReview(
                    {
                        organizationAndTeamData: target.organizationAndTeamData,
                        repository: target.repository,
                        prNumber: target.pullRequest.number,
                    },
                )) ?? [];
            const shas = commits.map((commit: { sha?: string }) => commit?.sha);
            const pushIndex = shas.indexOf(pushHeadSha);
            const analyzedIndex = shas.indexOf(analyzedSha);
            return (
                pushIndex !== -1 &&
                analyzedIndex !== -1 &&
                pushIndex <= analyzedIndex
            );
        } catch (error) {
            this.logger.warn({
                message: `Could not tell whether PR#${target.pullRequest.number} was already reviewed; running the catch-up`,
                context: ReviewCatchUpService.name,
                error: error instanceof Error ? error : undefined,
                metadata: {
                    organizationId:
                        target.organizationAndTeamData?.organizationId,
                    repositoryId: target.repository.id,
                    pullRequestNumber: target.pullRequest.number,
                },
            });
            return false;
        }
    }
}
