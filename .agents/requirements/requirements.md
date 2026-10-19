# Requirements

<!-- reqtracker:v2 next=19 -->

> Tracked automatically from what the user stated or explicitly approved. Suggestions Claude made on its own are not recorded here.

- [ ] `R1` Claude is available as a model provider through the Claude Agent SDK, using the user's Claude subscription login via an environment token. `+llm-provider` `+claude-sdk` `2026-10-09`
- [ ] `R2` Requirements are a first-class part of the product: all requirements in the repo are listed, including their graph edges. `+requirements-tracking` `2026-10-09`
- [ ] `R3` Requirements changes are highlighted for each PR. `+requirements-tracking` `+pr-review` `2026-10-09`
- [ ] `R4` A PR that violates a requirement fails, and the user must remove the violated requirement, rewrite it to meet the new spec, or mark it unmet. `+requirements-tracking` `+pr-review` `2026-10-09`
- [ ] `R5` The user resolves a violated requirement through emoji feedback on a suggested rewrite that keeps the requirement graph consistent, and can disagree by replying to Kody. `+requirements-tracking` `+pr-review` `2026-10-09`
- [ ] `R6` An integrated agent sweeps commits against requirements, more thoroughly than the Claude plugin commit sweep, with a better UI, more options, and better simplification and explanation. `+requirements-tracking` `+agent` `2026-10-09`
- [ ] `R7` Issues can be discussed directly in the UI, and a model can review them there. `+issues` `2026-10-09`
- [ ] `R8` From the UI, the AI can close an issue or open a GitHub issue for it. `+issues` `2026-10-09`
- [ ] `R9` Business logic is integrated into reviews and tied to the Claude requirements tracking file. `+requirements-tracking` `+pr-review` `2026-10-09`
- [x] `R10` API model calls deal gracefully with timeouts. `+llm-provider` `2026-10-09 → met 2026-10-18: confirmed by the user before a commit`
- [ ] `R11` Requests per minute are throttled per provider, not just parallel requests per model. `+llm-provider` `+throttling` `2026-10-09`
- [ ] `R12` API model calls fall back to Claude through the Agent SDK, with the interface choosing the model and effort level. `+llm-provider` `+claude-sdk` `2026-10-09`
- [ ] `R13` Kodus can commit requirement changes to PR branches, and its commits are signed. `+requirements-tracking` `+pr-review` `2026-10-09`
- [ ] `R14` All requirements with subjects related to the PR are checked on each PR. `+requirements-tracking` `+pr-review` `2026-10-09`
- [ ] `R15` The issues discussed in the UI are the Kodus issues that remain open after a PR, listed at repo level. `+issues` `2026-10-09`
- [x] `R16` Model fallback is graceful and reliable, so Claude can serve as another provider. `+llm-provider` `+claude-sdk` `2026-10-16 → met 2026-10-18: confirmed by the user before a commit`
- [x] `R17` When an agent loop fails partway, findings are saved as they are reported, so a run that moves to the backup model keeps the findings already made and finishes the rest instead of restarting. `+agent` `+llm-provider` `+pr-review` `2026-10-18 → met 2026-10-18: confirmed by the user before a commit`
- [x] `R18` A push that arrives while a review is running waits and then runs as an incremental review, so reviews are eventually consistent with the PR. `+pr-review` `2026-10-19 → met 2026-10-19: automationCodeReview refuseRun throws a catch-up PrReviewInProgressError for pushes, code-review-job-processor deferCatchUp waits via PrReviewDeferralService nextCatchUp until the holder is gone, then`
