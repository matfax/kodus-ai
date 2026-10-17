# Requirements

<!-- reqtracker:v2 next=4 -->

> Tracked automatically from what the user stated or explicitly approved. Suggestions Claude made on its own are not recorded here.

- [ ] `R1` A fallback chain holds up to three fallback models, tried in order after the main model fails. `+llm-provider` `+agent` `2026-10-17`
- [x] `R2` An open-source API endpoint saves the ordered fallback model list. `+llm-provider` `2026-10-17 → met 2026-10-17: PUT /organization-parameters/byok-fallback-chain reported done in task 5`
- [x] `R3` A settings page outside the Enterprise code lets the user add, remove and reorder fallback models. `+llm-provider` `2026-10-17 → met 2026-10-17: Page at /organization/model-fallbacks reported done in task 5`
