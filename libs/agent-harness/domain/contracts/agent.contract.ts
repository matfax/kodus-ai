/**
 * agent-harness — AgentSpec, AgentRunner, SubAgent.
 *
 * Key principle (Anthropic practice): role specialization does NOT require
 * a forked loop. finder and verifier are the SAME runner with different
 * AgentSpec (prompt + tool surface + policies). An "agent" is DATA, not a
 * class hierarchy.
 */

import type { AgentPolicy } from './policy.contract';
import type { Artifact, RunState, RunStep } from './run-state.contract';
import type { AgentTool, ToolContext, ToolRegistry } from './tool.contract';

/** Declarative configuration of an agent role. Pure data — swap prompt or
 *  tools to get a different role on the same runtime. */
export interface AgentSpec {
    readonly id: string;
    readonly systemPrompt: string;
    /** Cost-attribution label — the observability span's `agentName`. */
    readonly agentName?: string;
    /** Names the Langfuse observation + cost row (defaults to agentName / id). */
    readonly runName?: string;
    /** Phase label on the cost row (e.g. 'conversation' vs 'conversation-retry'). */
    readonly phase?: string;
    /** Observability span name (defaults to runName). */
    readonly spanName?: string;
    /** Tools this role may use. */
    readonly tools: ToolRegistry;
    /** The same tools as built for one model, when a tool's shape depends on it
     *  (a done-tool that is strict only on strict-capable models). The model
     *  call can fail over to another model before the first step, and each
     *  attempt gets the tools for its own model id (undefined for the managed
     *  default). Must list the same tool names as `tools`. Unset → `tools`
     *  for every attempt. */
    readonly toolsForModel?: (modelId: string | undefined) => ToolRegistry;
    /** Composable policies (budget, progress, compression, verify...). */
    readonly policies: readonly AgentPolicy[];
    /** Hard ceiling — the runner's fail-open even if no policy stops. */
    readonly maxSteps: number;
    /** Sampling temperature for the model call. Omitted -> provider default.
     *  Generic model-call config (not provider-specific), so it lives on the
     *  spec rather than in `providerOptions`. */
    readonly temperature?: number;
    /** Hard cap on output tokens PER model call. Omitted -> provider default. */
    readonly maxOutputTokens?: number;
    /** Opaque providerOptions (reasoning / thinking) that OVERRIDE the slot-
     *  derived ones. Unset → LLM.run derives them from the slot. The review
     *  finder sets this (config-derived reasoning that can differ from the slot);
     *  the chat/business agents leave it unset. The runner forwards it verbatim. */
    readonly providerOptions?: Readonly<Record<string, unknown>>;
    /** Name of the "final tool" whose call IS the run's structured output.
     *  When set, the runner materializes each call to it into
     *  `RunState.artifacts` (the "result tool" convention) — so the domain
     *  reads `state.artifacts` instead of re-scanning `steps` by hand. This is
     *  the CAPTURE concern; stopping ON that tool is a separate concern owned
     *  by a policy (e.g. CompletionGatePolicy.doneToolName) — same tool,
     *  distinct roles. */
    readonly resultToolName?: string;
    /** Other tools whose calls are captured into `RunState.artifacts` the same
     *  way, in step order, for a run that reports results as it goes (e.g. a
     *  non-terminal `reportFinding` next to the final `submitResult`). They do
     *  not stop the run. */
    readonly reportToolNames?: readonly string[];
    /**
     * What happens when the model call fails after the run has taken steps
     * and the org has a fallback model to move to:
     * - `'continue'`: the fallback picks up the run. It gets the opening
     *   messages plus a hand-over turn (`continuation`), the steps left
     *   (never more than `maxSteps` in all; none left ends the run), and
     *   the run keeps its steps and artifacts.
     * - `'restart'`: the fallback runs the task again from the opening
     *   messages; the failed attempt's steps are dropped.
     * Unset: the run ends with an error, as it always has. Only set it for an
     * agent whose tools have no external side effects, since the fallback
     * redoes some or all of the work.
     */
    readonly onFailover?: 'continue' | 'restart';
    /** The hand-over turn for a `'continue'` run. Unset → a generic note that
     *  lists the tool calls made so far. */
    readonly continuation?: (progress: ContinuationProgress) => string;
}

/** What a `'continue'` run's hand-over turn is built from. */
export interface ContinuationProgress {
    /** Every step taken so far, on every model. */
    readonly steps: readonly RunStep[];
    /** The artifacts captured from those steps. */
    readonly artifacts: readonly Artifact[];
    /** The steps the fallback may take. */
    readonly stepsLeft: number;
    readonly failedModel: string;
    readonly nextModel: string;
}

/** The single agent loop. The ENTIRE harness has exactly one of these.
 *  finder, verifier, replicas, sub-agents — all go through here. */
export interface AgentRunner {
    run(
        spec: AgentSpec,
        input: AgentRunInput,
        ctx: ToolContext,
    ): Promise<RunState>;
}

export interface AgentRunInput {
    /** Opening user message(s) that frame the task. */
    readonly prompt: string;
    /** Optional seed messages (prior context). */
    readonly seedMessages?: readonly { role: 'user' | 'assistant'; content: string }[];
    /** Langfuse observation metadata (org / team / repo / provider ...). Passed
     *  to LLM.run, which builds the vendor telemetry shape — so the caller hands
     *  over the RAW metadata, not a pre-built SDK payload. Opaque here to keep the
     *  domain contract free of the vendor type. */
    readonly telemetryMetadata?: Readonly<Record<string, unknown>>;
    /** Per-run telemetry in the SDK's shape (e.g. `toAiSdkTelemetryArgs`).
     *  LLM.run builds the model call's telemetry itself, named after the run's
     *  `runName`; this field only signals that the caller wants the run traced.
     *  The metadata it opts into travels in `runtimeContext`, which the runner
     *  forwards as `telemetryMetadata` when that is not set.
     *
     *  Type the payload with a `type` alias, never an `interface`: interfaces
     *  get no implicit index signature and so do not satisfy `Record<string,
     *  unknown>`, however identical their members look. */
    readonly telemetry?: Readonly<Record<string, unknown>>;
    /** Opaque per-run runtime context, forwarded VERBATIM as the model call's
     *  `runtimeContext`. Carries the values whose keys the telemetry payload
     *  opts into — the tracer records nothing that was not opted in, and both
     *  halves are the domain's call. */
    readonly runtimeContext?: Readonly<Record<string, unknown>>;
}

/** Adapter that exposes an AgentSpec AS A TOOL (sub-agent-as-tool pattern).
 *  This is how orchestration composes agents: a parent calls a sub-agent
 *  the same way it calls grep. Gives context isolation (own window) and
 *  returns a distilled summary, not the full transcript. */
export interface SubAgentFactory {
    asTool(params: {
        name: string;
        description: string;
        spec: AgentSpec;
        /** Input contract the parent calls with. Defaults to {task:string}. */
        inputSchema?: AgentTool['inputSchema'];
        /** Maps the parent's tool input -> the sub-agent's prompt. */
        toPrompt: (input: unknown) => string;
        /** Distills the sub-agent RunState -> the string returned to parent. */
        summarize: (state: RunState) => string;
    }): AgentTool;
}
