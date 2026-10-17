/**
 * Strict tool use is decided per failover attempt, end to end.
 *
 * A Gemini primary takes the strict done-tool; an OpenAI chain entry rejects a
 * strict schema up front. The finder and verifier specs rebuild their tools for
 * each attempt's model (AgentSpec.toolsForModel → AgentLoopSeams.toolsForModel),
 * so a primary that fails before its first step hands a non-strict tool to the
 * fallback while keeping strict itself. Runs the real runner, LLM.run and the
 * failover loop; only the model build is mocked, so the assertions read the
 * tools the provider would actually have been sent.
 */
jest.mock('@libs/llm/model-invocation', () => ({
    resolveModelConfig: jest.fn(),
}));

import { MockLanguageModelV3 } from 'ai/test';

import { resolveModelConfig } from '@libs/llm/model-invocation';
import type { NormalizedModel } from '@libs/llm/byok-config';
import {
    httpProviderError,
    jsonErrorBody,
} from '@libs/llm/testing/provider-errors';
import { scriptedToolModel } from '@libs/agent-harness/infrastructure/ai-sdk/__test-utils__/scripted-tool-model';
import type { ProgressLedger } from '@libs/agent-harness/domain/contracts/progress.contract';
import type { ToolContext } from '@libs/agent-harness/domain/contracts/tool.contract';
import { AiSdkAgentRunner } from '@libs/agent-harness/infrastructure/ai-sdk/ai-sdk-agent-runner';
import { InMemoryToolRegistry } from '@libs/agent-harness/infrastructure/tools/in-memory-tool-registry';

import {
    buildFinderAgentSpec,
    extractFindings,
    FINDER_DONE_TOOL,
} from '@libs/code-review/infrastructure/agents/core/finder.agent';
import {
    LlmVerifier,
    VERIFY_DONE_TOOL,
} from '@libs/code-review/infrastructure/agents/core/verifier.agent';

const PRIMARY = 'gemini-2.5-pro';
const FALLBACK = 'gpt-5';

const geminiWithOpenAiFallback: NormalizedModel = {
    model: PRIMARY,
    byokModelId: 'g',
    provider: 'google',
    apiKey: 'enc',
    fallbacks: [
        {
            model: FALLBACK,
            byokModelId: 'o',
            provider: 'openai',
            apiKey: 'enc',
            chainIndex: 0,
        },
    ],
} as any;

/** The `strict` flag of `toolName` in each model call, keyed by model id. */
type SeenStrict = Array<{ model: string; strict: unknown }>;

/**
 * Builds the model for each attempt: the primary records the tools it was sent
 * and fails before any step with a bad key (cascade-worthy); the fallback
 * records its tools and calls the done-tool with `doneInput`.
 */
function mockModels(toolName: string, doneInput: unknown, seen: SeenStrict) {
    const record = (model: string, opts: any) =>
        seen.push({
            model,
            strict: opts.tools?.find((t: any) => t.name === toolName)?.strict,
        });
    (resolveModelConfig as jest.Mock).mockImplementation(
        (slot?: NormalizedModel) => {
            const id = slot?.model ?? 'managed';
            const model =
                id === PRIMARY
                    ? new MockLanguageModelV3({
                          doGenerate: (async (opts: any) => {
                              record(id, opts);
                              throw await httpProviderError(
                                  401,
                                  'Unauthorized',
                                  jsonErrorBody('invalid api key'),
                              );
                          }) as any,
                      })
                    : wrapRecording(
                          scriptedToolModel(() => ({
                              id: 'c1',
                              name: toolName,
                              input: doneInput,
                          })),
                          (opts) => record(id, opts),
                      );
            return {
                model,
                callOptions: {},
                providerOptions: {},
                modelName: id,
                usageIdentity: {},
            };
        },
    );
}

function wrapRecording(
    model: MockLanguageModelV3,
    onCall: (opts: any) => void,
): MockLanguageModelV3 {
    const inner = model.doGenerate.bind(model);
    return new MockLanguageModelV3({
        doGenerate: (async (opts: any) => {
            onCall(opts);
            return inner(opts);
        }) as any,
    });
}

const noCriticalLedger: ProgressLedger = {
    markFromToolCall: () => undefined,
    summary: () => ({
        totalTargets: 0,
        pendingTargets: 0,
        criticalTotal: 0,
        criticalPending: 0,
    }),
    debtNote: () => null,
};

/** One primary call (strict), then every step of the fallback run non-strict.
 *  The verifier does not stop on its done-tool, so the fallback may take more
 *  than one step; each must carry the fallback's tools. */
function expectStrictOnlyOnPrimary(seen: SeenStrict) {
    expect(seen[0]).toEqual({ model: PRIMARY, strict: true });
    expect(seen.length).toBeGreaterThan(1);
    for (const call of seen.slice(1)) {
        expect(call).toEqual({ model: FALLBACK, strict: false });
    }
}

const ctx: ToolContext = { runId: 'strict-per-attempt' };

beforeEach(() => {
    (resolveModelConfig as jest.Mock).mockReset();
});

describe('strict tools are decided per failover attempt', () => {
    it('finder: the Gemini primary is sent strict, the OpenAI fallback is not', async () => {
        const seen: SeenStrict = [];
        mockModels(
            FINDER_DONE_TOOL,
            { reasoning: 'nothing found', suggestions: [] },
            seen,
        );
        const spec = buildFinderAgentSpec({
            systemPrompt: 'find bugs',
            modelId: PRIMARY,
            tools: new InMemoryToolRegistry([]),
            coverageLedger: noCriticalLedger,
            maxSteps: 3,
        });

        const state = await new AiSdkAgentRunner(geminiWithOpenAiFallback).run(
            spec,
            { prompt: 'review this PR' },
            ctx,
        );

        expect(state.status).not.toBe('error');
        expect(extractFindings(state).reasoning).toBe('nothing found');
        expectStrictOnlyOnPrimary(seen);
    });

    it('verifier: the Gemini primary is sent strict, the OpenAI fallback is not', async () => {
        const seen: SeenStrict = [];
        mockModels(
            VERIFY_DONE_TOOL,
            { keep: false, rationale: 'refuted' },
            seen,
        );
        const verifier = new LlmVerifier(
            new AiSdkAgentRunner(geminiWithOpenAiFallback),
            { modelId: PRIMARY, tools: new InMemoryToolRegistry([]) },
        );

        const verdict = await verifier.verify(
            {
                relevantFile: 'src/x.ts',
                suggestionContent: 'null deref',
                existingCode: 'x.y',
                improvedCode: 'x?.y',
                relevantLinesStart: 1,
                relevantLinesEnd: 1,
                severity: 'high',
                confidence: 8,
            } as any,
            ctx,
        );

        expect(verdict.keep).toBe(false);
        expectStrictOnlyOnPrimary(seen);
    });
});
