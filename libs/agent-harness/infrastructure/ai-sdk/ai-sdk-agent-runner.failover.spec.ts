/**
 * AiSdkAgentRunner — what the runner hands LLM.run for a failover after steps
 * ran (`AgentSpec.onFailover`), and how the run reads afterwards.
 *
 * LLM.run is mocked. Each test plays the failover the way LLM.run does: leg 1's
 * steps, then `loop.continueFrom(info)` (what LLM.run calls when the attempt
 * fails mid-loop), then leg 2's steps numbered from 0 again, as the SDK does
 * for every new attempt.
 */
jest.mock('@libs/llm/llm', () => ({ LLM: { run: jest.fn() } }));

import { LLM } from '@libs/llm/llm';
import type { ResumeInfo } from '@libs/llm/model-failover';

import type {
    AgentSpec,
    ContinuationProgress,
} from '../../domain/contracts/agent.contract';
import type { AgentPolicy, StepView } from '../../domain/contracts/policy.contract';
import type { AgentTool, ToolContext } from '../../domain/contracts/tool.contract';
import type { RunState } from '../../domain/contracts/run-state.contract';
import { InMemoryToolRegistry } from '../tools/in-memory-tool-registry';
import { AiSdkAgentRunner } from './ai-sdk-agent-runner';

const mockRun = LLM.run as jest.Mock;

const tool = (name: string): AgentTool => ({
    name,
    description: name,
    inputSchema: { type: 'object', properties: {} },
    execute: async () => ({ output: 'ok' }),
});

const ctx: ToolContext = { runId: 'run-1' };

const INFO: ResumeInfo = {
    failedModel: 'primary',
    nextModel: 'backup',
    chainIndex: 1,
    reason: 'deadline',
};

function spec(over: Partial<AgentSpec> = {}): AgentSpec {
    return {
        id: 'finder',
        systemPrompt: 'find bugs',
        tools: new InMemoryToolRegistry([
            tool('readFile'),
            tool('reportFinding'),
            tool('submitResult'),
        ]),
        policies: [],
        maxSteps: 10,
        resultToolName: 'submitResult',
        reportToolNames: ['reportFinding'],
        ...over,
    };
}

const step = (name: string, input: unknown = {}) => ({
    text: '',
    toolCalls: [{ toolCallId: `${name}-id`, toolName: name, input }],
    toolResults: [{ toolCallId: `${name}-id` }],
    finishReason: { unified: 'tool-calls', raw: 'tool-calls' },
    usage: { inputTokens: 10, outputTokens: 1 },
});

interface Played {
    state: RunState;
    req: any;
    leg?: { messages: any[]; maxSteps: number };
}

/**
 * Play leg 1, hand over, play leg 2. `leg2` undefined → the backup throws
 * `backupErr` before taking a step.
 */
async function play(opts: {
    spec: AgentSpec;
    leg1: any[];
    leg2?: any[];
    backupErr?: Error;
    onLeg2Start?: (req: any) => Promise<void>;
}): Promise<Played> {
    let req: any;
    let leg: Played['leg'];
    mockRun.mockImplementation(async (r: any) => {
        req = r;
        for (const ev of opts.leg1) await r.loop.onStepFinish(ev);
        leg = r.loop.continueFrom?.(INFO);
        if (!leg) throw new Error('primary failed');
        await opts.onLeg2Start?.(r);
        if (!opts.leg2) throw opts.backupErr ?? new Error('backup failed');
        for (const ev of opts.leg2) await r.loop.onStepFinish(ev);
        return { usage: { inputTokens: 100, outputTokens: 5 }, steps: [] };
    });
    const state = await new AiSdkAgentRunner(undefined).run(
        opts.spec,
        { prompt: 'review this PR' },
        ctx,
    );
    return { state, req, leg };
}

beforeEach(() => mockRun.mockReset());

describe('AiSdkAgentRunner — failover after steps ran', () => {
    it('offers no continuation when the spec does not opt in', async () => {
        mockRun.mockResolvedValue({ usage: {}, steps: [] });
        await new AiSdkAgentRunner(undefined).run(
            spec({ reportToolNames: undefined }),
            { prompt: 'go' },
            ctx,
        );
        expect(mockRun.mock.calls[0][0].loop.continueFrom).toBeUndefined();
    });

    describe("'continue'", () => {
        it('hands the backup the seed plus the hand-over turn, and the steps left', async () => {
            const continuation = jest.fn(
                (p: ContinuationProgress) =>
                    `take over: ${p.artifacts.length} artifacts, ${p.stepsLeft} steps, from ${p.failedModel} to ${p.nextModel}`,
            );
            const { leg } = await play({
                spec: spec({ onFailover: 'continue', continuation }),
                leg1: [step('readFile'), step('reportFinding', { title: 'f1' })],
                leg2: [step('submitResult', { suggestions: [] })],
            });

            expect(leg).toEqual({
                messages: [
                    { role: 'user', content: 'review this PR' },
                    {
                        role: 'user',
                        content:
                            'take over: 1 artifacts, 8 steps, from primary to backup',
                    },
                ],
                maxSteps: 8,
            });
            expect(continuation).toHaveBeenCalledTimes(1);
        });

        it('keeps every step and artifact of both legs, and sums their usage', async () => {
            const { state } = await play({
                spec: spec({ onFailover: 'continue', continuation: () => 'go on' }),
                leg1: [step('readFile'), step('reportFinding', { title: 'f1' })],
                leg2: [
                    step('reportFinding', { title: 'f2' }),
                    step('submitResult', { suggestions: [] }),
                ],
            });

            expect(state.status).toBe('completed');
            expect(state.steps.map((s) => s.index)).toEqual([0, 1, 2, 3]);
            expect(state.artifacts.map((a) => [a.type, a.location])).toEqual([
                ['reportFinding', 'step:1'],
                ['reportFinding', 'step:2'],
                ['submitResult', 'step:3'],
            ]);
            // Leg 1's two steps (10/1 each) + leg 2's SDK total (100/5).
            expect(state.usage.inputTokens).toBe(120);
            expect(state.usage.outputTokens).toBe(7);
            expect(state.trace).toContainEqual(
                expect.objectContaining({
                    kind: 'failover.continue',
                    detail: expect.objectContaining({
                        mode: 'continue',
                        from: 'primary',
                        to: 'backup',
                        step: 2,
                    }),
                }),
            );
        });

        it('numbers the backup steps after the primary ones for the policies', async () => {
            const seen: number[] = [];
            const policy: AgentPolicy = {
                name: 'probe',
                prepareStep: (view: StepView) => {
                    seen.push(view.stepNumber);
                    return {};
                },
                shouldStop: (view: StepView) => {
                    seen.push(view.stepNumber);
                    return false;
                },
            };
            await play({
                spec: spec({
                    onFailover: 'continue',
                    continuation: () => 'go on',
                    policies: [policy],
                }),
                leg1: [step('readFile'), step('readFile'), step('readFile')],
                leg2: [step('submitResult')],
                onLeg2Start: async (req) => {
                    // The SDK's first step of the new attempt is step 0.
                    await req.loop.prepareStep({ stepNumber: 0, messages: [] });
                    await req.loop.stopWhen[0]({ steps: [{}] });
                },
            });
            expect(seen).toEqual([3, 4]);
        });

        it('never gives the backup more than the steps left', async () => {
            const { leg } = await play({
                spec: spec({
                    onFailover: 'continue',
                    continuation: () => 'go on',
                    maxSteps: 4,
                }),
                leg1: [step('readFile'), step('readFile'), step('readFile')],
                leg2: [step('submitResult')],
            });
            expect(leg?.maxSteps).toBe(1);
        });

        it('ends the run when the failed attempt used the whole budget, keeping its reports', async () => {
            const { state, leg } = await play({
                spec: spec({
                    onFailover: 'continue',
                    continuation: () => 'go on',
                    maxSteps: 2,
                }),
                leg1: [step('readFile'), step('reportFinding', { title: 'f1' })],
                leg2: [step('submitResult')],
            });
            expect(leg).toBeUndefined();
            expect(state.status).toBe('error');
            expect(state.artifacts.map((a) => a.type)).toEqual(['reportFinding']);
        });

        it('falls back to a note listing the tool calls when the spec has no continuation', async () => {
            const { leg } = await play({
                spec: spec({ onFailover: 'continue' }),
                leg1: [step('readFile', { path: 'src/a.ts' })],
                leg2: [step('submitResult')],
            });
            const note = leg?.messages.at(-1).content as string;
            expect(note).toContain('stopped after 1 steps');
            expect(note).toContain('You have 9 steps');
            expect(note).toContain('- readFile {"path":"src/a.ts"}');
        });

        it('keeps what both legs reported when the backup fails too', async () => {
            const { state } = await play({
                spec: spec({ onFailover: 'continue', continuation: () => 'go on' }),
                leg1: [step('reportFinding', { title: 'f1' })],
                backupErr: new Error('backup down'),
            });
            expect(state.status).toBe('error');
            expect(state.artifacts.map((a) => a.type)).toEqual(['reportFinding']);
        });
    });

    describe("'restart'", () => {
        it('reruns the seed with the full budget and drops the failed steps', async () => {
            const { state, leg } = await play({
                spec: spec({ onFailover: 'restart' }),
                leg1: [step('readFile'), step('reportFinding', { title: 'f1' })],
                leg2: [step('submitResult', { suggestions: [] })],
            });
            expect(leg).toEqual({
                messages: [{ role: 'user', content: 'review this PR' }],
                maxSteps: 10,
            });
            expect(state.steps.map((s) => s.index)).toEqual([0]);
            expect(state.artifacts.map((a) => a.type)).toEqual(['submitResult']);
            expect(state.usage.inputTokens).toBe(100);
        });
    });
});
