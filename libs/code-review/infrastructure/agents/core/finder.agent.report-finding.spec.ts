/**
 * The finder reports findings as it confirms them (`reportFinding`), so a run
 * a failing model cuts short keeps them, and a fallback that takes the run
 * over is told what was found and what is still unread.
 */
import type { ContinuationProgress } from '@libs/agent-harness/domain/contracts/agent.contract';
import type { ProgressLedger } from '@libs/agent-harness/domain/contracts/progress.contract';
import { InMemoryToolRegistry } from '@libs/agent-harness/infrastructure/tools/in-memory-tool-registry';
import {
    buildFinderAgentSpec,
    extractFindings,
    FINDER_DONE_TOOL,
    FINDER_REPORT_TOOL,
    finderContinuation,
    reportFindingTool,
} from '@libs/code-review/infrastructure/agents/core/finder.agent';
import { buildVerifierAgentSpec } from '@libs/code-review/infrastructure/agents/core/verifier.agent';

const ledger = (debt: string | null = null): ProgressLedger => ({
    markFromToolCall: () => undefined,
    summary: () =>
        ({
            totalTargets: 0,
            pendingTargets: 0,
            criticalTotal: 0,
            criticalPending: 0,
        }),
    debtNote: () => debt,
});

const spec = (over: { acceptsRevisions?: boolean; modelId?: string } = {}) =>
    buildFinderAgentSpec({
        systemPrompt: 'find bugs',
        modelId: over.modelId ?? 'gpt-x',
        tools: new InMemoryToolRegistry([]),
        coverageLedger: ledger(),
        acceptsRevisions: over.acceptsRevisions,
    });

const finding = (file: string, content: string, start?: number, end?: number) => ({
    relevantFile: file,
    suggestionContent: content,
    existingCode: 'a',
    improvedCode: 'b',
    oneSentenceSummary: `${content} summary`,
    ...(start != null ? { relevantLinesStart: start } : {}),
    ...(end != null ? { relevantLinesEnd: end } : {}),
});

const state = (artifacts: Array<{ type: string; payload: unknown }>): any => ({
    runId: 'r',
    agentId: 'finder',
    status: 'error',
    steps: [],
    artifacts,
    usage: {},
    trace: [],
});

describe('finder spec: report as you go, continue on a fallback', () => {
    it('offers reportFinding next to submitResult, with the same strictness', () => {
        for (const modelId of ['gpt-x', 'gemini-3-pro']) {
            const tools = spec({ modelId }).tools.list();
            const report = tools.find((t) => t.name === FINDER_REPORT_TOOL);
            const done = tools.find((t) => t.name === FINDER_DONE_TOOL);
            expect(report).toBeDefined();
            expect(report?.strict).toBe(done?.strict);
        }
    });

    it('captures reportFinding calls and continues on a fallback', () => {
        const s = spec();
        expect(s.reportToolNames).toEqual([FINDER_REPORT_TOOL]);
        expect(s.onFailover).toBe('continue');
        expect(s.continuation).toEqual(expect.any(Function));
    });

    it('takes one finding, the same item schema submitResult takes', () => {
        const done = spec().tools.list().find((t) => t.name === FINDER_DONE_TOOL);
        expect(reportFindingTool.inputSchema).toEqual(
            (done?.inputSchema as any).properties.suggestions.items,
        );
    });

    it('accepts revisesSuggestionId only when the PR has history, without touching the shared tool', () => {
        const report = (acceptsRevisions?: boolean) =>
            spec({ acceptsRevisions })
                .tools.list()
                .find((t) => t.name === FINDER_REPORT_TOOL)?.inputSchema as any;
        expect(report().properties).not.toHaveProperty('revisesSuggestionId');
        expect(report(true).properties.revisesSuggestionId).toEqual({
            type: 'string',
        });
        expect((reportFindingTool.inputSchema as any).properties).not.toHaveProperty(
            'revisesSuggestionId',
        );
    });

    it('lets a fallback rerun a verification from the start', () => {
        expect(
            buildVerifierAgentSpec({
                modelId: 'gpt-x',
                tools: new InMemoryToolRegistry([]),
            }).onFailover,
        ).toBe('restart');
    });
});

describe('extractFindings: reported findings plus the final submitResult', () => {
    it('keeps reported findings when the run never reached submitResult', () => {
        const st = state([
            { type: FINDER_REPORT_TOOL, payload: finding('a.ts', 'one') },
            { type: FINDER_REPORT_TOOL, payload: finding('b.ts', 'two') },
        ]);
        expect(extractFindings(st).suggestions.map((s) => s.relevantFile)).toEqual([
            'a.ts',
            'b.ts',
        ]);
        expect(st.__findingsOutcome).toBe('reported-only');
    });

    it('merges reported and submitted findings, dropping a finding submitted again', () => {
        const out = extractFindings(
            state([
                { type: FINDER_REPORT_TOOL, payload: finding('a.ts', 'one', 3) },
                {
                    type: FINDER_DONE_TOOL,
                    payload: {
                        reasoning: 'done',
                        suggestions: [
                            finding('a.ts', 'one', 3),
                            finding('c.ts', 'three'),
                        ],
                    },
                },
            ]),
        );
        expect(out.suggestions.map((s) => s.suggestionContent)).toEqual([
            'one',
            'three',
        ]);
        expect(out.reasoning).toBe('done');
    });

    it('drops a reported finding that fails validation and keeps the rest', () => {
        const out = extractFindings(
            state([
                { type: FINDER_REPORT_TOOL, payload: { suggestionContent: 'no file' } },
                { type: FINDER_REPORT_TOOL, payload: finding('a.ts', 'one') },
            ]),
        );
        expect(out.suggestions.map((s) => s.relevantFile)).toEqual(['a.ts']);
    });

    it('reads a run without reported findings exactly as before', () => {
        const submitted = {
            reasoning: 'r',
            suggestions: [finding('a.ts', 'one')],
        };
        const out = extractFindings(
            state([{ type: FINDER_DONE_TOOL, payload: submitted }]),
        );
        expect(out).toEqual({ reasoning: 'r', suggestions: submitted.suggestions });
    });
});

describe('finderContinuation', () => {
    const progress = (
        artifacts: ContinuationProgress['artifacts'],
    ): ContinuationProgress => ({
        steps: Array.from({ length: 6 }, (_, index) => ({
            index,
            message: { role: 'assistant', content: '' },
        })),
        artifacts,
        stepsLeft: 9,
        failedModel: 'primary',
        nextModel: 'backup',
    });

    it('lists what was reported and what is still unread', () => {
        const note = finderContinuation(
            progress([
                {
                    type: FINDER_REPORT_TOOL,
                    payload: finding('src/a.ts', 'null deref', 10, 12),
                },
                { type: FINDER_REPORT_TOOL, payload: finding('src/b.ts', 'leak', 4, 4) },
                { type: FINDER_DONE_TOOL, payload: { suggestions: [] } },
            ]),
            ledger('CRITICAL pending (1/2) — readFile src/c.ts:5-9'),
        );
        expect(note).toContain('stopped after 6 steps');
        expect(note).toContain('You have 9 steps');
        expect(note).toContain('- src/a.ts:10-12: null deref summary');
        expect(note).toContain('- src/b.ts:4: leak summary');
        expect(note).toContain('do not report them again');
        expect(note).toContain('CRITICAL pending (1/2) — readFile src/c.ts:5-9');
        expect(note).toContain(`with ${FINDER_REPORT_TOOL}`);
    });

    it('says so when nothing was reported and every hunk was read', () => {
        const note = finderContinuation(progress([]), ledger(null));
        expect(note).toContain('No findings were reported yet.');
        expect(note).toContain('Every changed hunk has already been read.');
    });
});
