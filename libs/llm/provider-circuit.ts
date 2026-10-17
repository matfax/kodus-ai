/**
 * Per-model circuit breaker — the memory the failover cascade lacks between
 * calls.
 *
 * Without it, every call to a model whose provider is down pays the full price
 * of finding that out again: SDK retries, backoff, a round-trip deadline, and
 * only then the cascade. The breaker remembers that a served model kept
 * failing in a way only the provider can cause, and lets the cascade skip it
 * for a while, straight to the next chain entry.
 *
 * Pure state, no I/O: `runWithModelFailover` asks `admit()` before each
 * attempt and reports how the attempt ended; the BYOK wrapper reports the
 * first provider answer (that is where a half-open probe resolves). Nothing
 * here retries, sleeps or calls a provider.
 *
 * Rules that keep it from doing harm:
 *  - The last (or only) chain entry is never skipped — every call makes at
 *    least one real attempt, and a single-model org sees no change.
 *  - Only failures a different model can fix count: a provider outage, an
 *    exhausted quota, our full-length round-trip deadline. A 429 never counts
 *    (the limiter owns those), nor does a caller's cancel or budget.
 *  - Every state change bumps a generation, and a result reported against an
 *    older generation is ignored: a loop admitted while CLOSED that ends forty
 *    minutes later cannot flip the state it never saw.
 *
 * State is process-local (like the limiter cache): each replica learns an
 * outage on its own.
 */
import { createHash } from 'node:crypto';

import { createLogger } from '@libs/core/log/logger';
import type { NormalizedModel } from '@libs/llm/byok-config';
import {
    classifyLLMError,
    failureChain,
    isCallerAbortOrBudget,
    isProviderTimeout,
    LlmErrorCategory,
} from '@libs/llm/error-classifier';
import { isPlatformFundedProvider } from '@libs/llm/platform-funded-provider';
import { isProviderDeadlineError } from '@libs/llm/provider-deadline-error';
import { splitKodusModelId } from '@libs/llm/providers/kodus/model-id';

const logger = createLogger('ProviderCircuit');

export const LLM_CIRCUIT_TAG = '[LLM-CIRCUIT]';

// ─── Configuration ───────────────────────────────────────────────────────────

/** `off` keeps no state; `shadow` keeps state and logs what it would skip,
 *  but runs everything; `enforce` skips. */
export type CircuitMode = 'off' | 'shadow' | 'enforce';

export interface CircuitConfig {
    /** Consecutive counted failures (inside `windowMs`) that open a key. */
    failureThreshold: number;
    /** A counted failure older than this restarts the count. */
    windowMs: number;
    /** First open period; doubles on each failed probe, up to `maxOpenMs`. */
    openMs: number;
    maxOpenMs: number;
    /** Cap on how far a provider's `retry-after` can push the open period. */
    maxResetMs: number;
    /** A CLOSED key idle this long is dropped from memory. */
    idleEvictMs: number;
    /** Hard cap on tracked keys. */
    maxKeys: number;
}

export const DEFAULT_CIRCUIT_CONFIG: Readonly<CircuitConfig> = Object.freeze({
    failureThreshold: 3,
    windowMs: 120_000,
    openMs: 30_000,
    maxOpenMs: 10 * 60_000,
    maxResetMs: 60 * 60_000,
    idleEvictMs: 30 * 60_000,
    maxKeys: 10_000,
});

type Env = Readonly<Record<string, string | undefined>>;

/** `LLM_CIRCUIT_MODE`, default `enforce`. An unknown value falls back to the
 *  default rather than silently turning the breaker off. */
export function circuitModeFromEnv(env: Env = process.env): CircuitMode {
    const raw = env.LLM_CIRCUIT_MODE?.trim().toLowerCase();
    return raw === 'off' || raw === 'shadow' || raw === 'enforce'
        ? raw
        : 'enforce';
}

function positiveIntEnv(env: Env, name: string, fallback: number): number {
    const raw = env[name];
    if (raw === undefined || raw.trim() === '') return fallback;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

/** The `LLM_CIRCUIT_*` knobs; an unset or invalid one keeps its default. */
export function circuitConfigFromEnv(env: Env = process.env): CircuitConfig {
    const d = DEFAULT_CIRCUIT_CONFIG;
    const openMs = positiveIntEnv(env, 'LLM_CIRCUIT_OPEN_MS', d.openMs);
    return {
        ...d,
        failureThreshold: positiveIntEnv(
            env,
            'LLM_CIRCUIT_FAILURE_THRESHOLD',
            d.failureThreshold,
        ),
        windowMs: positiveIntEnv(env, 'LLM_CIRCUIT_WINDOW_MS', d.windowMs),
        openMs,
        maxOpenMs: Math.max(
            openMs,
            positiveIntEnv(env, 'LLM_CIRCUIT_MAX_OPEN_MS', d.maxOpenMs),
        ),
        maxResetMs: positiveIntEnv(
            env,
            'LLM_CIRCUIT_MAX_RESET_MS',
            d.maxResetMs,
        ),
    };
}

// ─── Keys ────────────────────────────────────────────────────────────────────

/** Opaque, stable digest of key material, so a breaker key never holds the
 *  ciphertext itself. */
function digest(value: string): string {
    return createHash('sha256').update(value).digest('hex').slice(0, 16);
}

/**
 * The provider ACCOUNT a slot draws on, without the model: the same identity
 * as the BYOK limiter minus `model`. Null for the managed/env slot, for a BYOK
 * slot without an org, and for an unparseable platform-funded model id.
 *
 * - BYOK: `org :: provider :: hash(key) :: baseURL`. The key is the apiKey
 *   ciphertext, or the credential id for a key-less credential. A credential
 *   id alone is never the account: ids like `cred-main` repeat across orgs.
 * - Platform-funded (`kodus`): `platform :: provider :: upstream`, org-blind —
 *   every org shares Kodus's upstream account, so its outage is learned once.
 *
 * Exported for an account-level breaker and for budget scopes; the circuit
 * itself keys per model (`circuitKeyFor`).
 */
export function providerAccountKey(
    slot: NormalizedModel | undefined,
    organizationId: string | undefined,
): string | null {
    if (!slot) return null;
    if (isPlatformFundedProvider(slot.provider)) {
        const ref = splitKodusModelId(slot.model);
        return ref
            ? ['platform', slot.provider, ref.upstream].join('::')
            : null;
    }
    if (!organizationId) return null;
    const secret = slot.apiKey || slot.credentialId;
    if (!secret) return null;
    return [
        organizationId,
        slot.provider,
        digest(secret),
        slot.baseURL || '',
    ].join('::');
}

/**
 * The breaker key: one served model on one provider account. Per model so one
 * upstream's outage (one model behind OpenRouter, say) never skips a healthy
 * sibling on the same account. Null means "no breaker for this attempt".
 */
export function circuitKeyFor(
    slot: NormalizedModel | undefined,
    organizationId: string | undefined,
): string | null {
    const account = providerAccountKey(slot, organizationId);
    if (!account || !slot) return null;
    const model = isPlatformFundedProvider(slot.provider)
        ? splitKodusModelId(slot.model)?.model
        : slot.model;
    return model ? `${account}::${model}` : null;
}

/** Log labels for a key — never key material. */
export interface CircuitLabels {
    provider?: string;
    model?: string;
    baseURLHost?: string;
    organizationId?: string;
    credentialId?: string;
}

export function circuitLabelsFor(
    slot: NormalizedModel | undefined,
    organizationId: string | undefined,
): CircuitLabels {
    let baseURLHost: string | undefined;
    if (slot?.baseURL) {
        try {
            baseURLHost = new URL(slot.baseURL).host;
        } catch {
            baseURLHost = undefined;
        }
    }
    return {
        provider: slot?.provider,
        model: slot?.model,
        baseURLHost,
        organizationId: isPlatformFundedProvider(slot?.provider)
            ? 'platform'
            : organizationId,
        credentialId: slot?.credentialId,
    };
}

// ─── What counts ─────────────────────────────────────────────────────────────

/**
 * Does this failure say the PROVIDER is unwell, in a way a different model can
 * fix? Only those trip the breaker:
 *  - a provider timeout (504/408/524, an undici timeout) or another TRANSIENT;
 *  - QUOTA_EXCEEDED;
 *  - our round-trip deadline at its full length (`source: 'round-trip'`). One
 *    taken from a caller's shorter budget says the caller was in a hurry, not
 *    that the provider stalled.
 *
 * Never counted: a caller's cancel or budget (queue timeouts included), any
 * RATE_LIMIT (the limiter owns 429s; counting load-driven ones would divert
 * one-shot calls the cascade deliberately keeps on the primary), and the
 * per-call categories failover already handles (context overflow, auth, model
 * not found / denied, unknown, parse errors).
 *
 * The caller also requires the failure to be cascade-worthy, so only failures
 * that actually moved the call to the next entry are counted.
 */
export function circuitCountsFailure(err: unknown): boolean {
    if (isCallerAbortOrBudget(err)) return false;
    const deadline = failureChain(err).find(isProviderDeadlineError);
    if (deadline) {
        return (deadline as { source?: unknown }).source === 'round-trip';
    }
    if (isProviderTimeout(err)) return true;
    const { category } = classifyLLMError(err);
    return (
        category === LlmErrorCategory.TRANSIENT ||
        category === LlmErrorCategory.QUOTA_EXCEEDED
    );
}

/**
 * How long the provider asked us to stay away (`retry-after` in seconds or as
 * an HTTP date, or `retry-after-ms`), read from any layer of the failure.
 */
export function retryAfterMsOf(
    err: unknown,
    now: number = Date.now(),
): number | undefined {
    for (const layer of failureChain(err)) {
        const headers = (layer as { responseHeaders?: unknown })
            .responseHeaders;
        if (!headers || typeof headers !== 'object') continue;
        const h = headers as Record<string, unknown>;
        const ms = Number(h['retry-after-ms']);
        if (Number.isFinite(ms) && ms > 0) return ms;
        const raw = h['retry-after'];
        if (typeof raw !== 'string' && typeof raw !== 'number') continue;
        const seconds = Number(raw);
        if (Number.isFinite(seconds) && seconds > 0) return seconds * 1000;
        const at = Date.parse(String(raw));
        if (Number.isFinite(at) && at > now) return at - now;
    }
    return undefined;
}

// ─── State machine ───────────────────────────────────────────────────────────

export type CircuitState = 'closed' | 'open' | 'half-open';

/**
 * - `run`: CLOSED, a normal attempt.
 * - `probe`: HALF_OPEN, this attempt holds the one probe lease.
 * - `skip`: not CLOSED, the entry is not last and the mode enforces.
 * - `forced`: not CLOSED but the entry is last, so it runs anyway.
 */
export type CircuitDecision = 'run' | 'probe' | 'skip' | 'forced';

/** Proof of which state an attempt was admitted under. Results reported
 *  against a stale generation are ignored. */
export interface CircuitTicket {
    readonly key: string;
    readonly generation: number;
    readonly isProbe: boolean;
    readonly forced: boolean;
}

export interface CircuitAdmission {
    decision: CircuitDecision;
    /** The state the key was in when admitted. */
    state: CircuitState;
    /** Null when the result of this attempt must not change state: mode off,
     *  a skip, or a shadow-mode run of an entry enforce would have skipped. */
    ticket: CircuitTicket | null;
    /** Shadow mode ran an entry that enforce mode would have skipped. */
    wouldSkip: boolean;
}

export interface CircuitAdmitOptions {
    /** The entry is the last in its chain: never skipped. */
    isLast: boolean;
    labels?: CircuitLabels;
    /** How long a probe lease lasts: the attempt's round-trip deadline plus a
     *  margin. A second probe is never leased while the first is running. */
    probeLeaseMs: number;
    /** The slot's limiter is cooling down after a 429: no probe is handed
     *  out into it. */
    isCoolingDown?: () => boolean;
}

interface CircuitEntry {
    state: CircuitState;
    generation: number;
    consecutiveFailures: number;
    lastFailureAt: number;
    /** The current open period; doubles on a failed probe. */
    openMs: number;
    /** OPEN until this time, then HALF_OPEN. */
    until: number;
    /** The probe lease, while one is out. */
    probe: { generation: number; expiresAt: number } | null;
    lastTouched: number;
    /** False while the key was only ever admitted as a last entry: it never
     *  diverted anything, so its transitions say "not enforced". */
    everNonLast: boolean;
    labels: CircuitLabels;
}

export interface ProviderCircuitOptions {
    config?: () => CircuitConfig;
    mode?: () => CircuitMode;
    now?: () => number;
}

export class ProviderCircuitBreaker {
    private readonly entries = new Map<string, CircuitEntry>();
    private readonly transitions = new Map<string, number>();
    private readonly config: () => CircuitConfig;
    private readonly mode: () => CircuitMode;
    private readonly now: () => number;

    constructor(opts: ProviderCircuitOptions = {}) {
        this.config = opts.config ?? (() => circuitConfigFromEnv());
        this.mode = opts.mode ?? (() => circuitModeFromEnv());
        this.now = opts.now ?? (() => Date.now());
    }

    currentMode(): CircuitMode {
        return this.mode();
    }

    /** Decide whether an attempt on `key` runs, and hand it its ticket. */
    admit(key: string, opts: CircuitAdmitOptions): CircuitAdmission {
        const mode = this.mode();
        if (mode === 'off') {
            return {
                decision: 'run',
                state: 'closed',
                ticket: null,
                wouldSkip: false,
            };
        }
        const now = this.now();
        const entry = this.touch(key, now, opts.labels);
        if (!opts.isLast) entry.everNonLast = true;
        this.advance(entry, now);

        const state = entry.state;
        if (state === 'closed') {
            return {
                decision: 'run',
                state,
                ticket: this.ticket(key, entry, false, false),
                wouldSkip: false,
            };
        }

        // The last entry always runs, whatever the state.
        if (opts.isLast) {
            return {
                decision: 'forced',
                state,
                ticket: this.ticket(key, entry, false, true),
                wouldSkip: false,
            };
        }

        // The one probe lease is taken synchronously, so concurrent admits
        // cannot both get it.
        if (
            state === 'half-open' &&
            !entry.probe &&
            !(opts.isCoolingDown?.() ?? false)
        ) {
            entry.probe = {
                generation: entry.generation,
                expiresAt: now + opts.probeLeaseMs,
            };
            return {
                decision: 'probe',
                state,
                ticket: this.ticket(key, entry, true, false),
                wouldSkip: false,
            };
        }

        if (mode === 'enforce') {
            return { decision: 'skip', state, ticket: null, wouldSkip: false };
        }
        return { decision: 'run', state, ticket: null, wouldSkip: true };
    }

    /** The attempt's provider answered. A probe or a forced run closes the
     *  key; a normal run resets the failure count. */
    onSuccess(ticket: CircuitTicket | null): void {
        const entry = this.current(ticket, 'success');
        if (!entry || !ticket) return;
        if (entry.state === 'closed') {
            entry.consecutiveFailures = 0;
            return;
        }
        if (ticket.isProbe || ticket.forced) {
            entry.openMs = this.config().openMs;
            entry.probe = null;
            entry.consecutiveFailures = 0;
            this.transition(entry, 'closed');
        }
    }

    /** The attempt ended in a COUNTED failure (`circuitCountsFailure`). */
    onFailure(
        ticket: CircuitTicket | null,
        opts: { retryAfterMs?: number } = {},
    ): void {
        const entry = this.current(ticket, 'failure');
        if (!entry || !ticket) return;
        const cfg = this.config();
        const now = this.now();

        if (entry.state === 'closed') {
            const fresh =
                entry.consecutiveFailures > 0 &&
                now - entry.lastFailureAt <= cfg.windowMs;
            entry.consecutiveFailures = fresh
                ? entry.consecutiveFailures + 1
                : 1;
            entry.lastFailureAt = now;
            if (entry.consecutiveFailures >= cfg.failureThreshold) {
                entry.openMs = cfg.openMs;
                this.open(entry, now, opts.retryAfterMs, cfg);
            }
            return;
        }

        if (ticket.isProbe && entry.state === 'half-open') {
            entry.openMs = Math.min(entry.openMs * 2, cfg.maxOpenMs);
            entry.probe = null;
            this.open(entry, now, opts.retryAfterMs, cfg);
        }
        // A forced failure leaves the key as it is: no doubling, `until`
        // unmoved — the entry ran only because nothing came after it.
    }

    /** The attempt ended without a verdict (an uncounted failure): a probe
     *  hands its lease back and the key stays HALF_OPEN. */
    release(ticket: CircuitTicket | null): void {
        if (!ticket?.isProbe) return;
        const entry = this.entries.get(ticket.key);
        if (entry?.probe?.generation === ticket.generation) {
            entry.probe = null;
        }
    }

    stateOf(key: string): CircuitState | undefined {
        const entry = this.entries.get(key);
        if (!entry) return undefined;
        this.advance(entry, this.now());
        return entry.state;
    }

    /** `llm_circuit_transitions{provider,state}` as a process counter. */
    transitionCounts(): Record<string, number> {
        return Object.fromEntries(this.transitions);
    }

    size(): number {
        return this.entries.size;
    }

    reset(): void {
        this.entries.clear();
        this.transitions.clear();
    }

    // ── internals ──

    private ticket(
        key: string,
        entry: CircuitEntry,
        isProbe: boolean,
        forced: boolean,
    ): CircuitTicket {
        return { key, generation: entry.generation, isProbe, forced };
    }

    /** The entry a ticket may still act on, or undefined (logged) when the
     *  state moved on since it was handed out. */
    private current(
        ticket: CircuitTicket | null,
        result: 'success' | 'failure',
    ): CircuitEntry | undefined {
        if (!ticket) return undefined;
        const entry = this.entries.get(ticket.key);
        if (!entry) return undefined;
        this.advance(entry, this.now());
        if (entry.generation !== ticket.generation) {
            logger.debug({
                message: `${LLM_CIRCUIT_TAG} ignored a stale ${result} for ${entry.labels.model ?? 'model'}`,
                context: 'ProviderCircuitBreaker',
                metadata: {
                    ...entry.labels,
                    ticketGeneration: ticket.generation,
                    generation: entry.generation,
                    state: entry.state,
                },
            });
            return undefined;
        }
        return entry;
    }

    /** Time-driven moves: OPEN past `until` → HALF_OPEN; an expired probe
     *  lease is withdrawn (its late result then reads as stale). */
    private advance(entry: CircuitEntry, now: number): void {
        if (entry.state === 'open' && now >= entry.until) {
            this.transition(entry, 'half-open');
        }
        if (
            entry.state === 'half-open' &&
            entry.probe &&
            now >= entry.probe.expiresAt
        ) {
            entry.probe = null;
            entry.generation++;
        }
    }

    private open(
        entry: CircuitEntry,
        now: number,
        retryAfterMs: number | undefined,
        cfg: CircuitConfig,
    ): void {
        let until = now + entry.openMs;
        if (retryAfterMs && retryAfterMs > 0) {
            until = Math.max(
                until,
                now + Math.min(retryAfterMs, cfg.maxResetMs),
            );
        }
        entry.until = until;
        this.transition(entry, 'open');
    }

    private transition(entry: CircuitEntry, state: CircuitState): void {
        entry.state = state;
        entry.generation++;
        const counter = `${entry.labels.provider ?? 'unknown'}:${state}`;
        this.transitions.set(counter, (this.transitions.get(counter) ?? 0) + 1);
        const notEnforced = entry.everNonLast
            ? ''
            : ' (not enforced: last entry)';
        const line = {
            message: `${LLM_CIRCUIT_TAG} ${state} ${entry.labels.provider ?? 'provider'}/${entry.labels.model ?? 'model'}${notEnforced}`,
            context: 'ProviderCircuitBreaker',
            metadata: {
                ...entry.labels,
                state,
                consecutiveFailures: entry.consecutiveFailures,
                openMs: entry.openMs,
                ...(state === 'open'
                    ? { openUntil: new Date(entry.until).toISOString() }
                    : {}),
                mode: this.mode(),
                enforced: entry.everNonLast,
            },
        };
        // Opening is the event someone acts on; the recovery steps are info.
        if (state === 'open') {
            logger.warn(line);
        } else {
            logger.log(line);
        }
    }

    private touch(
        key: string,
        now: number,
        labels: CircuitLabels | undefined,
    ): CircuitEntry {
        let entry = this.entries.get(key);
        if (entry) {
            // LRU: re-insert at the tail so eviction meets stale keys first.
            this.entries.delete(key);
        } else {
            this.evict(now);
            entry = {
                state: 'closed',
                generation: 0,
                consecutiveFailures: 0,
                lastFailureAt: 0,
                openMs: this.config().openMs,
                until: 0,
                probe: null,
                lastTouched: now,
                everNonLast: false,
                labels: labels ?? {},
            };
        }
        entry.lastTouched = now;
        if (labels) entry.labels = labels;
        this.entries.set(key, entry);
        return entry;
    }

    /** Drop CLOSED keys idle past `idleEvictMs`; past the hard cap, drop the
     *  least recently used CLOSED keys, then the oldest of any state. */
    private evict(now: number): void {
        const cfg = this.config();
        for (const [key, entry] of this.entries) {
            if (
                entry.state === 'closed' &&
                now - entry.lastTouched > cfg.idleEvictMs
            ) {
                this.entries.delete(key);
            }
        }
        if (this.entries.size < cfg.maxKeys) return;
        for (const [key, entry] of this.entries) {
            if (this.entries.size < cfg.maxKeys) return;
            if (entry.state === 'closed') this.entries.delete(key);
        }
        for (const key of this.entries.keys()) {
            if (this.entries.size < cfg.maxKeys) return;
            this.entries.delete(key);
        }
    }
}

/** The process-wide breaker `runWithModelFailover` consults. Mode and knobs
 *  are read from env on each use, so no state caches a stale value. */
export const providerCircuit = new ProviderCircuitBreaker();
