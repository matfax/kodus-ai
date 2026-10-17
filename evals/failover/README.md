# failover

> - **Answers:** When an org's primary model fails, stalls or is rate-limited, does the call move to the next model in its fallback chain, and only when it should?
> - **Runs:** every engine PR (`evals/wiring-smoke.js`, step `failover`).
> - **Run it:** `node evals/failover/run.js --gate` (one case: `--case=<id>`)
> - **Gate:** `--gate` exits 1 on any failed check; every case is deterministic.
> - **Cost:** none: two local fake servers, no key. About 10 seconds, most of it the circuit case waiting out its open periods.

The eval runs `LLM.run` in agent-loop mode, the path the finder and the
verifier take, for an org whose stored BYOK config holds a fallback chain:
production routing (`libs/llm/resolve-task-model.ts`) stamps the chain on the
slot, and the cascade (`libs/llm/model-failover.ts`), the circuit breaker
(`libs/llm/provider-circuit.ts`), the round-trip deadline
(`libs/llm/provider-deadline-error.ts`), the BYOK wrapper and the
openai_compatible provider all run for real over HTTP. Only the vendors are
fake: each case's primary model sits on a `evals/shared/fake-llm-server.js`
that answers it with a fault (an HTML 504, a JSON 503, a 429, a hang), and the
fallback is the scripted `eval-fake` model. The env-only `eval-fake` route
cannot test this: it is one managed slot, with no chain and no breaker.

`cases.js` holds the cases. Each one checks the outcome the caller and the
logs see: which chain entry answered, which hops `ModelFailoverEvent` recorded
and why, how many requests the faulty model got, how long the call took, and
the breaker state it left. Every case runs in one process, with its own
primary model, because the breaker keeps its state per model in memory.

The runner sets the knobs the cases depend on (`ENV` in `evals/failover/run.js`),
so a change to their defaults does not move the eval: the provider-timeout
cascade on, a 1s round-trip deadline, the breaker in `enforce` mode, opened by
one counted failure, and a 1s open period.

## What it measures

The runner prints three numbers (`measures` in the `--output` result), each
against its baseline from before the fallback chain:

| Metric | Before | This eval |
|---|---|---|
| Share of HTML 504 "Gateway Timeout" faults that cascade (`504-html`, `html504Cascaded`) | 0%: the status phrase read as a caller timeout and stopped the cascade | 100% |
| Time to give up on a primary that hangs before its first step (`hang`, `hungPrimaryGiveUpMs`) | about 80 min in a loop (undici's 20 min bound on each of the 4 SDK attempts), 20 min one-shot | the deadline, plus under 2s |
| Requests to a model with an open circuit on the next call (`circuit`, `openCircuitRequests`) | every call: there was no breaker | 0 |

Setting `LLM_PROVIDER_TIMEOUT_CASCADE` to `off` in `ENV` (`evals/failover/run.js`)
restores the old timeout gate: the `504-html` and `chain` cases then fail,
which is the 0% above. With `LLM_CIRCUIT_MODE` at `shadow` the `circuit` case
fails instead.

A failed check names what it saw (the hops, the request count, the time), so
the line in the output is the diagnosis.
