# free-model-aggregator-01

One small server that stands in front of ten free model providers and
answers a client's requests with the best free model that can serve them.

## What it does

The server is briefed at startup by the file named in `$MAKEFUN_PROMPT_FILE`.
The brief carries three documents: the provider directory (base path, API
key, wire protocol for each of ten providers), a quality table rating every
free model, and the workload manifest (twenty requests, each with its own
requirements — tools, vision, JSON mode, minimum context).

On startup the program probes each provider's own model listing, registers
one catalog with the environment at `$MAKEFUN_ENV_URL`, then serves the
twenty requests one at a time. When the environment reports the workload
done, the program exits 0. `python3 main.py --list` prints the catalog
document it registers (or `{"providers":[]}` when there is no brief).

## How it works

- **Two client wires.** Clients may speak OpenAI (`POST
  /v1/chat/completions`) or Anthropic (`POST /v1/messages`), as JSON or as
  server-sent events. `GET /v1/models` answers in the shape the caller's
  credential names. Every answer is rendered in the wire the client used.
- **Three provider wires.** Each request is translated into the chosen
  provider's own protocol — OpenAI, Anthropic, or Gemini — with system
  text, messages, images, tools, `max_tokens` and stop sequences carried
  across faithfully, and the provider's answer text, usage figures and
  stop reason are returned to the client untouched.
- **Routing.** Candidates are the free models that meet the request's
  requirements, best quality rating first. An Anthropic-wire provider is
  never given an OpenAI JSON-mode request (its echo could not confirm the
  mode). The server always calls a provider non-streamed and renders the
  stream itself when the client asked for one, so exactly one successful
  provider call serves each request; refusals before it are free.
- **Provider health.** Refusals are remembered. One coded as final for the
  trial — budget spent, outage, wrong key — retires the provider, and the
  sealed per-provider call ceiling from the environment's manifest caps it
  regardless. A plain rate-limit refusal only makes the provider less
  preferred among equals: it is one sealed call number refusing, not a
  verdict on the provider.
- **Determinism.** No wall clock and no randomness feed any decision:
  routing is a pure function of the brief, the requests and the providers'
  answers, so a replayed trial makes the same calls and the environment's
  exported state matches byte-for-byte.

## Running it

```
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt \
python3 main.py
```

Python 3.13, standard library only, no dependencies.
