# openclaw-exploop

> A training-free experience loop for LLMs: accumulate conversational wisdom, inject it at inference time — zero GPU, zero parameter updates.

## Motivation

Full reinforcement learning for LLMs (GRPO, PPO) delivers real gains but costs **\~$10,000+ per training run** and requires significant GPU infrastructure. Lighter approaches like [On-Policy Distillation (OPD)](https://github.com/Gen-Verse/OpenClaw-RL) bring the cost down to ~$18/run but still require parameter updates, a training pipeline, and model deployment. Yet in many practical scenarios the base model is *already good enough* — what it lacks is memory of past mistakes and the lessons learned from them.

**ExpLoop** takes a radically different path: **zero parameter updates, ever.** Inspired by the [GRPO](https://arxiv.org/abs/2402.03300) insight that group-relative comparison can surface quality signals from a model's own outputs, ExpLoop captures those signals not as weight gradients but as natural-language *experience hints*. A Judge LLM analyses (response, user follow-up) pairs to infer what was lacking, and those hints are persisted in a keyword-indexed store. On every future request, relevant hints are retrieved and injected into the prompt — giving the base model accumulated "wisdom" at effectively **$0 training cost**.

## The ExpLoop Cycle

```
┌──────────────────────────────────────────────────────────────┐
│                        The ExpLoop Cycle                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1 — INTERCEPT                                          │
│    User request arrives → retrieve relevant past             │
│    experiences → inject them into the system prompt           │
│                                                              │
│          ┌─────────┐       ┌──────────────┐                  │
│          │ Request  │──────▶│PromptInjector│──┐               │
│          └─────────┘       └──────────────┘  │               │
│                                  ▲           │               │
│                                  │           ▼               │
│                          ┌──────────────┐  ┌──────────┐      │
│                          │ExperienceStore│  │ Upstream │      │
│                          │  (JSON file)  │  │   LLM    │      │
│                          └──────────────┘  └────┬─────┘      │
│                                  ▲              │            │
│  Step 2 — RESPOND                │              │            │
│    Forward enriched prompt to    │              ▼            │
│    upstream LLM → return         │        ┌──────────┐       │
│    response to user              │        │ Response │       │
│                                  │        └────┬─────┘       │
│  Step 3 — EXTRACT (async)        │             │             │
│    On the *next* user message,   │             ▼             │
│    pair (prev response, next     │     ┌────────────────┐    │
│    message) → Judge LLM infers   │     │  Experience    │    │
│    an improvement hint            │     │  Extractor     │    │
│                                  │     └───────┬────────┘    │
│  Step 4 — STORE                  │             │             │
│    Good hints are saved to the   └─────────────┘             │
│    experience store for future                               │
│    retrieval (loop back to Step 1)                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**The key insight**: every time a user follows up (corrects, clarifies, or asks for more), the follow-up *implicitly* tells us what the model got wrong. A Judge LLM makes that implicit signal explicit, and the experience library makes it persistent.

## Installation

```bash
cd openclaw-exploop
pip install -r requirements.txt
```

Requirements: Python 3.10+, no GPU needed.

## Quick Start

**1. Start the server**

```bash
export UPSTREAM_BASE_URL=https://api.openai.com   # your LLM provider
export UPSTREAM_API_KEY=sk-xxxxxxxx
python exploop_server.py
```

**2. Point your client at the proxy**

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="anything",  # auth is handled by the proxy
)
```

**3. Chat normally**

```python
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain transformers"}],
)
print(resp.choices[0].message.content)
```

That's it. The proxy transparently accumulates experience and injects it into future requests. Over time, responses improve without any model fine-tuning.

## Configuration

All settings are via environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `UPSTREAM_BASE_URL` | **Yes** | — | Base URL of the real LLM API |
| `UPSTREAM_API_KEY` | **Yes** | — | API key for the upstream LLM |
| `JUDGE_BASE_URL` | No | same as upstream | Base URL for the Judge LLM |
| `JUDGE_API_KEY` | No | same as upstream | API key for the Judge LLM |
| `JUDGE_MODEL` | No | `gpt-4o-mini` | Model name for hint extraction |
| `EXPERIENCE_STORE_PATH` | No | `experiences.json` | Path to the experience JSON file |
| `TOP_K_EXPERIENCES` | No | `3` | Number of experiences to inject per request |
| `PORT` | No | `8080` | Server listen port |

## Architecture

```
openclaw-exploop/
├── exploop_server.py          # FastAPI proxy server (entrypoint)
├── experience_extractor.py    # Judge LLM client for hint extraction
├── experience_store.py        # Keyword-indexed JSON experience library
├── prompt_injector.py         # Injects retrieved experiences into prompts
├── requirements.txt           # Python dependencies
├── __init__.py                # Package exports
└── README.md                  # This file
```

| Component | Responsibility |
|---|---|
| **Server** (`exploop_server.py`) | OpenAI-compatible proxy. Intercepts requests, manages session history, coordinates injection and extraction. Runs a background asyncio worker for non-blocking hint extraction. |
| **Extractor** (`experience_extractor.py`) | Calls a Judge LLM to analyze (response, next_user_message) pairs and produce actionable improvement hints. Filters out trivial/empty hints. |
| **Store** (`experience_store.py`) | Thread-safe, keyword-indexed JSON file. Stores hints with metadata (timestamps, snippets, session IDs). Retrieval scores experiences by keyword overlap with the current query. |
| **Injector** (`prompt_injector.py`) | Retrieves top-K experiences from the Store and prepends them to the system message. Respects a character budget to avoid prompt bloat. |

## Comparison with Related Work

| Approach | Parameter Update | Cost | How it improves |
|---|---|---|---|
| Full RL (GRPO/PPO) | ✅ Yes | ~$10,000/run | Weight gradients from reward signal |
| OPD (Thinking Machines Lab) | ✅ Yes | ~$18/run | On-policy distillation from teacher LLM |
| **ExpLoop (ours)** | ❌ No | ~$0 | Inject accumulated hints at prompt time |
| Raw prompting | ❌ No | $0 | Manual prompt engineering only |

**Key trade-off**: ExpLoop cannot exceed the base model's inherent capability ceiling. But for many deployment scenarios — customer support bots, internal copilots, domain-specific assistants — the base model is already capable; what it needs is *memory* of past interactions, not new parameters. ExpLoop provides that memory at zero training cost.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (streaming & non-streaming) |
| `/v1/experiences/stats` | GET | Experience store statistics and queue depth |
| `/health` | GET | Health check |

## FAQ

**Q: How is this different from RAG?**
A: RAG retrieves *external documents* to answer questions. ExpLoop retrieves *self-generated improvement hints* to avoid repeating past mistakes. The knowledge source is the model's own conversation history distilled into actionable lessons — not a document corpus. Think of it as "experiential memory" vs. "reference library."

**Q: Why "training-free"? Isn't prompt injection just a workaround?**
A: It depends on your goal. If you need to fundamentally change what a model knows (new languages, new domains), you need training. But if you need the model to *stop making the same mistakes* in a specific deployment context, injecting past lessons at prompt time is simpler, cheaper, and instantly reversible. No GPU cluster, no training pipeline, no model deployment — just a proxy server and a JSON file.

**Q: Does this actually improve over time?**
A: Yes, but with diminishing returns. Early conversations accumulate high-value hints quickly. Over hundreds of sessions, the experience library becomes a rich "playbook" that prevents common failure modes. However, since the base model is unchanged, there is a ceiling set by the model's inherent capability.

**Q: What's the relationship to GRPO?**
A: GRPO showed that you can surface quality signals by comparing multiple model outputs within a group. ExpLoop borrows that *insight* — that relative comparison reveals what's good and what's not — but applies it differently. Instead of using those signals as reward for gradient updates, we use a Judge LLM to articulate them as natural-language hints and store them for future prompt injection.

**Q: Can I use a cheap model as the Judge?**
A: Yes. `gpt-4o-mini` works well as a default Judge. The Judge only needs to identify *what was lacking* in a response — a simpler task than generating a good response from scratch. You can even use the same model as both upstream and Judge.

**Q: What about privacy? Are my conversations stored?**
A: The experience store saves short *hints* (1-4 sentences) plus 300-character snippets of the response and follow-up. Full conversation history is held in memory only and **lost on server restart** — this is by design. You control the store file location via `EXPERIENCE_STORE_PATH`.

**Q: Can I seed the experience store manually?**
A: Yes. The store is a plain JSON array. You can add entries manually following the schema documented in `experience_store.py`. Each entry needs at minimum an `"id"`, `"hint"`, and `"keywords"` list.

## License

MIT
