"""
Training-Free GRPO API Server

Proxies requests to a real LLM, intercepts conversations, and
asynchronously collects experience hints for future prompt injection.

Usage::

    UPSTREAM_BASE_URL=https://api.openai.com \\
    UPSTREAM_API_KEY=sk-xxx \\
    python openclaw_tf_server.py

The server exposes an OpenAI-compatible ``/v1/chat/completions``
endpoint.  Point any client at ``http://localhost:8080`` and chat
normally — experience accumulation happens transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from experience_extractor import ExperienceExtractor
from experience_store import ExperienceStore
from prompt_injector import PromptInjector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("tf-server")

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

UPSTREAM_BASE_URL: str = os.environ.get(
    "UPSTREAM_BASE_URL", ""
).rstrip("/")
UPSTREAM_API_KEY: str = os.environ.get("UPSTREAM_API_KEY", "")

JUDGE_BASE_URL: str = os.environ.get(
    "JUDGE_BASE_URL", ""
) or UPSTREAM_BASE_URL
JUDGE_API_KEY: str = os.environ.get(
    "JUDGE_API_KEY", ""
) or UPSTREAM_API_KEY
JUDGE_MODEL: str = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

EXPERIENCE_STORE_PATH: str = os.environ.get(
    "EXPERIENCE_STORE_PATH", "experiences.json"
)
TOP_K_EXPERIENCES: int = int(
    os.environ.get("TOP_K_EXPERIENCES", "3")
)
PORT: int = int(os.environ.get("PORT", "8080"))

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

store = ExperienceStore(store_path=EXPERIENCE_STORE_PATH)
extractor = ExperienceExtractor(
    base_url=JUDGE_BASE_URL,
    api_key=JUDGE_API_KEY,
    model=JUDGE_MODEL,
)
injector = PromptInjector(store=store)

# session_id → list of {"role": ..., "content": ...}
conversation_history: dict[str, list[dict[str, str]]] = (
    defaultdict(list)
)


@dataclass
class PendingPair:
    """A (response, next_user_message) pair waiting for hint extraction."""
    response_text: str
    next_user_message: str
    session_id: str
    created_at: float = field(default_factory=time.time)


hint_queue: asyncio.Queue[PendingPair] = asyncio.Queue()

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _hint_worker() -> None:
    """Consume pending pairs, extract hints, store good ones."""
    logger.info("[hint_worker] started")
    while True:
        pair: PendingPair = await hint_queue.get()
        try:
            hint = await extractor.extract_hint(
                response_text=pair.response_text,
                next_user_message=pair.next_user_message,
                session_id=pair.session_id,
            )
            if hint:
                store.add(
                    hint=hint,
                    response_text=pair.response_text,
                    next_state_text=pair.next_user_message,
                    session_id=pair.session_id,
                )
                logger.info(
                    "[hint_worker] stored hint for session=%s",
                    pair.session_id,
                )
            else:
                logger.debug(
                    "[hint_worker] no usable hint for session=%s",
                    pair.session_id,
                )
        except Exception:
            logger.exception(
                "[hint_worker] unexpected error for session=%s",
                pair.session_id,
            )
        finally:
            hint_queue.task_done()


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start the background hint-extraction worker on startup."""
    worker_task = asyncio.create_task(_hint_worker())
    logger.info(
        "[server] Training-Free GRPO server starting on port %d", PORT,
    )
    logger.info(
        "[server] upstream=%s  judge=%s  store=%s  top_k=%d",
        UPSTREAM_BASE_URL, JUDGE_BASE_URL,
        EXPERIENCE_STORE_PATH, TOP_K_EXPERIENCES,
    )
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("[server] shutdown complete")


app = FastAPI(
    title="Training-Free GRPO Proxy",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_session_id(body: dict[str, Any]) -> str:
    """Derive a session id from the request body.

    Uses ``session_id`` or ``user`` field if present, otherwise
    generates a new UUID.
    """
    return str(
        body.get("session_id")
        or body.get("user")
        or uuid.uuid4().hex[:12]
    )


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _upstream_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return headers


# ---------------------------------------------------------------------------
# Streaming proxy
# ---------------------------------------------------------------------------

async def _stream_upstream(
    payload: dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    """Yield raw SSE chunks from the upstream LLM."""
    url = f"{UPSTREAM_BASE_URL}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", url,
            json=payload,
            headers=_upstream_headers(),
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk


async def _collect_streaming_content(
    payload: dict[str, Any],
) -> tuple[AsyncGenerator[bytes, None], asyncio.Future[str]]:
    """Return a streaming generator AND a future that resolves to
    the full assistant content once the stream is consumed.

    We tee the stream: yield each chunk to the client while
    accumulating delta content tokens.
    """
    loop = asyncio.get_running_loop()
    content_future: asyncio.Future[str] = loop.create_future()
    collected_parts: list[str] = []

    async def _gen() -> AsyncGenerator[bytes, None]:
        url = f"{UPSTREAM_BASE_URL}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", url,
                    json=payload,
                    headers=_upstream_headers(),
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_lines():
                        line = chunk.strip()
                        yield (chunk + "\n").encode()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            continue
                        try:
                            obj = json.loads(data_str)
                            delta = (
                                obj.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                collected_parts.append(delta)
                        except (json.JSONDecodeError, IndexError):
                            pass
        except Exception as exc:
            logger.error(
                "[stream] upstream error: %s", exc,
            )
            if not content_future.done():
                content_future.set_result("")
            return
        if not content_future.done():
            content_future.set_result("".join(collected_parts))

    return _gen(), content_future


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """OpenAI-compatible chat completions endpoint.

    1. Inject past experiences into the prompt.
    2. Forward to upstream LLM.
    3. Enqueue the (prev_response, current_user_msg) pair for
       asynchronous hint extraction.
    """
    if not UPSTREAM_BASE_URL:
        return JSONResponse(
            {"error": "UPSTREAM_BASE_URL not configured"},
            status_code=500,
        )

    body: dict[str, Any] = await request.json()
    messages: list[dict[str, Any]] = body.get("messages", [])
    session_id = _resolve_session_id(body)
    is_stream = body.get("stream", False)

    user_text = _last_user_text(messages)

    # --- Enqueue previous turn for hint extraction ----------------------
    history = conversation_history[session_id]
    if history and user_text:
        # Previous assistant response + current user message
        prev_assistant = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                prev_assistant = msg.get("content", "")
                break
        if prev_assistant:
            await hint_queue.put(PendingPair(
                response_text=prev_assistant,
                next_user_message=user_text,
                session_id=session_id,
            ))

    # Record user message in history.
    if user_text:
        history.append({"role": "user", "content": user_text})

    # --- Inject experiences ---------------------------------------------
    enriched_messages = injector.inject(
        messages, query=user_text, top_k=TOP_K_EXPERIENCES,
    )
    body["messages"] = enriched_messages

    # --- Forward to upstream --------------------------------------------
    if is_stream:
        gen, content_future = await _collect_streaming_content(body)

        async def _wrapped_stream() -> AsyncGenerator[bytes, None]:
            async for chunk in gen:
                yield chunk
            # After stream completes, record assistant reply.
            full_content = await content_future
            if full_content:
                history.append({
                    "role": "assistant",
                    "content": full_content,
                })

        return StreamingResponse(
            _wrapped_stream(),
            media_type="text/event-stream",
        )

    # Non-streaming path.
    url = f"{UPSTREAM_BASE_URL}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                url, json=body, headers=_upstream_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("[proxy] upstream HTTP %s", exc.response.status_code)
        return JSONResponse(
            {"error": f"upstream error: {exc.response.status_code}"},
            status_code=exc.response.status_code,
        )
    except (httpx.RequestError, Exception) as exc:
        logger.error("[proxy] upstream request failed: %s", exc)
        return JSONResponse(
            {"error": f"upstream request failed: {exc}"},
            status_code=502,
        )

    # Record assistant reply.
    try:
        assistant_content = (
            data["choices"][0]["message"]["content"]
        )
        history.append({
            "role": "assistant",
            "content": assistant_content,
        })
    except (KeyError, IndexError):
        pass

    return JSONResponse(data)


@app.get("/v1/experiences/stats")
async def experience_stats() -> dict[str, Any]:
    """Return experience store statistics."""
    stats = store.stats()
    stats["pending_hints"] = hint_queue.qsize()
    stats["active_sessions"] = len(conversation_history)
    return stats


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic health check."""
    return {"status": "ok", "service": "training-free-grpo"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "openclaw_tf_server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
