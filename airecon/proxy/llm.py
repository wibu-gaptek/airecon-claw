"""LLM backend for AIRecon (OpenAI + Anthropic compatible).

AIRecon talks to a single LLM gateway but can speak **two** wire formats, chosen
by the ``llm_provider`` config key:

* ``openai`` (default) → OpenAI Chat Completions, ``POST {base}/chat/completions``
* ``anthropic`` → Anthropic Messages API, ``POST {base}/messages`` (real Claude,
  or any gateway that exposes the Anthropic-compatible surface — e.g. 9router).

The gateway may itself proxy a local Ollama/LLM server, so there is no separate
native-LLM backend.

``LLMClient`` is fully self-contained: it owns the shared httpx client, the
request semaphore, dynamic timeouts and the performance-recording hooks the rest
of the agent relies on, and speaks the selected wire format directly (no SDK).

The streaming method yields **LLM-shaped** chunks — dicts of the form
``{"message": {"content"/"thinking": ..., "tool_calls": [...]}, "done": bool}``
with tool-call ``arguments`` decoded to a ``dict`` — which is exactly what
``loop_tool_cycle`` already consumes, so the ~30 downstream modules need no
changes to how they read responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict

import httpx

from .config import get_config
from .memory import get_memory_manager

logger = logging.getLogger("airecon.llm")

# Conversations longer than this (in tokens) trigger the agent loop's
# context-compaction pass. Kept here as the single source of truth that
# loop_lifecycle imports.
_CONTEXT_RESET_THRESHOLD = 65536

# Keys that are valid on an OpenAI chat message. Anything else AIRecon stores
# internally (``thinking``, ``_bucket``, ``name`` on non-tool roles, ...) is
# stripped before sending upstream.
_ALLOWED_MESSAGE_KEYS = frozenset(
    {"role", "content", "name", "tool_calls", "tool_call_id"}
)


def _to_openai_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AIRecon-internal tool calls to OpenAI request format.

    AIRecon stores ``arguments`` as a dict; OpenAI expects a JSON string and a
    stable ``id`` + ``type`` on every call.
    """
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        fn = tc.get("function", {}) or {}
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = "{}"
        out.append(
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args},
            }
        )
    return out


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize AIRecon's conversation into strict OpenAI chat messages.

    AIRecon often pairs a ``tool`` result with its call **by order** and stores
    no ``tool_call_id``. OpenAI-compatible providers reject such a message with
    ``tool_call_id is not set``, so we keep a FIFO of the ids assigned to the most
    recent assistant ``tool_calls`` and bind each following ``tool`` result to the
    next pending id.
    """
    converted: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []
    # Every id that actually appears in a preceding assistant ``tool_calls``.
    # A ``tool`` result is only valid if it references one of these — strict
    # gateways (e.g. the OpenAI Responses backend) reject a function-call output
    # whose call_id has no matching call with
    # "No tool call found for function call output with call_id ...".
    emitted_call_ids: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict):
            converted.append({"role": "user", "content": str(msg)})
            continue

        role = msg.get("role", "user")
        new_msg: dict[str, Any] = {"role": role}

        content = msg.get("content", "")
        # OpenAI allows null content only when tool_calls are present.
        new_msg["content"] = content if content is not None else ""

        if role == "assistant" and msg.get("tool_calls"):
            oai_calls = _to_openai_tool_calls(msg["tool_calls"])
            new_msg["tool_calls"] = oai_calls
            pending_tool_ids = [c["id"] for c in oai_calls]
            emitted_call_ids.update(pending_tool_ids)

        if role == "tool":
            tcid = msg.get("tool_call_id")
            if not tcid and pending_tool_ids:
                tcid = pending_tool_ids.pop(0)
            # A tool result MUST bind to a real tool_call that already appeared
            # in THIS payload. Fabricating an id (previous behaviour) created an
            # orphan the gateway rejects, so drop unbindable results instead —
            # their originating assistant turn was compacted/trimmed away.
            if not tcid or tcid not in emitted_call_ids:
                logger.debug(
                    "Dropping orphaned tool result (tool_call_id=%r) with no "
                    "matching assistant tool_call in payload",
                    tcid,
                )
                continue
            new_msg["tool_call_id"] = tcid
            name = msg.get("name")
            if name:
                new_msg["name"] = name

        new_msg = {k: v for k, v in new_msg.items() if k in _ALLOWED_MESSAGE_KEYS}

        # Strict gateways (e.g. Gemini via gemini-cli) reject a message/part with
        # empty text and return HTTP 400 "Request contains an invalid argument".
        # OpenAI tolerates empty content, so these slip in — notably after
        # compression, when an assistant turn that held only `thinking` is left
        # with empty content and no tool_calls. Never emit an empty part: drop
        # empty user/assistant/system messages, and give an empty tool result a
        # placeholder so its tool_call pairing is preserved.
        _has_text = bool(str(new_msg.get("content", "")).strip())
        if role == "tool":
            if not _has_text:
                new_msg["content"] = "[no output]"
        elif role == "assistant":
            if not _has_text and not new_msg.get("tool_calls"):
                continue
        else:
            if not _has_text:
                continue

        converted.append(new_msg)
    return converted


def _to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert OpenAI tool schemas to Anthropic ``input_schema`` tools.

    AIRecon builds tools as ``{"type": "function", "function": {name, description,
    parameters}}``; Anthropic wants ``{name, description, input_schema}``.
    """
    out: list[dict[str, Any]] = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            if isinstance(t, dict) and t.get("name"):
                out.append(
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("input_schema")
                        or {"type": "object", "properties": {}},
                    }
                )
            continue
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return out


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Sanitize AIRecon's conversation into the Anthropic Messages shape.

    Returns ``(system_prompt, messages)``: Anthropic takes ``system`` as a
    top-level string and requires ``user``/``assistant`` turns whose content is a
    list of blocks. Tool calls become ``tool_use`` blocks; tool results become
    ``user`` turns holding a ``tool_result`` block. Thinking blocks are NOT
    replayed (Anthropic requires a ``signature`` we do not hold) — only live
    streamed thinking reaches the agent loop.

    Mirrors ``_to_openai_messages``'s tool_call_id FIFO binding and orphan-drop
    rule, then coalesces consecutive same-role turns so strict Anthropic
    gateways accept the alternation.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []
    emitted_call_ids: set[str] = set()

    for msg in messages or []:
        if not isinstance(msg, dict):
            converted.append(
                {"role": "user", "content": [{"type": "text", "text": str(msg)}]}
            )
            continue

        role = msg.get("role", "user")

        if role == "system":
            text = msg.get("content", "")
            if text:
                system_parts.append(str(text))
            continue

        blocks: list[dict[str, Any]] = []

        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {}) or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
                pending_tool_ids.append(call_id)
                emitted_call_ids.add(call_id)
            converted.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            tcid = msg.get("tool_call_id")
            if not tcid and pending_tool_ids:
                tcid = pending_tool_ids.pop(0)
            if not tcid or tcid not in emitted_call_ids:
                logger.debug(
                    "Dropping orphaned tool result (tool_call_id=%r) with no "
                    "matching assistant tool_use in payload",
                    tcid,
                )
                continue
            content = msg.get("content", "")
            if not str(content).strip():
                content = "[no output]"
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tcid,
                            "content": str(content),
                        }
                    ],
                }
            )
            continue

        content = msg.get("content", "")
        if not str(content).strip():
            # Strict gateways reject an empty text part; drop it (mirrors OpenAI path).
            continue
        blocks.append({"type": "text", "text": str(content)})
        converted.append({"role": role, "content": blocks})

    # Anthropic requires alternating user/assistant turns.
    merged: list[dict[str, Any]] = []
    for m in converted:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append({"role": m["role"], "content": list(m["content"])})

    return "\n\n".join(p for p in system_parts if p.strip()), merged


# Hints like "reset after 6s", "retry after 10 seconds", "try again in 3s".
_RETRY_AFTER_RE = re.compile(
    r"(?:reset|retry|again|available)[^0-9]{0,20}?(\d+(?:\.\d+)?)\s*(s|sec|second)",
    re.IGNORECASE,
)


def _is_retryable_status(status_code: int) -> bool:
    """Transient HTTP statuses worth retrying: 5xx server errors and 429 rate
    limits (and 408 request timeout). 4xx client errors are not retried."""
    return status_code == 429 or status_code == 408 or 500 <= status_code < 600


def _retry_wait_seconds(status_code: int, message: str, attempt: int) -> float:
    """Backoff for a retryable status. Honors a "reset/retry after Ns" hint in
    the body (common on 429) when present, else exponential, capped at 30s."""
    base = 5.0 * (attempt + 1)
    if status_code == 429:
        m = _RETRY_AFTER_RE.search(message or "")
        if m:
            try:
                # +1s cushion so we retry just AFTER the quota resets.
                return min(30.0, max(base, float(m.group(1)) + 1.0))
            except (TypeError, ValueError):
                pass
    return min(30.0, base)


class LLMBackendHTTPError(RuntimeError):
    """Raised when the LLM backend returns an HTTP error status.

    Carries ``status_code`` so callers can distinguish retryable 5xx server
    errors from non-retryable 4xx client errors.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMClient:
    """Standalone OpenAI-compatible (LiteLLM/vLLM/hosted) LLM client."""

    _global_semaphore: asyncio.Semaphore | None = None
    _httpx_client: httpx.AsyncClient | None = None
    _initialized: bool = False
    _init_lock: asyncio.Lock | None = None
    _semaphore_init_lock = threading.Lock()
    # Models discovered at runtime to reject reasoning params (HTTP 400). Shared
    # across instances so the probe runs at most once per model per process.
    _reasoning_unsupported: set[str] = set()

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        cfg = get_config()
        host = (base_url or cfg.openai_base_url).rstrip("/")
        self._host = host
        self.model = model or cfg.openai_model

        self._api_key = (cfg.openai_api_key or "").strip()
        provider = str(getattr(cfg, "llm_provider", "openai") or "openai").strip().lower()
        self._provider = "anthropic" if provider in ("anthropic", "claude") else "openai"
        self._is_anthropic = self._provider == "anthropic"
        self._backend_name = (
            "Anthropic-compatible" if self._is_anthropic else "OpenAI-compatible"
        )
        self._supports_native_tools = bool(cfg.openai_supports_native_tools)

        # ── Deep-thinking support (restored for the OpenAI/gateway path) ──────
        # `llm_enable_thinking` is the master switch; `llm_thinking_request_mode`
        # decides HOW we ask the gateway for reasoning. We resolve a concrete
        # strategy for THIS model so a plain model (gpt-4o/gemini-flash) never
        # gets reasoning params it would reject, while a reasoning model
        # (o-series/qwen3/deepseek-r1) actually thinks before acting.
        self._enable_thinking = bool(getattr(cfg, "llm_enable_thinking", False))
        self._thinking_intensity = (
            str(getattr(cfg, "llm_thinking_mode", "low") or "low").strip().lower()
        )
        self._thinking_request_mode = (
            str(getattr(cfg, "llm_thinking_request_mode", "auto") or "auto")
            .strip()
            .lower()
        )
        self._thinking_strategy = self._resolve_thinking_strategy(
            self.model, self._thinking_request_mode
        )
        # `supports_thinking` drives the agent's per-iteration thinking gate.
        # It is True when we can actually surface OR request reasoning, so the
        # documented "Deep Thinking Model Support" feature works on this backend.
        self._supports_thinking = bool(cfg.openai_supports_thinking) or (
            self._enable_thinking and self._thinking_strategy != "off"
        )

        logger.info(
            "Initializing LLM client host=%s model=%s thinking=%s strategy=%s tools=%s",
            host,
            self.model,
            self._supports_thinking,
            self._thinking_strategy,
            self._supports_native_tools,
        )

        if LLMClient._global_semaphore is None:
            with LLMClient._semaphore_init_lock:
                if LLMClient._global_semaphore is None:
                    try:
                        _n = max(1, int(getattr(cfg, "llm_max_concurrent_requests", 1)))
                    except (TypeError, ValueError):
                        _n = 1
                    LLMClient._global_semaphore = asyncio.Semaphore(_n)
        self._request_semaphore = LLMClient._global_semaphore

    async def _async_init(self) -> None:
        if LLMClient._initialized:
            return

        if LLMClient._init_lock is None:
            LLMClient._init_lock = asyncio.Lock()

        async with LLMClient._init_lock:
            if LLMClient._initialized:
                return

            logger.info(
                "Initializing LLM httpx client (async init) for model: %s", self.model
            )
            if LLMClient._httpx_client is None:
                _cfg = get_config()
                _http_timeout = _cfg.llm_timeout
                LLMClient._httpx_client = httpx.AsyncClient(  # nosec B113: timeout configured below
                    timeout=httpx.Timeout(
                        _http_timeout, connect=10.0, read=_http_timeout, write=10.0
                    ),
                    headers={"Content-Type": "application/json"},
                )
                LLMClient._initialized = True
            logger.info("LLM httpx client initialized")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _endpoint(self) -> str:
        return "/messages" if self._is_anthropic else "/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if not self._api_key:
            return headers
        if self._is_anthropic:
            # Anthropic Messages API auth. Real Claude uses x-api-key; gateways
            # such as 9router also accept the Bearer form, so send both.
            headers["x-api-key"] = self._api_key
            headers["anthropic-version"] = "2023-06-01"
        headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def embed(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]] | None:
        """Embed texts via the gateway's /v1/embeddings endpoint.

        Reuses the same host, auth headers and shared httpx client as chat, so no
        new dependency is needed. Returns None (never raises) when embeddings are
        unavailable — no embedding_model configured, endpoint missing, or the
        gateway returns an error — so callers can fall back to lexical retrieval.
        """
        cfg = get_config()
        embed_model = (model or getattr(cfg, "embedding_model", "") or "").strip()
        if not embed_model or not texts:
            return None
        if not bool(getattr(cfg, "intelligence_embeddings_enabled", True)):
            return None

        try:
            await self._async_init()
        except Exception as exc:  # pragma: no cover - init failure is non-fatal
            logger.debug("embed: client init failed: %s", exc)
            return None

        payload: dict[str, Any] = {"model": embed_model, "input": texts}
        timeout = min(float(getattr(cfg, "llm_timeout", 30.0) or 30.0), 60.0)
        try:
            resp = await self._post("/embeddings", payload, timeout)
            if resp.status_code >= 400:
                logger.debug(
                    "embed: /embeddings returned HTTP %d for model=%s",
                    resp.status_code,
                    embed_model,
                )
                return None
            data = resp.json()
        except Exception as exc:
            logger.debug("embed: request failed: %s", exc)
            return None

        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return None
        # Preserve input order (OpenAI returns an `index` per row).
        try:
            ordered = sorted(rows, key=lambda r: int(r.get("index", 0)))
            vectors = [r.get("embedding") for r in ordered]
        except Exception:
            vectors = [r.get("embedding") for r in rows]
        if len(vectors) != len(texts) or not all(
            isinstance(v, list) and v for v in vectors
        ):
            return None
        return [[float(x) for x in v] for v in vectors]  # type: ignore[arg-type]

    def _apply_options(
        self, payload: dict[str, Any], options: dict[str, Any] | None
    ) -> None:
        """Translate AIRecon generation options to OpenAI request params."""
        cfg = get_config()
        max_tokens = cfg.openai_max_tokens
        temperature: float | None = getattr(cfg, "openai_temperature", None)
        if options:
            if options.get("num_predict") is not None:
                try:
                    np = int(options["num_predict"])
                    if np > 0:
                        max_tokens = np
                except (TypeError, ValueError):
                    pass
            if options.get("temperature") is not None:
                try:
                    temperature = float(options["temperature"])
                except (TypeError, ValueError):
                    pass
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)
        elif self._is_anthropic:
            # Anthropic requires max_tokens.
            payload["max_tokens"] = 4096
        if temperature is not None:
            payload["temperature"] = float(temperature)

    @staticmethod
    def _resolve_thinking_strategy(model: str, mode: str) -> str:
        """Decide how to request reasoning from the gateway.

        Returns one of: "off", "reasoning_effort", "enable_thinking".

        There is deliberately NO model-name list here. Guessing a model's
        reasoning capability from its name does not scale (GPT, Claude, Qwen,
        Gemini, Grok, DeepSeek… all differ and new models ship constantly).
        Instead:
          * An explicit ``llm_thinking_request_mode`` (off|reasoning_effort|
            enable_thinking) is always honored.
          * In ``auto`` we use the OpenAI-standard ``reasoning_effort`` parameter
            and rely on RUNTIME capability detection: if the backend rejects it
            with an HTTP 400 "unsupported parameter" (which is exactly what the
            OpenAI API returns for non-reasoning models), the client strips the
            param, remembers that for the model, and retries — see
            ``_maybe_degrade_reasoning``. ``enable_thinking`` (a non-standard
            vLLM/SGLang chat-template flag) stays an explicit opt-in.
        """
        mode = (mode or "auto").strip().lower()
        if mode in ("off", "reasoning_effort", "enable_thinking"):
            return mode
        return "reasoning_effort"

    def _reasoning_effort_value(self) -> str:
        return {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "adaptive": "medium",
        }.get(self._thinking_intensity, "medium")

    _THINKING_BUDGETS = {"low": 1024, "medium": 4096, "high": 16000, "adaptive": 4096}

    def _apply_thinking(self, payload: dict[str, Any], think: bool) -> None:
        """Translate the agent's `think` decision into gateway request params."""
        if not self._enable_thinking:
            return
        if self._is_anthropic:
            if not think:
                return
            budget = self._THINKING_BUDGETS.get(self._thinking_intensity, 4096)
            max_tokens = int(payload.get("max_tokens") or 4096)
            # Anthropic requires budget_tokens < max_tokens.
            budget = max(1024, min(budget, max_tokens - 1))
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            return
        strat = self._thinking_strategy
        if strat == "reasoning_effort":
            # Reasoning models bill/latency-scale with effort; only request it
            # when the agent actually wants a thinking turn — and never for a
            # model we've already learned (at runtime) rejects the parameter.
            if think and self.model not in LLMClient._reasoning_unsupported:
                payload["reasoning_effort"] = self._reasoning_effort_value()
        elif strat == "enable_thinking":
            # vLLM/SGLang pass this through to the model's chat template, so we
            # can both enable AND suppress reasoning per-turn to save tokens.
            ctk = dict(payload.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = bool(think)
            payload["chat_template_kwargs"] = ctk

    @staticmethod
    def _is_unsupported_reasoning_error(status_code: int, body: str) -> bool:
        """True if an HTTP 400 indicates the backend rejected reasoning params.

        OpenAI (and compatible gateways) return 400 with messages like
        "Unsupported parameter: 'reasoning_effort'" or "'reasoning.effort' is not
        supported with this model" for non-reasoning models. We detect that text
        so we can degrade gracefully instead of failing the run — no model-name
        list required.
        """
        if status_code != 400:
            return False
        b = (body or "").lower()
        mentions_reasoning = (
            "reasoning_effort" in b
            or "reasoning.effort" in b
            or "reasoning" in b
            or "thinking" in b
        )
        if not mentions_reasoning:
            return False
        return any(
            kw in b
            for kw in (
                "unsupported",
                "not supported",
                "does not support",
                "unknown",
                "unexpected",
                "invalid",
                "not permitted",
                "unrecognized",
            )
        )

    def _maybe_degrade_reasoning(
        self, payload: dict[str, Any], status_code: int, body: str
    ) -> bool:
        """Strip reasoning params if the backend rejected them; remember the model.

        Returns True when something was stripped (caller should retry the request
        without reasoning params).
        """
        if not self._is_unsupported_reasoning_error(status_code, body):
            return False
        removed = False
        if payload.pop("reasoning_effort", None) is not None:
            removed = True
        if payload.pop("thinking", None) is not None:
            removed = True
        ctk = payload.get("chat_template_kwargs")
        if isinstance(ctk, dict) and "enable_thinking" in ctk:
            ctk.pop("enable_thinking", None)
            removed = True
            if not ctk:
                payload.pop("chat_template_kwargs", None)
        if removed:
            LLMClient._reasoning_unsupported.add(self.model)
            logger.warning(
                "Backend rejected reasoning params for model=%s; disabling "
                "reasoning for this model and retrying.",
                self.model,
            )
        return removed

    async def _post(
        self, endpoint: str, json_data: dict[str, Any], timeout: float
    ) -> httpx.Response:
        async with self._request_semaphore:
            client = LLMClient._httpx_client
            if client is None:
                raise RuntimeError("HTTP client not initialized")
            url = f"{self._host}{endpoint}"
            to = httpx.Timeout(timeout, connect=10.0, read=timeout, write=10.0)
            resp = await client.request(
                "POST",
                url,
                json=json_data,
                headers=self._auth_headers(),
                timeout=to,
            )
            if resp.status_code >= 400:
                try:
                    body = resp.text[:500]
                except Exception:
                    body = "<unreadable body>"
                logger.error(
                    "LLM backend POST %s -> HTTP %d: %s",
                    endpoint,
                    resp.status_code,
                    body,
                )
                raise LLMBackendHTTPError(
                    resp.status_code,
                    f"{self._backend_name} returned HTTP {resp.status_code} "
                    f"for {endpoint}: {body}",
                )
            return resp

    # ── lifecycle / capability ───────────────────────────────────────────────
    async def reset_context(self, system_prompt: str | None = None) -> bool:
        # Remote stateless API — there is no server-side KV cache to reset.
        self._last_reset_error = ""
        self._last_reset_status = None
        return True

    async def unload_model(self) -> None:
        # No local VRAM to release for a remote backend.
        return None

    async def close(self) -> None:
        client = LLMClient._httpx_client
        if client is not None:
            await client.aclose()
        # Reset shared class-level state so a subsequent _async_init() rebuilds
        # a fresh client instead of re-using the now-closed one.
        LLMClient._httpx_client = None
        LLMClient._initialized = False

    async def _detect_capabilities(self) -> tuple[bool, bool] | None:
        return self._supports_thinking, self._supports_native_tools

    @property
    def supports_thinking(self) -> bool:
        return self._supports_thinking

    @property
    def supports_native_tools(self) -> bool:
        return self._supports_native_tools

    async def health_check(self) -> bool:
        try:
            client = LLMClient._httpx_client
            if client is None:
                return False
            resp = await client.get(
                f"{self._host}/models",
                headers=self._auth_headers(),
                timeout=httpx.Timeout(10.0),
            )
            # Some gateways gate /models behind auth or don't implement it;
            # treat any non-5xx response as "reachable".
            return resp.status_code < 500
        except Exception as e:
            logger.warning("LLM health check failed: %s", e)
            return False

    # ── non-streaming completion ─────────────────────────────────────────────
    async def complete(
        self,
        messages: list[dict[str, Any]],
        max_retries: int = 3,
        options: dict[str, Any] | None = None,
        operation: str = "compression",
    ) -> str:
        return await self._complete_impl(messages, max_retries, options, operation)

    async def _complete_impl(
        self,
        messages: list[dict[str, Any]],
        max_retries: int = 3,
        options: dict[str, Any] | None = None,
        operation: str = "compression",
    ) -> str:
        max_retries = max(0, max_retries)
        request_started = time.monotonic()
        if self._is_anthropic:
            system_prompt, anthropic_messages = _to_anthropic_messages(messages)
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": anthropic_messages,
                "stream": False,
            }
            if system_prompt:
                payload["system"] = system_prompt
        else:
            payload = {
                "model": self.model,
                "messages": _to_openai_messages(messages),
                "stream": False,
            }
        self._apply_options(payload, options)

        try:
            for attempt in range(max_retries + 1):
                try:
                    timeout = self._get_dynamic_timeout(operation)
                    resp = await self._post(self._endpoint(), payload, timeout)
                    data = resp.json()

                    content: str | None = None
                    if self._is_anthropic and isinstance(data, dict):
                        parts = [
                            b.get("text", "")
                            for b in (data.get("content") or [])
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        content = "".join(parts) if parts else None
                    elif isinstance(data, dict):
                        choices = data.get("choices") or []
                        if choices:
                            content = (choices[0].get("message") or {}).get("content")

                    if content is None:
                        logger.warning(
                            "LLM backend returned unexpected format: %r (attempt %d/%d)",
                            data,
                            attempt + 1,
                            max_retries + 1,
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** (attempt + 1))
                            continue
                        raise RuntimeError(
                            "Invalid LLM response: missing choices[0].message.content"
                        )

                    elapsed = time.monotonic() - request_started
                    self._record_response_time(elapsed)
                    self._record_model_performance(
                        operation=operation,
                        response_time_sec=elapsed,
                        success=True,
                        messages=messages,
                        options=options,
                    )
                    return content or ""

                except LLMBackendHTTPError as e:
                    if _is_retryable_status(e.status_code) and attempt < max_retries:
                        _wait = _retry_wait_seconds(e.status_code, str(e), attempt)
                        logger.warning(
                            "LLM HTTP %d (retryable) — backing off %.1fs "
                            "(attempt %d/%d)",
                            e.status_code,
                            _wait,
                            attempt + 1,
                            max_retries + 1,
                        )
                        await asyncio.sleep(_wait)
                        continue
                    raise
                except RuntimeError:
                    raise
                except httpx.HTTPStatusError as e:
                    if 500 <= e.response.status_code < 600 and attempt < max_retries:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    raise
                except (httpx.NetworkError, httpx.TimeoutException, asyncio.TimeoutError):
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    raise

            raise RuntimeError("Unexpected code path in LLM _complete_impl()")
        except Exception:
            elapsed = time.monotonic() - request_started
            self._record_model_performance(
                operation=operation,
                response_time_sec=elapsed,
                success=False,
                messages=messages,
                options=options,
            )
            raise

    # ── streaming chat ───────────────────────────────────────────────────────
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        think: bool = False,
        max_retries: int = 3,
        operation: str = "chat",
        stop_requested_fn: Callable[[], bool] | None = None,
    ) -> AsyncIterator[Any]:
        async for chunk in self._chat_stream_impl(
            messages,
            tools,
            options,
            think,
            max_retries,
            operation,
            stop_requested_fn,
        ):
            yield chunk

    async def _iter_stream(
        self,
        resp: httpx.Response,
        stop_requested_fn: Callable[[], bool] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Normalize an SSE response from either protocol into event dicts.

        Yields ``{"kind": usage|finish|thinking|text|tool, ...}`` so the caller's
        accumulator works identically for OpenAI ``chat.completion.chunk`` and
        Anthropic ``message`` events.
        """
        if self._is_anthropic:
            async for ev in self._iter_anthropic_stream(resp, stop_requested_fn):
                yield ev
            return
        async for ev in self._iter_openai_stream(resp, stop_requested_fn):
            yield ev

    async def _iter_openai_stream(
        self,
        resp: httpx.Response,
        stop_requested_fn: Callable[[], bool] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        async for raw_line in resp.aiter_lines():
            if stop_requested_fn and stop_requested_fn():
                return
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if isinstance(evt.get("usage"), dict):
                yield {"kind": "usage", "usage": evt["usage"]}

            choices = evt.get("choices") or []
            if not choices:
                continue
            choice0 = choices[0]
            delta = choice0.get("delta") or {}
            if choice0.get("finish_reason"):
                yield {"kind": "finish", "reason": choice0["finish_reason"]}

            # Reasoning models (o1, DeepSeek-R1, many hosted models) stream
            # chain-of-thought in `reasoning_content`/`reasoning`, not `content`.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                yield {"kind": "thinking", "text": reasoning}

            content = delta.get("content")
            if not content and delta.get("refusal"):
                content = delta["refusal"]
            if content:
                yield {"kind": "text", "text": content}

            for tc in delta.get("tool_calls") or []:
                tc_fn = tc.get("function") or {}
                yield {
                    "kind": "tool",
                    "index": tc.get("index", 0),
                    "id": tc.get("id"),
                    "name": tc_fn.get("name"),
                    "args": tc_fn.get("arguments"),
                }

    async def _iter_anthropic_stream(
        self,
        resp: httpx.Response,
        stop_requested_fn: Callable[[], bool] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        blocks: dict[int, dict[str, Any]] = {}
        async for raw_line in resp.aiter_lines():
            if stop_requested_fn and stop_requested_fn():
                return
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            etype = evt.get("type")
            if etype == "content_block_start":
                block = evt.get("content_block") or {}
                if block.get("type") == "tool_use":
                    idx = evt.get("index", 0)
                    blocks[idx] = {"id": block.get("id"), "name": block.get("name")}
                    # Emit up front so a tool with empty input still registers.
                    yield {
                        "kind": "tool",
                        "index": idx,
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "args": "",
                    }
            elif etype == "content_block_delta":
                delta = evt.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "thinking_delta" and delta.get("thinking"):
                    yield {"kind": "thinking", "text": delta["thinking"]}
                elif dtype == "text_delta" and delta.get("text"):
                    yield {"kind": "text", "text": delta["text"]}
                elif dtype == "input_json_delta":
                    yield {
                        "kind": "tool",
                        "index": evt.get("index", 0),
                        "id": blocks.get(evt.get("index", 0), {}).get("id"),
                        "name": blocks.get(evt.get("index", 0), {}).get("name"),
                        "args": delta.get("partial_json", ""),
                    }
            elif etype == "message_delta":
                stop = (evt.get("delta") or {}).get("stop_reason")
                if stop:
                    yield {
                        "kind": "finish",
                        "reason": "tool_calls" if stop == "tool_use" else stop,
                    }
                use = evt.get("usage")
                if isinstance(use, dict):
                    yield {
                        "kind": "usage",
                        "usage": {
                            "prompt_tokens": use.get("input_tokens", 0),
                            "completion_tokens": use.get("output_tokens", 0),
                        },
                    }
            elif etype == "message_stop":
                break

    async def _chat_stream_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        think: bool = False,
        max_retries: int = 3,
        operation: str = "chat",
        stop_requested_fn: Callable[[], bool] | None = None,
    ) -> AsyncIterator[Any]:
        cfg = get_config()
        if self._is_anthropic:
            system_prompt, anthropic_messages = _to_anthropic_messages(messages)
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": anthropic_messages,
                "stream": True,
            }
            if system_prompt:
                payload["system"] = system_prompt
            if tools:
                payload["tools"] = _to_anthropic_tools(tools)
        else:
            payload = {
                "model": self.model,
                "messages": _to_openai_messages(messages),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if tools:
                # AIRecon already builds tools in OpenAI function-schema shape
                # ({"type": "function", "function": {...}}), so pass them through.
                payload["tools"] = tools
        self._apply_options(payload, options)
        self._apply_thinking(payload, think)

        overall_timeout = cfg.llm_timeout
        chunk_timeout = cfg.llm_chunk_timeout
        request_started = time.monotonic()

        for attempt in range(max_retries + 1):
            tool_acc: dict[int, dict[str, str]] = {}
            tool_ids: dict[int, str] = {}
            usage: dict[str, Any] = {}
            produced_any = False
            finish_reason: str | None = None
            try:
                async with self._request_semaphore:
                    client = LLMClient._httpx_client
                    if client is None:
                        raise RuntimeError("HTTP client not initialized")
                    url = f"{self._host}{self._endpoint()}"
                    timeout_obj = httpx.Timeout(
                        overall_timeout, connect=10.0, read=chunk_timeout, write=10.0
                    )
                    async with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=self._auth_headers(),
                        timeout=timeout_obj,
                    ) as resp:
                        if resp.status_code >= 400:
                            # Read the body so the real cause (bad model name,
                            # rejected `tools`/`stream_options`, auth, ...) is
                            # visible instead of a generic error loop.
                            try:
                                await resp.aread()
                                body = resp.text[:500]
                            except Exception:
                                body = "<unreadable body>"
                            # Automatic capability detection: if the backend
                            # rejected reasoning params, strip them and retry the
                            # same request instead of failing the run.
                            if self._maybe_degrade_reasoning(
                                payload, resp.status_code, body
                            ):
                                continue
                            logger.error(
                                "LLM backend STREAM %s -> HTTP %d: %s",
                                self._endpoint(),
                                resp.status_code,
                                body,
                            )
                            raise LLMBackendHTTPError(
                                resp.status_code,
                                f"{self._backend_name} returned HTTP "
                                f"{resp.status_code} for {self._endpoint()}: {body}",
                            )
                        async for _ev in self._iter_stream(
                            resp, stop_requested_fn
                        ):
                            if _ev["kind"] == "usage":
                                usage = _ev["usage"]
                                continue
                            if _ev["kind"] == "finish":
                                finish_reason = _ev["reason"]
                                continue
                            if _ev["kind"] == "thinking":
                                produced_any = True
                                yield {
                                    "message": {
                                        "role": "assistant",
                                        "thinking": _ev["text"],
                                    },
                                    "done": False,
                                }
                                continue
                            if _ev["kind"] == "text":
                                produced_any = True
                                yield {
                                    "message": {
                                        "role": "assistant",
                                        "content": _ev["text"],
                                    },
                                    "done": False,
                                }
                                continue
                            if _ev["kind"] == "tool":
                                idx = _ev["index"]
                                slot = tool_acc.setdefault(
                                    idx, {"name": "", "args": ""}
                                )
                                if _ev.get("id"):
                                    tool_ids[idx] = _ev["id"]
                                if _ev.get("name"):
                                    slot["name"] = _ev["name"]
                                if _ev.get("args"):
                                    slot["args"] += _ev["args"]

                # Build the consolidated final chunk (LLM shape).
                final_tool_calls: list[dict[str, Any]] = []
                for idx in sorted(tool_acc):
                    slot = tool_acc[idx]
                    if not slot["name"]:
                        continue
                    raw_args = slot["args"].strip()
                    try:
                        parsed_args: Any = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        logger.warning(
                            "Tool-call arguments were not valid JSON for %s: %r",
                            slot["name"],
                            raw_args[:200],
                        )
                        parsed_args = {}
                    final_tool_calls.append(
                        {
                            "id": tool_ids.get(idx, f"call_{uuid.uuid4().hex[:24]}"),
                            "function": {
                                "name": slot["name"],
                                "arguments": parsed_args,
                            },
                        }
                    )

                final_message: dict[str, Any] = {"role": "assistant", "content": ""}
                if final_tool_calls:
                    final_message["tool_calls"] = final_tool_calls
                    produced_any = True

                # 200 but zero usable output. Almost always a provider-side safety
                # block or rejected request — fail fast with the real reason so the
                # loop doesn't retry and then print irrelevant advice.
                if not produced_any and finish_reason in ("content_filter", "error"):
                    self._record_model_performance(
                        operation=operation,
                        response_time_sec=time.monotonic() - request_started,
                        success=False,
                        messages=messages,
                        options=options,
                    )
                    raise RuntimeError(
                        f"{self._backend_name} returned an empty completion "
                        f"(finish_reason={finish_reason!r}) — the model produced no "
                        "content, reasoning, or tool calls. This is typically a "
                        "provider-side safety block or a rejected request."
                    )

                yield {
                    "message": final_message,
                    "done": True,
                    "prompt_eval_count": int(usage.get("prompt_tokens", 0) or 0),
                    "eval_count": int(usage.get("completion_tokens", 0) or 0),
                }

                elapsed_total = time.monotonic() - request_started
                if produced_any:
                    self._record_response_time(elapsed_total)
                self._record_model_performance(
                    operation=operation,
                    response_time_sec=elapsed_total,
                    success=True,
                    messages=messages,
                    options=options,
                )
                return

            except LLMBackendHTTPError as e:
                # Retry transient errors: 5xx server errors and 429 rate limits
                # (which often reset within seconds — honor the body's reset
                # hint). 4xx client errors (bad model, rejected params, auth)
                # won't resolve on retry, so fail fast.
                if _is_retryable_status(e.status_code) and attempt < max_retries:
                    _wait = _retry_wait_seconds(e.status_code, str(e), attempt)
                    logger.warning(
                        "LLM stream HTTP %d (retryable) — backing off %.1fs "
                        "(attempt %d/%d)",
                        e.status_code,
                        _wait,
                        attempt + 1,
                        max_retries + 1,
                    )
                    await asyncio.sleep(_wait)
                    continue
                self._record_model_performance(
                    operation=operation,
                    response_time_sec=time.monotonic() - request_started,
                    success=False,
                    messages=messages,
                    options=options,
                )
                raise
            except RuntimeError:
                # 4xx from the gateway (bad model, rejected params, auth, ...) or a
                # content-filter empty. Retrying won't help, so fail fast.
                self._record_model_performance(
                    operation=operation,
                    response_time_sec=time.monotonic() - request_started,
                    success=False,
                    messages=messages,
                    options=options,
                )
                raise
            except httpx.HTTPStatusError as e:
                if 500 <= e.response.status_code < 600 and attempt < max_retries:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                self._record_model_performance(
                    operation=operation,
                    response_time_sec=time.monotonic() - request_started,
                    success=False,
                    messages=messages,
                    options=options,
                )
                raise
            except (httpx.NetworkError, httpx.TimeoutException, asyncio.TimeoutError):
                if attempt < max_retries:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                self._record_model_performance(
                    operation=operation,
                    response_time_sec=time.monotonic() - request_started,
                    success=False,
                    messages=messages,
                    options=options,
                )
                raise

        # Reaching here means the loop exhausted every attempt without yielding a
        # terminal chunk or raising — the only way in is a reasoning-degrade
        # `continue` on the final attempt. Guarantee a terminal signal so the
        # async generator never ends silently (which would look like an empty
        # response to the caller).
        self._record_model_performance(
            operation=operation,
            response_time_sec=time.monotonic() - request_started,
            success=False,
            messages=messages,
            options=options,
        )
        raise RuntimeError(
            f"{self._backend_name} stream ended without a response after "
            f"{max_retries + 1} attempt(s)."
        )

    # ── performance / timeout bookkeeping ────────────────────────────────────
    def _record_model_performance(
        self,
        operation: str,
        response_time_sec: float,
        success: bool,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> None:
        model_name = str(getattr(self, "model", "") or "").strip()
        if not model_name:
            return
        try:
            get_memory_manager().record_model_performance(
                model_name=model_name,
                task_type=self._normalize_task_type(operation),
                response_time_sec=max(0.0, float(response_time_sec or 0.0)),
                success=success,
                context_size_used=self._estimate_context_size(messages, options),
            )
        except Exception as exc:
            logger.debug("Failed to record model performance: %s", exc)

    def _get_dynamic_timeout(self, operation: str = "inference") -> float:
        cfg = get_config()
        if operation == "compression":
            return max(180.0, cfg.llm_chunk_timeout)
        return cfg.llm_chunk_timeout

    def _record_response_time(self, response_time: float) -> None:
        """Record a response time for adaptive timeout calculations."""
        if not hasattr(self, "_response_times"):
            self._response_times = []
        if not hasattr(self, "_max_response_times"):
            self._max_response_times = 20
        self._response_times.append(response_time)
        max_len = self._max_response_times
        if len(self._response_times) > max_len:
            self._response_times = self._response_times[-max_len:]

    def get_response_time_stats(self) -> Dict[str, float]:
        """Avg/min/max of the last 10 response times."""
        if not hasattr(self, "_response_times"):
            self._response_times = []
        times = self._response_times[-10:] if self._response_times else []
        if not times:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "count": len(self._response_times),
        }

    @staticmethod
    def _normalize_task_type(operation: str) -> str:
        task_type = str(operation or "").strip().lower()
        aliases = {
            "inference": "chat",
            "validation": "analysis",
            "summarization": "compression",
        }
        return aliases.get(task_type, task_type or "general")

    @staticmethod
    def _estimate_context_size(
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> int:
        total_chars = 0
        for message in messages or []:
            if not isinstance(message, dict):
                total_chars += len(str(message))
                continue
            for key in ("content", "thinking", "tool_calls"):
                value = message.get(key)
                if value in (None, ""):
                    continue
                if isinstance(value, str):
                    total_chars += len(value)
                else:
                    try:
                        total_chars += len(json.dumps(value, ensure_ascii=False))
                    except Exception:
                        total_chars += len(str(value))

        estimated_tokens = total_chars // 4
        if estimated_tokens > 0:
            return estimated_tokens
        return 0


def create_llm_client(model: str | None = None) -> LLMClient:
    """Return the configured OpenAI-compatible LLM client."""
    cfg = get_config()
    return LLMClient(model=model or cfg.openai_model)
