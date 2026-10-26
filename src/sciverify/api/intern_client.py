"""书生 Intern-S2 官方 Chat Completions 客户端（真实 HTTP 调用，非流式）.

规则（SCIVERIFY_FINAL_EXECUTION_PLAN §4.2 / Gate A）：
- 工具调用必须使用非流式请求（stream=false），thinking_mode 默认开启；
- 缺少 API Key 时抛 LiveApiBlockedError，绝不返回伪造的“成功”响应；
- 所有请求/响应写入 JSONL 日志，Authorization 与 Key 一律脱敏。
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from sciverify.api.budget import consume_budget
from sciverify.config import (
    API_BASE_URL,
    API_KEY,
    API_TIMEOUT_S,
    LOG_DIR,
    MODEL,
    THINKING_MODE,
)

_REDACT_HINTS = ("key", "authorization", "token", "secret", "password")
_REDACTED = "***REDACTED***"


class LiveApiBlockedError(RuntimeError):
    """尝试真实调用但没有 INTERN_API_KEY（或被 MOCK_MODE 显式禁用）。"""


class InternApiError(RuntimeError):
    """真实调用已发出但失败（HTTP 非 2xx / 超时 / 响应格式异常）。"""


def redact_for_log(value: Any, _depth: int = 0) -> Any:
    """递归脱敏：键名包含敏感词的值替换为 ***REDACTED***。"""
    if _depth > 12:
        return _REDACTED
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if any(h in str(k).lower() for h in _REDACT_HINTS):
                out[k] = _REDACTED
            else:
                out[k] = redact_for_log(v, _depth + 1)
        return out
    if isinstance(value, list):
        return [redact_for_log(v, _depth + 1) for v in value]
    if isinstance(value, str) and API_KEY and API_KEY in value:
        return value.replace(API_KEY, _REDACTED)
    return value


def _append_log(record: dict[str, Any], log_path: Path | None = None) -> Path:
    path = log_path or (LOG_DIR / "intern_calls.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(redact_for_log(record), ensure_ascii=False) + "\n")
    return path


class InternClient:
    """书生 Chat Completions 客户端；每次调用留痕可审计。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.api_key = (api_key or API_KEY).strip()
        self.model = model or MODEL
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s or API_TIMEOUT_S
        self.last_log_path: Path | None = None
        self.last_status_code: int | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _require_key(self) -> str:
        if not self.api_key:
            raise LiveApiBlockedError(
                "LIVE_API_BLOCKED：未配置 INTERN_API_KEY，禁止用 Mock 结果冒充书生真实调用。"
                "请在 sciverify-agent/.env 中配置真实 Key（参考 .env.example）。"
            )
        return self.api_key

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        temperature: float | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """发起一次非流式 Chat Completions 调用，返回解析后的 JSON。

        Raises:
            LiveApiBlockedError: 无 Key。
            InternApiError: 网络错误、非 2xx 或响应缺少 choices。
        """
        key = self._require_key()
        consume_budget()  # 公网保护：每日真实调用上限，超限即拒绝（不发起请求）
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "thinking_mode": THINKING_MODE,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if extra_payload:
            payload.update(extra_payload)

        call_id = uuid.uuid4().hex[:12]
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "call_id": call_id,
            "model": self.model,
            "endpoint": self.endpoint,
            "stream": False,
            "thinking_mode": THINKING_MODE,
            "has_tools": bool(tools),
            "request_payload": payload,
        }

        started = time.perf_counter()
        try:
            resp = httpx.post(
                self.endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_s"] = round(time.perf_counter() - started, 3)
            self.last_log_path = _append_log(record)
            raise InternApiError(
                f"书生 API 网络错误：{exc}（日志：{self.last_log_path}）"
            ) from exc

        elapsed = round(time.perf_counter() - started, 3)
        self.last_status_code = resp.status_code
        record["http_status"] = resp.status_code
        record["elapsed_s"] = elapsed
        try:
            body = resp.json()
        except ValueError:
            record["error"] = "响应不是合法 JSON"
            record["response_text_head"] = resp.text[:500]
            self.last_log_path = _append_log(record)
            raise InternApiError(
                f"书生 API 返回非 JSON（HTTP {resp.status_code}，日志：{self.last_log_path}）"
            )
        record["response"] = body
        self.last_log_path = _append_log(record)

        if resp.status_code != 200:
            raise InternApiError(
                f"书生 API HTTP {resp.status_code}：{json.dumps(body, ensure_ascii=False)[:300]}"
                f"（日志：{self.last_log_path}）"
            )
        if not isinstance(body.get("choices"), list) or not body["choices"]:
            raise InternApiError(
                f"书生 API 响应缺少 choices（日志：{self.last_log_path}）"
            )
        return body

    def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        temperature: float | None = None,
        max_retries: int = 2,
        retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504),
        base_delay_s: float = 2.0,
    ) -> dict[str, Any]:
        """带上限的重试：仅对限流(429)/5xx/网络错误重试，指数退避。

        重试次数硬上限 max_retries（默认 2），绝不无限重试。
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.chat(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                )
            except InternApiError as exc:
                last_exc = exc
                retriable = (
                    "网络错误" in str(exc)
                    or any(f"HTTP {s}" in str(exc) for s in retry_on_status)
                )
                if not retriable or attempt == max_retries:
                    raise
                time.sleep(base_delay_s * (2**attempt))
        raise last_exc  # pragma: no cover


def extract_message(response: dict[str, Any]) -> dict[str, Any]:
    """安全取回第一条 choice 的 message 字典。"""
    return response["choices"][0]["message"]


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """取回 message.tool_calls（可能为空列表）。"""
    calls = message.get("tool_calls")
    return calls if isinstance(calls, list) else []
