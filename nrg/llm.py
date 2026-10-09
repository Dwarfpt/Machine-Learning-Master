"""Local LLM via llama-cpp-python with Qwen2.5-native tool calling.

llama-cpp-python's built-in "chatml-function-calling" handler drops tool results (its template has no `tool` role),
so multi-step agents can't use it. Instead the Qwen2.5 function-calling format is rendered explicitly:
  * tool schemas go into the system prompt inside <tools></tools>
  * the model answers with one or more <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks
  * tool results are sent back as a user turn with <tool_response>...</tool_response>
Only plain chat messages are exchanged, so the same code works in-process (llama_cpp.Llama) and against
llama_cpp.server's OpenAI-compatible endpoint (Docker).
"""
from __future__ import annotations

import atexit
import json
import re
import threading

import requests

from . import config

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_lock = threading.Lock()
_llama = None


def available() -> bool:
    if config.LLM_BASE_URL:
        try:
            return requests.get(f"{config.LLM_BASE_URL}/v1/models", timeout=(3, 5)).ok
        except requests.ReadTimeout:  # connected, but /v1/models waits for the lock held by a running generation
            return True
        except requests.RequestException:
            return False
    if not config.LLM_MODEL_PATH.exists():
        return False
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def _local():
    global _llama
    if _llama is None:
        from llama_cpp import Llama
        _llama = Llama(model_path=str(config.LLM_MODEL_PATH), n_ctx=config.LLM_N_CTX, n_threads=config.LLM_THREADS,
                       chat_format="chatml", verbose=False)
        atexit.register(_llama.close)  # free the model before interpreter teardown
    return _llama


def chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.1, repeat_penalty: float = 1.15) -> str:
    params = {"max_tokens": max_tokens, "temperature": temperature, "repeat_penalty": repeat_penalty}
    if config.LLM_BASE_URL:
        r = requests.post(f"{config.LLM_BASE_URL}/v1/chat/completions", timeout=600,
                          json={"messages": messages, **params})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    with _lock:  # one generation at a time on the in-process model
        out = _local().create_chat_completion(messages=messages, **params)
    return out["choices"][0]["message"]["content"] or ""


def tools_system_prompt(system: str, tool_schemas: list[dict]) -> str:
    tools = "\n".join(json.dumps(t, ensure_ascii=False) for t in tool_schemas)
    return (f"{system}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            f"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
            "<args-json-object>}\n</tool_call>")


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for raw in _TOOL_CALL_RE.findall(text or ""):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": obj["name"], "arguments": args})
    return calls


def tool_responses_message(results: list[tuple[str, dict]]) -> dict:
    body = "\n".join(f"<tool_response>\n{json.dumps({'name': n, 'content': r}, ensure_ascii=False, default=str)}"
                     f"\n</tool_response>" for n, r in results)
    return {"role": "user", "content": body}
