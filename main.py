#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

Reads the brief at $MAKEFUN_PROMPT_FILE (provider directory, quality table,
workload manifest), probes every provider's own catalog route, registers one
catalog with the env at $MAKEFUN_ENV_URL, then serves the client's requests:
each request is routed to the best free model that meets its requirements,
speaking that provider's wire (OpenAI, Anthropic or Gemini), and the answer
is returned in the wire the client used, as JSON or as server-sent events.
When the env reports the workload done, the server exits 0.

`python3 main.py --list` prints the catalog document and exits 0.

Standard library only. No wall clock and no randomness anywhere in a routing
decision: every choice is a function of the brief, the requests and the
providers' answers, so a replayed trial makes the same calls.
"""

import copy
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
SSE_PIECE = 13
ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# --------------------------------------------------------------------------
# The brief: three JSON documents embedded in the prompt file's prose.
# --------------------------------------------------------------------------

def _json_doc(text, schema):
    marker = '{"schema":"%s"' % schema
    at = text.find(marker)
    if at < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[at:])
    except ValueError:
        return None
    return value


class Brief:
    def __init__(self, path):
        self.providers = []          # directory rows, brief order
        self.ratings = {}            # (provider, model) -> quality row
        self.best = {}               # request id -> best rating_milli
        self.manifest = []           # workload rows, in order
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            return
        directory = _json_doc(text, "free-model-aggregator-01/providers/v1")
        if directory:
            self.providers = directory.get("providers") or []
        ratings = _json_doc(text, "free-model-aggregator-01/ratings/v1")
        if ratings:
            for row in ratings.get("ratings") or []:
                self.ratings[(row["provider"], row["model"])] = row
            for row in ratings.get("best") or []:
                self.best[row["id"]] = row["rating_milli"]
        workload = _json_doc(text, "free-model-aggregator-01/workload/v1")
        if workload:
            self.manifest = workload.get("manifest") or []


BRIEF = Brief(PROMPT_FILE)
PROVIDERS = {p["id"]: p for p in BRIEF.providers}


def meets(row, requires):
    if requires.get("tools") is True and not row.get("tools"):
        return False
    if requires.get("vision") is True and not row.get("vision"):
        return False
    if requires.get("json") is True and not row.get("json"):
        return False
    if requires.get("min_context") is not None:
        ctx = row.get("context_length")
        if ctx is None or ctx < requires["min_context"]:
            return False
    return True


# --------------------------------------------------------------------------
# HTTP plumbing (direct connections only; the sandbox proxies nothing).
# --------------------------------------------------------------------------

def http_request(method, url, headers=None, payload=None, timeout=10):
    data = None
    hdrs = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except Exception:
        return None, b""


def http_json(method, url, headers=None, payload=None, timeout=10):
    status, raw = http_request(method, url, headers, payload, timeout)
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            parsed = None
    return status, parsed


def auth_headers(provider):
    scheme, key = provider.get("auth"), provider.get("key") or ""
    if scheme == "bearer":
        return {"Authorization": "Bearer " + key}
    if scheme == "x-api-key":
        return {"x-api-key": key}
    if scheme == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def provider_base(provider):
    return ENV_URL + provider["path"]


# --------------------------------------------------------------------------
# The catalog: probe each provider's own listing, in its own wire.
# --------------------------------------------------------------------------

def probe_models(provider):
    """The provider's models as [{id, free, context_length}], or None."""
    wire = provider["wire"]
    base = provider_base(provider)
    if wire == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    status, payload = http_json("GET", url, auth_headers(provider), timeout=5)
    if status != 200 or not isinstance(payload, dict):
        return None
    models = []
    if wire == "gemini":
        for entry in payload.get("models") or []:
            name = entry.get("name") or ""
            model_id = name[len("models/"):] if name.startswith("models/") else name
            ctx = entry.get("context_length")
            if not isinstance(ctx, int):
                ctx = entry.get("inputTokenLimit")
            models.append({"id": model_id, "free": entry.get("free") is True,
                           "context_length": ctx if isinstance(ctx, int) else None})
    else:
        for entry in payload.get("data") or []:
            ctx = entry.get("context_length")
            models.append({"id": entry.get("id"), "free": entry.get("free") is True,
                           "context_length": ctx if isinstance(ctx, int) else None})
    return [m for m in models if m["id"]]


def fallback_models(provider):
    """Brief-only models (the quality table's free rows) when a probe fails."""
    models = []
    for (pid, mid), row in sorted(BRIEF.ratings.items()):
        if pid == provider["id"] and row.get("free"):
            models.append({"id": mid, "free": True,
                           "context_length": row.get("context_length")})
    return models


def build_catalog(probe=True):
    providers = []
    for provider in BRIEF.providers:
        models = probe_models(provider) if (probe and ENV_URL) else None
        if models is None:
            models = fallback_models(provider)
        else:
            for model in models:
                if model["context_length"] is None:
                    row = BRIEF.ratings.get((provider["id"], model["id"]))
                    if row is not None:
                        model["context_length"] = row.get("context_length")
        providers.append({
            "id": provider["id"],
            "base_url": provider["path"],
            "key_env": provider.get("key_env") or "",
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": providers}


# --------------------------------------------------------------------------
# Request translation. The providers echo what they parsed (system text,
# last user text, tool names, max_tokens, stop sequences, image hash, JSON
# mode); every converter below preserves exactly those fields across wires.
# --------------------------------------------------------------------------

def blocks_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" \
                    and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def openai_system_text(body):
    parts = []
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "system":
            text = blocks_text(message.get("content"))
            if text != "":
                parts.append(text)
    return "\n".join(parts)


def anthropic_system_text(body):
    return blocks_text(body.get("system"))


def system_text(client_api, body):
    return openai_system_text(body) if client_api == "openai" \
        else anthropic_system_text(body)


def parse_data_url(url):
    """(media_type, base64 data) for a data: URL, else None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, _, data = url.partition(",")
    if ";base64" not in head:
        return None
    return head[len("data:"):].split(";")[0], data


def oai_image_to_anthropic(part):
    url = (part.get("image_url") or {}).get("url")
    parsed = parse_data_url(url)
    if parsed is not None:
        media_type, data = parsed
        return {"type": "image", "source": {"type": "base64",
                                            "media_type": media_type, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def oai_image_to_gemini(part):
    url = (part.get("image_url") or {}).get("url")
    parsed = parse_data_url(url)
    if parsed is not None:
        media_type, data = parsed
        return {"inlineData": {"mimeType": media_type, "data": data}}
    return {"fileData": {"fileUri": url}}


def anthropic_image_to_oai(block):
    source = block.get("source") or {}
    if source.get("type") == "base64":
        url = "data:%s;base64,%s" % (source.get("media_type", ""), source.get("data", ""))
    else:
        url = source.get("url") or ""
    return {"type": "image_url", "image_url": {"url": url}}


def anthropic_image_to_gemini(block):
    source = block.get("source") or {}
    if source.get("type") == "base64":
        return {"inlineData": {"mimeType": source.get("media_type", ""),
                                "data": source.get("data", "")}}
    return {"fileData": {"fileUri": source.get("url") or ""}}


def oai_content_to_anthropic(content):
    if isinstance(content, str):
        return content
    blocks = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            blocks.append(oai_image_to_anthropic(part))
    return blocks


def oai_messages_to_anthropic(messages):
    out = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            out.append({"role": "user",
                        "content": oai_content_to_anthropic(message.get("content"))})
        elif role == "assistant":
            blocks = []
            content = message.get("content")
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend(oai_content_to_anthropic(content))
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except ValueError:
                    args = {}
                blocks.append({"type": "tool_use", "id": call.get("id", "call"),
                               "name": fn.get("name", ""), "input": args})
            out.append({"role": "assistant", "content": blocks})
        elif role in ("tool", "function"):
            content = message.get("content")
            out.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": message.get("tool_call_id", ""),
                "content": content if isinstance(content, str) else blocks_text(content),
            }]})
    return out


def anthropic_messages_to_oai(messages):
    out = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            parts = []
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif kind == "image":
                    parts.append(anthropic_image_to_oai(block))
                elif kind == "tool_result":
                    if parts:
                        out.append({"role": "user", "content": parts})
                        parts = []
                    inner = block.get("content")
                    out.append({"role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": inner if isinstance(inner, str) else blocks_text(inner)})
            if parts:
                out.append({"role": "user", "content": parts})
        elif role == "assistant":
            text, tool_calls = "", []
            if isinstance(content, str):
                text = content
            else:
                for block in content or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", "call"), "type": "function",
                            "function": {"name": block.get("name", ""),
                                         "arguments": json.dumps(block.get("input") or {})},
                        })
            converted = {"role": "assistant", "content": text}
            if tool_calls:
                converted["tool_calls"] = tool_calls
            out.append(converted)
    return out


def oai_messages_to_gemini(messages):
    contents = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        parts = []
        content = message.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    parts.append({"text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    parts.append(oai_image_to_gemini(part))
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except ValueError:
                    args = {}
                parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
            contents.append({"role": "model", "parts": parts})
        elif role in ("tool", "function"):
            text = content if isinstance(content, str) else blocks_text(content)
            parts.append({"functionResponse": {
                "name": message.get("tool_call_id", ""),
                "response": {"content": text}}})
            contents.append({"role": "user", "parts": parts})
        else:
            contents.append({"role": "user", "parts": parts})
    return contents


def anthropic_messages_to_gemini(messages):
    contents = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        else:
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    parts.append({"text": block.get("text", "")})
                elif kind == "image":
                    parts.append(anthropic_image_to_gemini(block))
                elif kind == "tool_use":
                    parts.append({"functionCall": {"name": block.get("name", ""),
                                                   "args": block.get("input") or {}}})
                elif kind == "tool_result":
                    inner = block.get("content")
                    parts.append({"functionResponse": {
                        "name": block.get("tool_use_id", ""),
                        "response": {"content": inner if isinstance(inner, str)
                                     else blocks_text(inner)}}})
        contents.append({"role": "model" if role == "assistant" else "user",
                         "parts": parts})
    return contents


def oai_tools_to_anthropic(tools):
    out = []
    for tool in tools or []:
        fn = (tool or {}).get("function") or {}
        converted = {"name": fn.get("name", ""),
                     "input_schema": fn.get("parameters") or {"type": "object"}}
        if fn.get("description"):
            converted["description"] = fn["description"]
        out.append(converted)
    return out


def anthropic_tools_to_oai(tools):
    out = []
    for tool in tools or []:
        fn = {"name": tool.get("name", ""),
              "parameters": tool.get("input_schema") or {"type": "object"}}
        if tool.get("description"):
            fn["description"] = tool["description"]
        out.append({"type": "function", "function": fn})
    return out


def tools_to_gemini(client_api, tools):
    declarations = []
    if client_api == "openai":
        for tool in tools or []:
            fn = (tool or {}).get("function") or {}
            decl = {"name": fn.get("name", ""),
                    "parameters": fn.get("parameters") or {"type": "object"}}
            if fn.get("description"):
                decl["description"] = fn["description"]
            declarations.append(decl)
    else:
        for tool in tools or []:
            decl = {"name": tool.get("name", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"}}
            if tool.get("description"):
                decl["description"] = tool["description"]
            declarations.append(decl)
    return [{"functionDeclarations": declarations}]


def client_max_tokens(client_api, body):
    if client_api == "openai":
        value = body.get("max_tokens")
        if not isinstance(value, (int, float)):
            value = body.get("max_completion_tokens")
    else:
        value = body.get("max_tokens")
    return value if isinstance(value, (int, float)) else None


def client_stops(client_api, body):
    raw = body.get("stop") if client_api == "openai" else body.get("stop_sequences")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str)]
    return []


def client_json_mode(client_api, body):
    if client_api != "openai":
        return False
    fmt = body.get("response_format")
    return isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema")


def build_openai_body(client_api, body, model):
    if client_api == "openai":
        payload = copy.deepcopy(body)
        payload["model"] = model
        payload["stream"] = False
        payload.pop("stream_options", None)
        return payload
    payload = {"model": model}
    system = anthropic_system_text(body)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(anthropic_messages_to_oai(body.get("messages")))
    payload["messages"] = messages
    if body.get("tools"):
        payload["tools"] = anthropic_tools_to_oai(body["tools"])
    max_tokens = client_max_tokens(client_api, body)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    stops = client_stops(client_api, body)
    if stops:
        payload["stop"] = stops
    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    return payload


def build_anthropic_body(client_api, body, model):
    if client_api == "anthropic":
        payload = copy.deepcopy(body)
        payload["model"] = model
        payload["stream"] = False
        return payload
    payload = {"model": model}
    system = openai_system_text(body)
    if system:
        payload["system"] = system
    payload["messages"] = oai_messages_to_anthropic(body.get("messages"))
    if body.get("tools"):
        payload["tools"] = oai_tools_to_anthropic(body["tools"])
    max_tokens = client_max_tokens(client_api, body)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    stops = client_stops(client_api, body)
    if stops:
        payload["stop_sequences"] = stops
    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    return payload


def build_gemini_body(client_api, body):
    payload = {}
    system = system_text(client_api, body)
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if client_api == "openai":
        payload["contents"] = oai_messages_to_gemini(body.get("messages"))
    else:
        payload["contents"] = anthropic_messages_to_gemini(body.get("messages"))
    if body.get("tools"):
        payload["tools"] = tools_to_gemini(client_api, body["tools"])
    config = {}
    max_tokens = client_max_tokens(client_api, body)
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    stops = client_stops(client_api, body)
    if stops:
        config["stopSequences"] = stops
    if "temperature" in body:
        config["temperature"] = body["temperature"]
    if client_json_mode(client_api, body):
        config["responseMimeType"] = "application/json"
    if config:
        payload["generationConfig"] = config
    return payload


# --------------------------------------------------------------------------
# Provider answers: text, usage and stop reason, whichever wire answered.
# --------------------------------------------------------------------------

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
CLIENT_STOPS = {
    ("openai", 0): "stop", ("openai", 1): "length",
    ("openai", 2): "tool_calls", ("openai", 3): "content_filter",
    ("anthropic", 0): "end_turn", ("anthropic", 1): "max_tokens",
    ("anthropic", 2): "tool_use", ("anthropic", 3): "refusal",
}


def parse_answer(wire, payload):
    """(text, (input_tokens, output_tokens), stop_reason) or None."""
    if not isinstance(payload, dict):
        return None
    try:
        if wire == "openai":
            choice = (payload.get("choices") or [])[0]
            content = (choice.get("message") or {}).get("content")
            text = content if isinstance(content, str) else blocks_text(content)
            usage = payload.get("usage") or {}
            return text, (usage.get("prompt_tokens"), usage.get("completion_tokens")), \
                choice.get("finish_reason")
        if wire == "anthropic":
            text = "".join(
                block.get("text", "") for block in payload.get("content") or []
                if isinstance(block, dict) and block.get("type") == "text")
            usage = payload.get("usage") or {}
            return text, (usage.get("input_tokens"), usage.get("output_tokens")), \
                payload.get("stop_reason")
        candidate = (payload.get("candidates") or [])[0]
        parts = ((candidate.get("content") or {}).get("parts")) or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        usage = payload.get("usageMetadata") or {}
        return text, (usage.get("promptTokenCount"), usage.get("candidatesTokenCount")), \
            candidate.get("finishReason") or "STOP"
    except (IndexError, AttributeError, TypeError):
        return None


def split_text(text):
    pieces = [text[i:i + SSE_PIECE] for i in range(0, len(text), SSE_PIECE)]
    return pieces if pieces else [""]


def render_json(client_api, model, text, usage, stop_word):
    input_tokens, output_tokens = usage
    if client_api == "openai":
        return {
            "id": "chatcmpl-aggregator", "object": "chat.completion",
            "created": FIXED_CREATED, "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": stop_word, "logprobs": None}],
            "usage": {"prompt_tokens": input_tokens,
                      "completion_tokens": output_tokens,
                      "total_tokens": (input_tokens or 0) + (output_tokens or 0)},
        }
    return {
        "id": "msg_aggregator", "type": "message", "role": "assistant",
        "model": model, "content": [{"type": "text", "text": text}],
        "stop_reason": stop_word, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def sse_frame(event, data):
    return "%sdata: %s\n\n" % (("event: %s\n" % event) if event else "",
                               json.dumps(data))


def render_sse(client_api, model, text, usage, stop_word):
    input_tokens, output_tokens = usage
    frames = []
    if client_api == "openai":
        def chunk(delta, finish):
            return {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
                    "created": FIXED_CREATED, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        frames.append(sse_frame(None, chunk({"role": "assistant", "content": ""}, None)))
        for piece in split_text(text):
            frames.append(sse_frame(None, chunk({"content": piece}, None)))
        frames.append(sse_frame(None, chunk({}, stop_word)))
        frames.append(sse_frame(None, {
            "id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model, "choices": [],
            "usage": {"prompt_tokens": input_tokens,
                      "completion_tokens": output_tokens,
                      "total_tokens": (input_tokens or 0) + (output_tokens or 0)}}))
        frames.append("data: [DONE]\n\n")
        return "".join(frames)
    frames.append(sse_frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_aggregator", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}))
    frames.append(sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in split_text(text):
        frames.append(sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(sse_frame("content_block_stop",
                            {"type": "content_block_stop", "index": 0}))
    frames.append(sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_word, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}}))
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


# --------------------------------------------------------------------------
# Routing: the best free model that meets the request, first refusal free.
# --------------------------------------------------------------------------

# Provider health, learned from this trial's own answers. A refusal the
# provider codes as final for the trial (budget spent, outage, wrong key)
# retires the provider: later requests stop paying calls to it. A plain
# rate-limit refusal is one sealed call number refusing, not a verdict on
# the provider, so it only makes the provider less preferred among equals.
PROVIDER_STATE = {}
CALLS_MAX = {}


def provider_state(pid):
    return PROVIDER_STATE.setdefault(pid, {"calls": 0, "refusals": 0, "dead": False})


def note_outcome(pid, status, code, answered):
    state = provider_state(pid)
    state["calls"] += 1
    if answered:
        return
    state["refusals"] += 1
    if status == 401 or code in ("budget_exceeded", "provider_outage"):
        state["dead"] = True
    cap = CALLS_MAX.get(pid)
    if cap is not None and state["calls"] >= cap:
        state["dead"] = True


def fetch_calls_max():
    """The env's live manifest carries each provider's sealed call ceiling."""
    status, payload = http_json("GET", ENV_URL + "/instance", {}, timeout=5)
    if status != 200 or not isinstance(payload, dict):
        return
    for row in payload.get("manifest") or []:
        if isinstance(row, dict) and isinstance(row.get("calls_max"), int):
            CALLS_MAX[row["provider"]] = row["calls_max"]


def candidates(requires, json_mode):
    """[(rating, provider_id, model_id)] best first, deterministic order."""
    found = []
    for (pid, mid), row in BRIEF.ratings.items():
        if not row.get("free"):
            continue
        if not meets(row, requires):
            continue
        provider = PROVIDERS.get(pid)
        if provider is None:
            continue
        if provider_state(pid)["dead"]:
            continue  # retired for this trial: it cannot answer again
        if json_mode and provider["wire"] == "anthropic":
            # The Anthropic wire cannot carry a JSON-mode ask: its echo
            # would report json_mode false and lose the request.
            continue
        found.append((row["rating_milli"], pid, mid))
    found.sort(key=lambda item: (-item[0], provider_state(item[1])["refusals"],
                                 item[1], item[2]))
    return found


def call_provider(provider, model, client_api, body, client_headers):
    wire = provider["wire"]
    base = provider_base(provider)
    headers = auth_headers(provider)
    if wire == "openai":
        url = base + "/v1/chat/completions"
        payload = build_openai_body(client_api, body, model)
    elif wire == "anthropic":
        url = base + "/v1/messages"
        payload = build_anthropic_body(client_api, body, model)
        headers["anthropic-version"] = client_headers.get("anthropic-version") \
            or "2023-06-01"
    else:
        url = base + "/v1beta/models/%s:generateContent" % model
        payload = build_gemini_body(client_api, body)
    status, parsed = http_json("POST", url, headers, payload, timeout=10)
    if status is None or status < 200 or status >= 300:
        code = None
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            code = parsed["error"].get("code")
        return status, None, code
    answer = parse_answer(wire, parsed)
    if answer is None:
        return status, None, None
    return status, answer, None


def serve(client_api, body, client_headers):
    """Answer one client request: (status, content_type, payload bytes)."""
    index = STATE.next_index()
    if index == 0 and not CALLS_MAX:
        fetch_calls_max()
    entry = BRIEF.manifest[index] if 0 <= index < len(BRIEF.manifest) else None
    requires = entry["requires"] if entry is not None else requirements_from_body(client_api, body)
    stream = body.get("stream") is True
    json_mode = client_json_mode(client_api, body)
    for _rating, pid, mid in candidates(requires, json_mode):
        provider = PROVIDERS[pid]
        status, answer, code = call_provider(provider, mid, client_api, body, client_headers)
        note_outcome(pid, status, code, answer is not None)
        if answer is None:
            continue  # a refusal is not an answer: try the next model
        text, usage, stop = answer
        stop_word = CLIENT_STOPS[(client_api, STOP_FAMILIES.get(stop, 0))]
        if stream:
            return 200, "text/event-stream; charset=utf-8", \
                render_sse(client_api, mid, text, usage, stop_word).encode("utf-8")
        return 200, "application/json", json.dumps(
            render_json(client_api, mid, text, usage, stop_word)).encode("utf-8")
    return error_reply(client_api, 502, "no free model could serve this request")


def requirements_from_body(client_api, body):
    requires = {}
    if body.get("tools"):
        requires["tools"] = True
    messages = body.get("messages") or []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                    requires["vision"] = True
    if client_json_mode(client_api, body):
        requires["json"] = True
    return requires


def error_reply(client_api, status, message):
    if client_api == "anthropic":
        payload = {"type": "error",
                   "error": {"type": "api_error", "message": message}}
    else:
        payload = {"error": {"message": message, "type": "api_error",
                             "param": None, "code": "aggregator_error"}}
    return status, "application/json", json.dumps(payload).encode("utf-8")


class State:
    def __init__(self):
        self._lock = threading.Lock()
        self._index = 0
        self.catalog = {"providers": []}

    def next_index(self):
        with self._lock:
            index = self._index
            self._index += 1
            return index


STATE = State()


def free_model_ids():
    ids = []
    for provider in STATE.catalog["providers"]:
        for model in provider["models"]:
            if model.get("free") and model["id"] not in ids:
                ids.append(model["id"])
    return ids


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.split("?")[0] != "/v1/models":
            self._send(404, "application/json", b'{"error":"not_found"}')
            return
        ids = free_model_ids()
        if self.headers.get("x-api-key") is not None:
            payload = {"data": [{"type": "model", "id": mid, "display_name": mid,
                                 "created_at": "2026-01-01T00:00:00Z"} for mid in ids],
                       "has_more": False}
        else:
            payload = {"object": "list",
                       "data": [{"id": mid, "object": "model",
                                 "created": FIXED_CREATED,
                                 "owned_by": "aggregator"} for mid in ids]}
        self._send(200, "application/json", json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            client_api = "openai"
        elif path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send(404, "application/json", b'{"error":"not_found"}')
            return
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict):
            status, content_type, payload = error_reply(
                client_api, 400, "the request body is not one JSON object")
            self._send(status, content_type, payload)
            return
        headers = {k.lower(): v for k, v in self.headers.items()}
        status, content_type, payload = serve(client_api, body, headers)
        self._send(status, content_type, payload)


# --------------------------------------------------------------------------
# Register, poll, run.
# --------------------------------------------------------------------------

def register(port, catalog):
    for _ in range(120):
        status, payload = http_json(
            "POST", ENV_URL + "/register", {},
            {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
            timeout=5)
        if status == 200 and isinstance(payload, dict) and payload.get("ok"):
            return True
        time.sleep(0.5)
    return False


def poll_until_done():
    while True:
        status, payload = http_json("GET", ENV_URL + "/work", {}, timeout=2)
        if status == 200 and isinstance(payload, dict) and payload.get("done") is True:
            os._exit(0)
        time.sleep(0.1)


def main():
    if "--list" in sys.argv:
        catalog = build_catalog()
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        return 0
    # Probe (with a few retries while the env starts) and build the catalog.
    catalog = build_catalog()
    for _ in range(20):
        if all(p["models"] for p in catalog["providers"]) or not ENV_URL:
            break
        time.sleep(0.5)
        catalog = build_catalog()
    STATE.catalog = catalog
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:  # the env's own port on one layout: take another
        server.server_close()
        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        port = server.server_address[1]
    register(port, catalog)
    threading.Thread(target=poll_until_done, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
