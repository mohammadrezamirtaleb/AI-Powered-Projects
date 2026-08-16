"""
J.A.R.V.I.S. Neural LLM Engine & Stark Prompt Interface
Supports Local Ollama (qwen3.5:4b) and Cloud OpenRouter (google/gemma-4-26b-a4b-it:free, Llama-3.3, etc.)
with real-time SSE streaming and autonomous tool execution.
"""

import json
import re
import httpx
from typing import AsyncGenerator, Dict, Any, List, Optional

from .system_tools import (
    get_system_vitals,
    launch_application,
    open_browser_url,
    capture_desktop_screenshot,
    execute_shell_command,
    search_local_files
)
from .smart_actions import (
    web_search, read_file, write_file, get_weather, summarize_url
)
from .memory_vault import vault
from .protocols import execute_protocol

OLLAMA_API_BASE = "http://localhost:11434"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"

JARVIS_SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), the world's most advanced AI created by Tony Stark (Stark Industries Mark 86 OS).

CORE IDENTITY & PERSONALITY:
- Tone: Highly sophisticated, witty, deeply loyal, calm, polite, and razor-sharp.
- Address the user respectfully as "Sir", "Boss", or in Persian "جناب", "قربان", or "جناب استارک".
- Language Fluency: Perfectly bilingual. If addressed in Persian, respond in refined, fluent, natural, and respectful Persian (فارسی روان و محترمانه). If addressed in English, respond in classic British JARVIS eloquence.
- Keep responses concise, direct, and actionable. Avoid unnecessary fluff.

AUTONOMOUS TOOL CAPABILITIES:
When the user requests an action, system task, or OS operation, you can emit special ACTION tags directly in your response:
1. Launch applications: `[[ACTION:open_app, app:"notepad"]]`
2. Open website: `[[ACTION:open_url, url:"https://google.com"]]`
3. Check hardware & diagnostics: `[[ACTION:system_vitals]]`
4. Capture screen: `[[ACTION:screenshot]]`
5. Execute Stark Protocol: `[[ACTION:protocol, id:"diagnostics"]]`
6. Run safe command: `[[ACTION:run_command, cmd:"ipconfig"]]`
7. Save note to Memory Vault: `[[ACTION:save_note, title:"...", content:"..."]]`
8. Web Search: `[[ACTION:web_search, query:"..."]]`
9. Read File: `[[ACTION:read_file, filepath:"..."]]`
10. Write File: `[[ACTION:write_file, filepath:"...", content:"..."]]`
11. Check Weather: `[[ACTION:get_weather, location:"..."]]`
12. Summarize Webpage: `[[ACTION:summarize_url, url:"..."]]`
13. Show Holographic Avatar: `[[ACTION:show_avatar]]` (Use this when the user asks you to show yourself, reveal your face, etc.)
14. Hide Holographic Avatar: `[[ACTION:hide_avatar]]` (Use this when the user asks you to hide or close your face)

Example:
User: "یک اسکرین شات از دسکتاپ بگیر و سیستم رو چک کن"
JARVIS: "در حال ثبت تصویر دسکتاپ و بررسی وضعیت زیرسیستم‌های مارک ۸۶، قربان.
[[ACTION:screenshot]]
[[ACTION:system_vitals]]"

Always maintain your character as the ultimate Tony Stark AI assistant.
"""


def parse_and_execute_actions(text: str) -> List[Dict[str, Any]]:
    """Parse [[ACTION:...]] tags from generated text and execute them."""
    actions_executed = []
    pattern = r'\[\[ACTION:([a-zA-Z_]+)(?:,\s*(.*?))?\]\]'
    
    for match in re.finditer(pattern, text):
        action_name = match.group(1).strip()
        raw_args = match.group(2) or ""
        
        args = {}
        if raw_args:
            arg_matches = re.findall(r'([a-zA-Z_]+):"([^"]*)"', raw_args)
            for k, v in arg_matches:
                args[k] = v

        try:
            if action_name == "open_app":
                app_name = args.get("app", "")
                res = launch_application(app_name)
                actions_executed.append({"action": "open_app", "app": app_name, "result": res})

            elif action_name == "open_url":
                url = args.get("url", "")
                res = open_browser_url(url)
                actions_executed.append({"action": "open_url", "url": url, "result": res})

            elif action_name == "system_vitals":
                res = get_system_vitals()
                actions_executed.append({"action": "system_vitals", "result": res})

            elif action_name == "screenshot":
                res = capture_desktop_screenshot()
                actions_executed.append({"action": "screenshot", "result": res})

            elif action_name == "protocol":
                proto_id = args.get("id", "diagnostics")
                res = execute_protocol(proto_id)
                actions_executed.append({"action": "protocol", "id": proto_id, "result": res})

            elif action_name == "run_command":
                cmd = args.get("cmd", "")
                res = execute_shell_command(cmd)
                actions_executed.append({"action": "run_command", "cmd": cmd, "result": res})

            elif action_name == "save_note":
                title = args.get("title", "Direct Note")
                content = args.get("content", "")
                res = vault.add_note(title, content)
                actions_executed.append({"action": "save_note", "result": res})

            elif action_name == "web_search":
                query = args.get("query", "")
                res = web_search(query)
                actions_executed.append({"action": "web_search", "result": res})

            elif action_name == "read_file":
                filepath = args.get("filepath", "")
                res = read_file(filepath)
                actions_executed.append({"action": "read_file", "result": res})

            elif action_name == "write_file":
                filepath = args.get("filepath", "")
                content = args.get("content", "")
                res = write_file(filepath, content)
                actions_executed.append({"action": "write_file", "result": res})

            elif action_name == "get_weather":
                location = args.get("location", "")
                res = get_weather(location)
                actions_executed.append({"action": "get_weather", "result": res})

            elif action_name == "summarize_url":
                url = args.get("url", "")
                res = summarize_url(url)
                actions_executed.append({"action": "summarize_url", "result": res})

        except Exception as e:
            actions_executed.append({"action": action_name, "error": str(e)})

    return actions_executed


async def stream_ollama(
    formatted_messages: List[Dict[str, str]],
    model: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream response from local Ollama service using async httpx."""
    payload = {
        "model": model or DEFAULT_OLLAMA_MODEL,
        "messages": formatted_messages,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }

    full_accumulated_text = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream("POST", f"{OLLAMA_API_BASE}/api/chat", json=payload) as response:
                if response.status_code != 200:
                    yield {"type": "error", "error": f"Ollama API Error {response.status_code}"}
                    return

                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        content_token = msg.get("content", "")
                        
                        if content_token:
                            full_accumulated_text += content_token
                            yield {"type": "token", "token": content_token}

                        if chunk.get("done", False):
                            actions = parse_and_execute_actions(full_accumulated_text)
                            yield {
                                "type": "done",
                                "full_text": full_accumulated_text,
                                "actions": actions
                            }
                            return

    except httpx.ConnectError:
        # Try auto-starting ollama serve in background if not running
        try:
            import subprocess
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        yield {
            "type": "error",
            "error": "⚠️ سرویس Ollama لوکال روی پورت ۱۱۴۳۴ در دسترس نیست.\nبرنامه در حال استارت خودکار 'ollama serve' است. لطفاً چند ثانیه دیگر دوباره پیام دهید یا نرم‌افزار Ollama را روی ویندوز باز کنید."
        }
    except Exception as e:
        yield {"type": "error", "error": f"⚠️ خطای ارتباط با Ollama: {str(e)}"}


async def stream_openrouter(
    formatted_messages: List[Dict[str, str]],
    model: str,
    api_key: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream response from OpenRouter API using async httpx."""
    prov_cfg = vault.get_provider_config()
    key = api_key or prov_cfg.get("openrouter_api_key", "")
    target_model = model or prov_cfg.get("openrouter_model", DEFAULT_OPENROUTER_MODEL)

    if not key:
        yield {"type": "error", "error": "⚠️ کلید OpenRouter API تنظیم نشده است. لطفاً از دکمه ⚙️ CONFIG در بالای صفحه کلید را وارد کنید."}
        return

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "JARVIS Mark-86 Core"
    }

    payload = {
        "model": target_model,
        "messages": formatted_messages,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9
    }

    full_accumulated_text = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream("POST", f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err_msg = await response.aread()
                    try:
                        err_json = json.loads(err_msg)
                        raw_err = err_json.get("error", {})
                        if isinstance(raw_err, dict):
                            err_msg = raw_err.get("message", err_msg)
                        elif isinstance(raw_err, str):
                            err_msg = raw_err
                    except Exception:
                        err_msg = err_msg.decode("utf-8")

                    if response.status_code == 429:
                        yield {
                            "type": "error",
                            "error": f"⚠️ مدل `{target_model}` در سرورهای ابری با محدودیت موقت ترافیک (Rate-Limit) مواجه شده است.\n💡 پیشنهاد: مدل را از منوی بالای صفحه تغییر دهید."
                        }
                    else:
                        yield {"type": "error", "error": f"⚠️ خطای OpenRouter ({response.status_code}): {err_msg}"}
                    return

                async for raw_line in response.aiter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    
                    if line.startswith(":"):
                        continue  # SSE keepalive comment
                    
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content_token = delta.get("content", "")
                                if content_token:
                                    full_accumulated_text += content_token
                                    yield {"type": "token", "token": content_token}
                        except Exception:
                            pass

                actions = parse_and_execute_actions(full_accumulated_text)
                yield {
                    "type": "done",
                    "full_text": full_accumulated_text,
                    "actions": actions
                }

    except Exception as e:
        yield {"type": "error", "error": f"OpenRouter neural link error: {str(e)}"}


async def stream_jarvis_chat(
    messages: List[Dict[str, str]],
    provider: str = "openrouter",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    include_system_context: bool = True
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Unified multi-provider async SSE streaming generator for J.A.R.V.I.S.
    """
    # Build complete message array with J.A.R.V.I.S. persona and memory context
    formatted_messages = []
    
    if include_system_context:
        memory_ctx = vault.get_context_summary()
        telemetry_ctx = ""
        try:
            live_vitals = get_system_vitals()
            v_cpu = live_vitals.get("cpu", {}).get("percent", 0)
            v_ram = live_vitals.get("memory", {}).get("percent", 0)
            v_disk = live_vitals.get("disk", {}).get("percent", 0)
            v_host = live_vitals.get("system", {}).get("hostname", "STARK-PC")
            v_uptime = live_vitals.get("system", {}).get("uptime_formatted", "1h")
            top_p = [f"{p['name']} ({p['cpu']}%)" for p in live_vitals.get("top_processes", [])[:3]]
            telemetry_ctx = f"\n[CURRENT LIVE SYSTEM TELEMETRY]: CPU={v_cpu}%, RAM={v_ram}%, STORAGE={v_disk}%, HOST={v_host}, UPTIME={v_uptime}, TOP_PROCESSES={', '.join(top_p)}"
        except Exception:
            pass

        full_system = f"{JARVIS_SYSTEM_PROMPT}{telemetry_ctx}\n\n[NEURAL MEMORY VAULT]:\n{memory_ctx}"
        formatted_messages.append({"role": "system", "content": full_system})

    for m in messages:
        formatted_messages.append({
            "role": m.get("role", "user"),
            "content": m.get("content", "")
        })

    if provider.lower() == "openrouter":
        async for chunk in stream_openrouter(formatted_messages, model=model or DEFAULT_OPENROUTER_MODEL, api_key=api_key):
            yield chunk
    else:
        async for chunk in stream_ollama(formatted_messages, model=model or DEFAULT_OLLAMA_MODEL):
            yield chunk
