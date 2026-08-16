"""
J.A.R.V.I.S. Mark-85 Core Server
FastAPI Server connecting Ollama (qwen3.5:4b, glm-ocr) & OpenRouter (google/gemma-4-26b-a4b-it:free)
to the Holographic Stark HUD UI.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.system_tools import (
    get_system_vitals,
    launch_application,
    open_browser_url,
    capture_desktop_screenshot,
    execute_shell_command,
    search_local_files
)
from core.memory_vault import vault
from core.ocr_engine import extract_ocr_text, analyze_image_with_vision
from core.protocols import PROTOCOLS, execute_protocol
from core.llm_engine import stream_jarvis_chat, OLLAMA_API_BASE, OPENROUTER_API_BASE
from core.conversation_store import init_db, get_all_conversations, get_messages, create_conversation, add_message, delete_conversation
from core.scheduler import scheduler
from core.watcher import init_watcher
from core.tts_engine import generate_tts

app = FastAPI(title="J.A.R.V.I.S. Mark 85 Core API", version="85.4.0")

# Store active notification websockets
notification_websockets = []

async def broadcast_notification(payload: dict):
    for ws in notification_websockets:
        try:
            await ws.send_json(payload)
        except:
            pass

scheduler.add_notification_callback(broadcast_notification)

@app.on_event("startup")
async def startup_event():
    await init_db()
    scheduler.start()
    init_watcher()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.stop()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    provider: Optional[str] = "openrouter" # "ollama" or "openrouter"
    model: Optional[str] = None
    api_key: Optional[str] = None


class OCRRequest(BaseModel):
    image: str
    prompt: Optional[str] = "Extract all text and tabular data accurately."
    model: Optional[str] = "glm-ocr:latest"


class ProtocolRequest(BaseModel):
    protocol_id: str


class CommandRequest(BaseModel):
    command: str


class NoteRequest(BaseModel):
    title: str
    content: str


class ProviderConfigRequest(BaseModel):
    active_provider: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    ollama_model: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    engine: str = "edge"
    voice: str = "fa_IR-Male"


def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except:
        pass


STATIC_FREE_MODELS = [
    {"id": "google/gemma-4-26b-a4b-it:free", "name": "Google: Gemma 4 26B A4B (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "google/gemma-4-31b-it:free", "name": "Google: Gemma 4 31B (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "name": "NVIDIA: Nemotron 3 Ultra 550B (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "name": "NVIDIA: Nemotron 3 Super 120B (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "name": "NVIDIA: Nemotron 3 Nano Omni (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "nvidia/nemotron-3-nano-30b-a3b:free", "name": "NVIDIA: Nemotron 3 Nano 30B (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "nvidia/nemotron-nano-12b-v2-vl:free", "name": "NVIDIA: Nemotron Nano 12B VL (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "nvidia/nemotron-nano-9b-v2:free", "name": "NVIDIA: Nemotron Nano 9B V2 (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "openai/gpt-oss-20b:free", "name": "OpenAI: GPT-OSS 20B (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "poolside/laguna-s-2.1:free", "name": "Poolside: Laguna S 2.1 (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "poolside/laguna-xs-2.1:free", "name": "Poolside: Laguna XS 2.1 (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "cohere/north-mini-code:free", "name": "Cohere: North Mini Code (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "inclusionai/ling-3.0-tiny:free", "name": "inclusionAI: Ling 3.0 Tiny (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"},
    {"id": "nvidia/nemotron-3.5-content-safety:free", "name": "NVIDIA: Nemotron 3.5 Content Safety (Free)", "provider": "openrouter", "badge": "CLOUD // FREE"}
]

_CACHED_MODELS: List[Dict[str, Any]] = []
_LAST_MODELS_FETCH = 0

def fetch_all_free_models() -> List[Dict[str, Any]]:
    global _CACHED_MODELS, _LAST_MODELS_FETCH
    now = time.time()
    if _CACHED_MODELS and (now - _LAST_MODELS_FETCH < 3600):
        return _CACHED_MODELS

    models_list = []
    try:
        import requests
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=6)
        if r.status_code == 200:
            data = r.json().get("data", [])
            for m in data:
                m_id = m.get("id", "")
                if ":free" in m_id:
                    models_list.append({
                        "provider": "openrouter",
                        "id": m_id,
                        "name": m.get("name", m_id),
                        "badge": "CLOUD // FREE",
                        "context_length": m.get("context_length", 0)
                    })
    except Exception:
        pass

    if not models_list:
        models_list = list(STATIC_FREE_MODELS)

    # Always add local Ollama
    models_list.append({
        "provider": "ollama",
        "id": "qwen3.5:4b",
        "name": "Qwen 3.5 4B (Ollama Local)",
        "badge": "LOCAL // OFFLINE"
    })

    _CACHED_MODELS = models_list
    _LAST_MODELS_FETCH = now
    return _CACHED_MODELS


@app.get("/api/status")
def get_status():
    """Return system online status, active provider and models."""
    prov_cfg = vault.get_provider_config()
    return {
        "status": "ONLINE",
        "system": "Stark Industries J.A.R.V.I.S. Mark 85",
        "active_provider": prov_cfg.get("active_provider", "openrouter"),
        "openrouter_model": prov_cfg.get("openrouter_model", "google/gemma-4-26b-a4b-it:free"),
        "ollama_model": prov_cfg.get("ollama_model", "qwen3.5:4b"),
        "ocr_model": "glm-ocr:latest",
        "has_openrouter_key": bool(prov_cfg.get("openrouter_api_key")),
        "timestamp": time.time()
    }


@app.get("/api/models")
def get_models():
    """Get list of all available free models on OpenRouter dynamically + Ollama."""
    return {
        "models": fetch_all_free_models(),
        "config": vault.get_provider_config()
    }


@app.get("/api/config/provider")
def get_provider_config():
    """Get provider configurations."""
    cfg = vault.get_provider_config().copy()
    # Mask API key partially for security in UI display
    key = cfg.get("openrouter_api_key", "")
    if key and len(key) > 12:
        cfg["masked_key"] = key[:8] + "..." + key[-4:]
    else:
        cfg["masked_key"] = key
    return cfg


@app.post("/api/config/provider")
def update_provider_config(req: ProviderConfigRequest):
    """Update active provider, custom models or API key."""
    updates = {}
    if req.active_provider:
        updates["active_provider"] = req.active_provider
    if req.openrouter_api_key:
        updates["openrouter_api_key"] = req.openrouter_api_key
    if req.openrouter_model:
        updates["openrouter_model"] = req.openrouter_model
    if req.ollama_model:
        updates["ollama_model"] = req.ollama_model
    
    vault.update_provider_config(updates)
    return {"success": True, "config": vault.get_provider_config()}


@app.post("/api/tts")
async def api_generate_tts(req: TTSRequest):
    try:
        filepath = await generate_tts(req.text, req.engine, req.voice)
        return FileResponse(
            filepath, 
            media_type="audio/wav" if req.engine == "piper" else "audio/mpeg",
            background=BackgroundTask(cleanup_file, filepath)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/telemetry")
@app.get("/api/vitals")
def get_telemetry():
    """Real-time system vitals endpoint."""
    return get_system_vitals()

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            vitals = get_system_vitals()
            await websocket.send_json(vitals)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    await websocket.accept()
    notification_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in notification_websockets:
            notification_websockets.remove(websocket)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events (SSE) streaming chat endpoint supporting Ollama and OpenRouter."""
    prov_cfg = vault.get_provider_config()
    provider = req.provider or prov_cfg.get("active_provider", "openrouter")
    
    if provider == "openrouter":
        model = req.model or prov_cfg.get("openrouter_model", "google/gemma-4-26b-a4b-it:free")
        api_key = req.api_key or prov_cfg.get("openrouter_api_key")
    else:
        model = req.model or prov_cfg.get("ollama_model", "qwen3.5:4b")
        api_key = None

    async def event_generator():
        async for chunk in stream_jarvis_chat(
            req.messages,
            provider=provider,
            model=model,
            api_key=api_key
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/ocr")
def process_ocr(req: OCRRequest):
    """Process image with GLM-OCR or Qwen Vision."""
    result = extract_ocr_text(req.image, prompt=req.prompt or "", model=req.model or "glm-ocr:latest")
    return result


@app.post("/api/screenshot")
def take_screenshot():
    """Take a full screenshot of current desktop."""
    return capture_desktop_screenshot()


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    prov_cfg = vault.get_provider_config()
    try:
        while True:
            # Wait for user message
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
            except Exception:
                continue

            messages = req.get("messages", [])
            provider = req.get("provider") or prov_cfg.get("active_provider", "openrouter")
            cid = req.get("conversation_id")
            
            if cid:
                # Save the last user message
                last_msg = messages[-1] if messages else None
                if last_msg and last_msg["role"] == "user":
                    await add_message(cid, "user", last_msg["content"])
            
            if provider == "openrouter":
                model = req.get("model") or prov_cfg.get("openrouter_model", "google/gemma-4-26b-a4b-it:free")
                api_key = req.get("api_key") or prov_cfg.get("openrouter_api_key")
            else:
                model = req.get("model") or prov_cfg.get("ollama_model", "qwen3.5:4b")
                api_key = None

            full_text = ""
            async for chunk in stream_jarvis_chat(
                messages,
                provider=provider,
                model=model,
                api_key=api_key
            ):
                if chunk["type"] == "done":
                    full_text = chunk["full_text"]
                await websocket.send_json(chunk)
                
            if cid and full_text:
                await add_message(cid, "assistant", full_text)
                
    except WebSocketDisconnect:
        pass
@app.get("/api/protocols")
def get_protocols():
    """List all pre-configured Stark protocols."""
    return list(PROTOCOLS.values())

@app.get("/api/conversations")
async def api_get_conversations():
    return await get_all_conversations()

@app.post("/api/conversations")
async def api_create_conversation(req: Request):
    data = await req.json()
    title = data.get("title", "New Conversation")
    cid = await create_conversation(title)
    return {"id": cid, "title": title}

@app.get("/api/conversations/{cid}")
async def api_get_conversation(cid: int):
    return await get_messages(cid)

@app.delete("/api/conversations/{cid}")
async def api_delete_conversation(cid: int):
    await delete_conversation(cid)
    return {"success": True}


@app.post("/api/protocols/execute")
@app.post("/api/protocol/execute")
def run_protocol(req: ProtocolRequest):
    """Execute a specific Stark protocol."""
    return execute_protocol(req.protocol_id)


@app.get("/api/vault")
def get_vault():
    """Retrieve full neural memory vault."""
    return vault.get_all()


@app.post("/api/vault/note")
def create_note(req: NoteRequest):
    """Save a new note to the vault."""
    return vault.add_note(req.title, req.content)


@app.delete("/api/vault/note/{note_id}")
def remove_note(note_id: int):
    """Remove a note by ID."""
    success = vault.delete_note(note_id)
    return {"success": success}


@app.post("/api/command")
def run_command(req: CommandRequest, request: Request):
    """Execute a safe shell directive."""
    client_host = request.client.host if request.client else ""
    if client_host not in ["127.0.0.1", "localhost", "::1"]:
        raise HTTPException(status_code=403, detail="Forbidden: Commands can only be executed from localhost.")
    return execute_shell_command(req.command)


# Static assets serving
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print(">>> J.A.R.V.I.S. Mark-85 Core Initializing on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
