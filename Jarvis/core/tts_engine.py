import os
import io
import asyncio
import uuid
from pathlib import Path
import urllib.request
import edge_tts

# Constants
DATA_DIR = Path("data/piper_models")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Piper Models Info (from HuggingFace/piper)
PIPER_MODELS = {
    "en_US-lessac-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true"
    },
    "fa_IR-amir-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx?download=true",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx.json?download=true"
    }
}

async def download_piper_model(voice_id: str):
    if voice_id not in PIPER_MODELS:
        raise ValueError(f"Unknown Piper voice ID: {voice_id}")
    
    onnx_path = DATA_DIR / f"{voice_id}.onnx"
    json_path = DATA_DIR / f"{voice_id}.onnx.json"
    
    # Download ONNX if missing
    if not onnx_path.exists():
        print(f"Downloading Piper ONNX model: {voice_id}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            lambda: urllib.request.urlretrieve(PIPER_MODELS[voice_id]["onnx"], onnx_path)
        )
        
    # Download JSON if missing
    if not json_path.exists():
        print(f"Downloading Piper JSON config: {voice_id}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: urllib.request.urlretrieve(PIPER_MODELS[voice_id]["json"], json_path)
        )
    return onnx_path, json_path

async def generate_edge_tts(text: str, voice: str) -> str:
    """Generates Edge TTS audio and returns the filepath."""
    temp_filename = f"temp_{uuid.uuid4().hex}.mp3"
    temp_path = DATA_DIR / temp_filename
    
    # Common mappings
    if voice == "en_US-Male":
        edge_voice = "en-US-ChristopherNeural"
    elif voice == "en_US-Female":
        edge_voice = "en-US-AriaNeural"
    elif voice == "fa_IR-Male":
        edge_voice = "fa-IR-FaridNeural"
    elif voice == "fa_IR-Female":
        edge_voice = "fa-IR-DilaraNeural"
    else:
        edge_voice = voice

    communicate = edge_tts.Communicate(text, edge_voice)
    await communicate.save(str(temp_path))
    return str(temp_path)

async def generate_piper_tts(text: str, voice: str) -> str:
    """Generates Piper TTS audio and returns the filepath."""
    if voice == "en_US-Male":
        model_id = "en_US-lessac-medium"
    elif voice == "fa_IR-Male":
        model_id = "fa_IR-amir-medium"
    else:
        model_id = "en_US-lessac-medium"

    onnx_path, json_path = await download_piper_model(model_id)
    
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    temp_path = DATA_DIR / temp_filename
    
    text_path = DATA_DIR / f"temp_{uuid.uuid4().hex}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    # Run piper binary with CUDA enabled
    cmd = f"venv\\Scripts\\piper.exe --cuda -m {onnx_path} -c {json_path} -f {temp_path} -i {text_path}"
    
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()
    
    # Cleanup text file
    if text_path.exists():
        text_path.unlink()
        
    return str(temp_path)

async def generate_tts(text: str, engine: str, voice: str) -> str:
    if engine == "edge":
        return await generate_edge_tts(text, voice)
    elif engine == "piper":
        return await generate_piper_tts(text, voice)
    else:
        raise ValueError(f"Unknown TTS Engine: {engine}")
