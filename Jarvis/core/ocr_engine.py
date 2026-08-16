"""
J.A.R.V.I.S. Multi-Model OCR & Vision Engine
Supports: GLM-OCR & Qwen3.5 Vision via Local Ollama
"""

import base64
import io
import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

OLLAMA_API_BASE = "http://localhost:11434"
DEFAULT_OCR_MODEL = "glm-ocr:latest"
FALLBACK_VISION_MODEL = "qwen3.5:4b"


def _prepare_image_base64(image_input: str) -> str:
    """Accepts file path, base64 data URI, or raw base64 and returns clean base64 string."""
    if image_input.startswith("data:image"):
        # Strip header like data:image/png;base64,...
        image_input = image_input.split(",", 1)[1]
        return image_input

    path = Path(image_input)
    if path.exists() and path.is_file():
        with Image.open(path) as img:
            # Resize if overly large to optimize inference speed
            if max(img.size) > 2048:
                img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img_format = "PNG" if img.mode == "RGBA" else "JPEG"
            img.save(buf, format=img_format, quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    return image_input


def _sanitize_ocr_output(raw_text: str) -> str:
    """Clean repeating backticks, excessive blank lines, or token artifacts."""
    if not raw_text:
        return ""

    # Remove endless repeating backticks e.g. ```\n```\n```
    cleaned = re.sub(r'(```\s*){3,}', '```', raw_text)
    
    # If the text has duplicate repeating blocks, take the cleanest representation
    lines = [line.rstrip() for line in cleaned.splitlines()]
    dedup_lines = []
    prev_line = None
    rep_count = 0
    for line in lines:
        if line == prev_line:
            rep_count += 1
            if rep_count < 2:
                dedup_lines.append(line)
        else:
            rep_count = 0
            dedup_lines.append(line)
            prev_line = line

    result = "\n".join(dedup_lines).strip()
    return result


def extract_ocr_text(
    image_input: str,
    prompt: str = "Extract all text, code, formulas, and tabular data accurately in Markdown.",
    model: str = DEFAULT_OCR_MODEL
) -> Dict[str, Any]:
    """
    Perform deep optical character recognition using Ollama GLM-OCR or Qwen Vision.
    """
    try:
        b64_img = _prepare_image_base64(image_input)
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64_img],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            raw_response = response.json().get("response", "")
            sanitized = _sanitize_ocr_output(raw_response)
            
            # If GLM-OCR produced empty output or errored, attempt with Qwen Vision
            if not sanitized.strip() and model != FALLBACK_VISION_MODEL:
                return extract_ocr_text(image_input, prompt, model=FALLBACK_VISION_MODEL)

            return {
                "success": True,
                "model_used": model,
                "text": sanitized,
                "raw": raw_response
            }
        else:
            # Fallback if primary model fails
            if model != FALLBACK_VISION_MODEL:
                return extract_ocr_text(image_input, prompt, model=FALLBACK_VISION_MODEL)
            return {
                "success": False,
                "error": f"Ollama API returned HTTP {response.status_code}: {response.text}"
            }

    except Exception as e:
        # Fallback to Qwen3.5
        if model != FALLBACK_VISION_MODEL:
            try:
                return extract_ocr_text(image_input, prompt, model=FALLBACK_VISION_MODEL)
            except Exception:
                pass
        return {
            "success": False,
            "error": f"OCR extraction failed: {str(e)}"
        }


def analyze_image_with_vision(
    image_input: str,
    user_prompt: str = "Analyze this image and explain its key elements in detail as J.A.R.V.I.S."
) -> Dict[str, Any]:
    """Analyze image visually using Qwen3.5-4b's multimodal capabilities."""
    return extract_ocr_text(image_input, prompt=user_prompt, model=FALLBACK_VISION_MODEL)
