# 🛡️ J.A.R.V.I.S. (Just A Rather Very Intelligent System)

<div align="center">
  <img src="https://img.shields.io/badge/Status-Online-success?style=for-the-badge" alt="Status Online">
  <img src="https://img.shields.io/badge/Version-Mark_86-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/Powered_by-FastAPI_%7C_Three.js-orange?style=for-the-badge" alt="Powered By">
</div>

<br>

Welcome to the **J.A.R.V.I.S.** AI Assistant project. This is a highly advanced, fully autonomous personal AI assistant modeled after Tony Stark's iconic J.A.R.V.I.S., complete with a stunning Cybernetic HUD, 3D Holographic Avatar, and real-time voice synthesis.

---

## ✨ Features

- **🌐 Holographic 3D Avatar:** Features a stunning 3D particle face powered by `Three.js` that organically forms from "powder", glows, and actively follows your mouse movements.
- **🧠 Hybrid AI Engine:** Seamlessly switches between Cloud AI (OpenRouter / Gemma / Llama) and Local Offline AI (Ollama - Qwen/Llama3) for ultimate privacy and speed.
- **🗣️ Voice Synthesis (TTS):** Integrated with Microsoft Edge TTS and local **Piper TTS** with GPU (CUDA) acceleration for natural, instantaneous speech in both English and Persian (فارسی).
- **👁️ Computer Vision (OCR):** "Vision Lab" capable of analyzing screenshots, extracting text, and understanding what's on your screen.
- **⚙️ Autonomous System Control:** JARVIS can autonomously execute commands, launch applications, open websites, search the web, and check system vitals.
- **💾 Neural Memory Vault:** Securely stores configurations, learned facts, and system notes with local Fernet encryption.
- **⚡ Real-time SSE Streaming:** Extremely fast token streaming directly into the futuristic terminal UI.

---

## 🛠️ Architecture

### Backend (Python/FastAPI)
- `server.py`: The core FastAPI server handling SSE streaming, TTS pipelines, and WebSockets.
- `core/llm_engine.py`: Manages the prompt engineering and neural routing between Ollama and OpenRouter.
- `core/tts_engine.py`: Handles audio generation using Piper and Edge TTS.
- `core/memory_vault.py`: Local database/JSON store for persistent memory and settings.
- `core/system_tools.py`: OS-level integrations for launching apps, capturing screenshots, and running shell commands safely.

### Frontend (HTML/JS/CSS)
- `static/index.html`: The master Stark Industries HUD interface.
- `static/js/app.js`: Core client logic, UI binding, Markdown rendering, and WebSocket management.
- `static/js/avatar.js`: The `Three.js` WebGL engine rendering the 3D particle hologram.
- `static/js/audio_synth.js`: Manages audio queues and Web Speech API fallbacks.
- `static/css/reactor.css & hud_core.css`: The beautiful neon-cyan Stark UI aesthetic.

---

## 🚀 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mohammadrezamirtaleb/AI-Powered-Projects.git
   cd AI-Powered-Projects/Jarvis
   ```

2. **Create a Virtual Environment & Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **(Optional) Setup Piper TTS for Offline Voice**
   Download the Piper TTS binary and place it in the `venv/Scripts/` folder, along with your preferred ONNX voice models in the `data/piper_models/` directory.

4. **Launch J.A.R.V.I.S.**
   ```bash
   python run.py
   ```
   *The system will automatically open your default browser to `http://127.0.0.1:8000`.*

---

## 🔒 Security Note
This repository contains the **raw source code without API keys**. 
When you first boot JARVIS, you can securely enter your API keys (e.g., OpenRouter) via the built-in configuration UI (`⚙️ CONFIG`). Keys are stored locally and symmetrically encrypted in your `data/memory_vault.json` file.

---
*Created as part of the AI-Powered-Projects initiative. "Sometimes you gotta run before you can walk."*
