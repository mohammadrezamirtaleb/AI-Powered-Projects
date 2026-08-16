"""
J.A.R.V.I.S. Mark-85 Master Launcher
Auto-checks local Ollama service and launches FastAPI HUD Server
"""

import sys
import time
import webbrowser
import threading
import subprocess
import requests
import uvicorn


def ensure_ollama_running():
    """Ensure local Ollama service is active in background."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            print(">>> [OLLAMA]: Local Ollama service is active & responding.")
            return
    except Exception:
        pass

    print(">>> [OLLAMA]: Starting background Ollama daemon...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
    except Exception as e:
        print(f">>> [OLLAMA WARNING]: Could not auto-start ollama serve: {e}")


def open_browser():
    """Launch Stark HUD in true fullscreen Kiosk mode (zero borders/tabs)."""
    import os
    import subprocess
    time.sleep(1.2)
    url = "http://127.0.0.1:8000"
    print(f">>> [HUD LAUNCHER]: Initializing Fullscreen Stark Terminal at {url} ...")

    # Browser paths for Windows Kiosk Mode
    edge_paths = [
        os.path.expandvars(r'%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe'),
        os.path.expandvars(r'%ProgramFiles%\Microsoft\Edge\Application\msedge.exe'),
        os.path.expandvars(r'%LocalAppData%\Microsoft\Edge\Application\msedge.exe'),
    ]
    chrome_paths = [
        os.path.expandvars(r'%ProgramFiles%\Google\Chrome\Application\chrome.exe'),
        os.path.expandvars(r'%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe'),
        os.path.expandvars(r'%LocalAppData%\Google\Chrome\Application\chrome.exe'),
    ]

    kiosk_flags = [
        f"--app={url}",
        "--start-fullscreen",
        "--kiosk",
        "--no-first-run",
        "--disable-session-crashed-bubble",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--enable-gpu-rasterization",
        "--enable-accelerated-2d-canvas"
    ]

    for edge_exe in edge_paths:
        if os.path.exists(edge_exe):
            try:
                subprocess.Popen([
                    edge_exe,
                    "--edge-kiosk-type=fullscreen",
                    *kiosk_flags
                ])
                return
            except Exception:
                pass

    for chrome_exe in chrome_paths:
        if os.path.exists(chrome_exe):
            try:
                subprocess.Popen([
                    chrome_exe,
                    *kiosk_flags
                ])
                return
            except Exception:
                pass

    # Fallback to default browser
    webbrowser.open(url)


if __name__ == "__main__":
    print("==================================================================")
    print("   J.A.R.V.I.S. MARK-85 AI ASSISTANT // STARK INDUSTRIES")
    print("   Multi-Provider: OpenRouter (Cloud) & Ollama (Local)")
    print("==================================================================")

    # 1. Check Ollama
    ensure_ollama_running()

    # 2. Launch browser in separate thread
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Run FastAPI server
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
