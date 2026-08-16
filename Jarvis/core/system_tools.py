"""
J.A.R.V.I.S. System Tools & Telemetry Engine
Author: Stark Industries Mark-85 AI Core
"""

import os
import sys
import time
import subprocess
import webbrowser
import platform
import psutil
import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image, ImageGrab

SCREENSHOTS_DIR = Path("data/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

APP_MAP = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "cmd": "cmd.exe",
    "terminal": "wt.exe",
    "powershell": "powershell.exe",
    "explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "taskmgr": "taskmgr.exe",
    "paint": "mspaint.exe",
    "settings": "start ms-settings:",
    "vscode": "code",
    "code": "code",
    "browser": "start chrome || start msedge",
    "chrome": "start chrome",
    "edge": "start msedge",
    "spotify": "start spotify:",
}

def get_system_vitals() -> Dict[str, Any]:
    """Gather real-time hardware telemetry and load statistics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_current = round(cpu_freq.current, 0) if cpu_freq else 0

        # Memory
        mem = psutil.virtual_memory()
        mem_total_gb = round(mem.total / (1024 ** 3), 1)
        mem_used_gb = round(mem.used / (1024 ** 3), 1)
        mem_percent = mem.percent

        # Disk
        disk = psutil.disk_usage('/')
        disk_total_gb = round(disk.total / (1024 ** 3), 1)
        disk_used_gb = round(disk.used / (1024 ** 3), 1)
        disk_percent = disk.percent

        # Battery
        battery = psutil.sensors_battery()
        battery_info = {
            "percent": round(battery.percent, 1) if battery else 100,
            "power_plugged": battery.power_plugged if battery else True,
            "has_battery": battery is not None
        }

        # Network IO
        net_io = psutil.net_io_counters()
        net_info = {
            "bytes_sent_mb": round(net_io.bytes_sent / (1024 ** 2), 2),
            "bytes_recv_mb": round(net_io.bytes_recv / (1024 ** 2), 2),
        }

        # Top processes by memory/cpu
        top_procs = []
        try:
            for p in sorted(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']),
                            key=lambda x: x.info.get('memory_percent') or 0, reverse=True)[:5]:
                top_procs.append({
                    "name": p.info.get('name', 'unknown'),
                    "pid": p.info.get('pid', 0),
                    "cpu": round(p.info.get('cpu_percent') or 0, 1),
                    "mem": round(p.info.get('memory_percent') or 0, 1),
                })
        except Exception:
            pass

        return {
            "status": "ONLINE",
            "hostname": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "uptime_seconds": int(time.time() - psutil.boot_time()),
            "cpu": {
                "percent": cpu_percent,
                "cores": cpu_count,
                "frequency_mhz": cpu_freq_current
            },
            "memory": {
                "total_gb": mem_total_gb,
                "used_gb": mem_used_gb,
                "percent": mem_percent
            },
            "disk": {
                "total_gb": disk_total_gb,
                "used_gb": disk_used_gb,
                "percent": disk_percent
            },
            "battery": battery_info,
            "network": net_info,
            "top_processes": top_procs,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "cpu": {"percent": 0},
            "memory": {"percent": 0},
            "disk": {"percent": 0},
            "battery": {"percent": 100, "power_plugged": True}
        }


def launch_application(app_query: str) -> Dict[str, Any]:
    """Launch Windows apps or custom commands safely."""
    clean_name = app_query.strip().lower()
    command = APP_MAP.get(clean_name, clean_name)

    try:
        if command.startswith("start "):
            subprocess.Popen(command, shell=True)
        else:
            subprocess.Popen(command, shell=True)
        return {
            "success": True,
            "message": f"Successfully launched {app_query}",
            "command": command
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to launch {app_query}: {str(e)}"
        }


def open_browser_url(url: str) -> Dict[str, Any]:
    """Open specified web URL in default browser."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    try:
        webbrowser.open(url)
        return {"success": True, "url": url, "message": f"Opened {url} in browser"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _grab_win32_screen() -> Image.Image:
    """Capture full Windows desktop using native GDI BitBlt."""
    import ctypes
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

    w = user32.GetSystemMetrics(0)
    h = user32.GetSystemMetrics(1)

    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    hbm = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
    old_bm = gdi32.SelectObject(hdc_mem, hbm)

    gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020)

    bmi = bytearray(40)
    bmi[0:4] = (40).to_bytes(4, 'little')
    bmi[4:8] = w.to_bytes(4, 'little', signed=True)
    bmi[8:12] = (-h).to_bytes(4, 'little', signed=True)
    bmi[12:14] = (1).to_bytes(2, 'little')
    bmi[14:16] = (32).to_bytes(2, 'little')

    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, bytes(bmi), 0)

    gdi32.SelectObject(hdc_mem, old_bm)
    gdi32.DeleteObject(hbm)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(0, hdc_screen)

    return Image.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1).convert('RGB')


def capture_desktop_screenshot() -> Dict[str, Any]:
    """Take a full screenshot, return base64 and saved path."""
    try:
        img = None
        try:
            img = ImageGrab.grab()
        except Exception:
            pass

        if img is None:
            img = _grab_win32_screen()

        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.png"
        filepath = SCREENSHOTS_DIR / filename
        img.save(filepath, format="PNG")

        # Create base64 preview
        buf = io.BytesIO()
        img.thumbnail((1280, 720))
        img.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "success": True,
            "filepath": str(filepath.resolve()),
            "filename": filename,
            "base64_data": f"data:image/jpeg;base64,{b64_str}",
            "message": f"Screenshot captured: {filename}"
        }
    except Exception as e:
        return {"success": False, "error": f"Screenshot capture failed: {str(e)}"}


def execute_shell_command(command: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute a safe shell command and return stdout/stderr."""
    from core.security import validate_command
    
    if not validate_command(command):
        return {"success": False, "error": f"Directive rejected: Command '{command}' is not in the allowed security whitelist."}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Directive execution timed out after {timeout} seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_local_files(query: str, root_dir: str = ".") -> Dict[str, Any]:
    """Search for files matching query within directory tree safely."""
    matches = []
    try:
        # Prevent path traversal
        base = Path(root_dir).resolve()
        workspace_root = Path(".").resolve()
        
        if not str(base).startswith(str(workspace_root)):
            return {"success": False, "error": "Access denied: Path traversal outside workspace is restricted."}

        for p in base.rglob(f"*{query}*"):
            if not any(part.startswith((".", "node_modules", "__pycache__", "venv")) for part in p.parts):
                matches.append({
                    "path": str(p),
                    "name": p.name,
                    "is_dir": p.is_dir(),
                    "size_bytes": p.stat().st_size if p.is_file() else 0
                })
                if len(matches) >= 25:
                    break
        return {"success": True, "count": len(matches), "files": matches}
    except Exception as e:
        return {"success": False, "error": str(e)}
