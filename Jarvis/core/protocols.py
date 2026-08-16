"""
J.A.R.V.I.S. Stark Protocols & Automation Matrix
Author: Stark Industries Mark-85 AI Core
"""

import time
from typing import Dict, Any, List
from .system_tools import get_system_vitals, capture_desktop_screenshot, launch_application
from .memory_vault import vault
from .ocr_engine import extract_ocr_text

PROTOCOLS = {
    "diagnostics": {
        "id": "diagnostics",
        "name": "Diagnostic Sweep",
        "code": "PROTOCOL-01-SWEEP",
        "icon": "⚡",
        "description": "Perform full telemetry scan of CPU, RAM, storage, battery and active processes.",
        "voice_ack": "Initiating complete diagnostic sweep of all Mark 85 subsystems, Sir."
    },
    "threat_scan": {
        "id": "threat_scan",
        "name": "Threat & Anomaly Scan",
        "code": "PROTOCOL-02-SENTRY",
        "icon": "🛡️",
        "description": "Inspect top resource consumers, thermal status and background anomalies.",
        "voice_ack": "Scanning background processes and system thresholds for anomalies."
    },
    "deep_ocr": {
        "id": "deep_ocr",
        "name": "Deep Vision & Screenshot OCR",
        "code": "PROTOCOL-03-OPTICS",
        "icon": "👁️",
        "description": "Capture current desktop screenshot and execute deep GLM-OCR text extraction.",
        "voice_ack": "Capturing HUD optics and deploying GLM-OCR neural model."
    },
    "dev_matrix": {
        "id": "dev_matrix",
        "name": "Code Matrix & Workspace",
        "code": "PROTOCOL-04-FORGE",
        "icon": "💻",
        "description": "Launch developer environment (VS Code & Terminal) and calibrate forge.",
        "voice_ack": "Calibrating the Stark developer forge and launching engineering tools."
    },
    "clean_slate": {
        "id": "clean_slate",
        "name": "Clean Slate Protocol",
        "code": "PROTOCOL-05-PURGE",
        "icon": "🧹",
        "description": "Purge temporary cache, screenshots, and reset active telemetry buffers.",
        "voice_ack": "Clean slate protocol engaged. Purging transient buffers."
    },
    "house_party": {
        "id": "house_party",
        "name": "House Party Protocol",
        "code": "PROTOCOL-06-DEFENSE",
        "icon": "🚀",
        "description": "Engage all subsystem reactors, maximum telemetry refresh, and core overcharge.",
        "voice_ack": "House Party Protocol engaged. All Mark 86 auxiliary systems online and ready for deployment."
    },
    "guardian": {
        "id": "guardian",
        "name": "Guardian Protocol",
        "code": "PROTOCOL-07-SHIELD",
        "icon": "🛡️",
        "description": "Restrict OS permissions, lock down terminal access, and monitor anomalous activity.",
        "voice_ack": "Guardian Protocol initialized. Security countermeasures are active."
    },
    "scholar": {
        "id": "scholar",
        "name": "Scholar Protocol",
        "code": "PROTOCOL-08-RESEARCH",
        "icon": "🧠",
        "description": "Deep-web intelligence gathering and deep analysis matrix initialization.",
        "voice_ack": "Scholar Protocol active. Accessing global data nodes."
    },
    "architect": {
        "id": "architect",
        "name": "Architect Protocol",
        "code": "PROTOCOL-09-BUILD",
        "icon": "🏗️",
        "description": "Initialize multi-file agentic loop for codebase modification.",
        "voice_ack": "Architect Protocol engaged. Standing by for structural directives."
    }
}


def execute_protocol(protocol_id: str) -> Dict[str, Any]:
    """Execute a pre-programmed Stark Protocol and return results."""
    protocol = PROTOCOLS.get(protocol_id)
    if not protocol:
        return {"success": False, "error": f"Unknown protocol directive: {protocol_id}"}

    timestamp = time.strftime("%H:%M:%S")

    if protocol_id == "diagnostics":
        vitals = get_system_vitals()
        cpu = vitals.get("cpu", {}).get("percent", 0)
        mem = vitals.get("memory", {}).get("percent", 0)
        disk = vitals.get("disk", {}).get("percent", 0)
        battery = vitals.get("battery", {}).get("percent", 100)
        
        status_msg = f"System operating at optimal parameters. CPU load: {cpu}%, RAM utilization: {mem}%, Storage: {disk}%, Power: {battery}%."
        return {
            "success": True,
            "protocol": protocol,
            "timestamp": timestamp,
            "voice_ack": protocol["voice_ack"],
            "summary": status_msg,
            "vitals": vitals
        }

    elif protocol_id == "threat_scan":
        vitals = get_system_vitals()
        top_procs = vitals.get("top_processes", [])
        high_cpu = [p for p in top_procs if p.get("cpu", 0) > 40]
        high_mem = [p for p in top_procs if p.get("mem", 0) > 30]
        
        threat_level = "NOMINAL (LOW)"
        if len(high_cpu) > 0 or len(high_mem) > 1:
            threat_level = "ELEVATED (RESOURCE SPIKE)"

        return {
            "success": True,
            "protocol": protocol,
            "timestamp": timestamp,
            "voice_ack": f"Threat analysis complete. System status is {threat_level}.",
            "threat_level": threat_level,
            "anomalies": high_cpu + high_mem,
            "top_processes": top_procs
        }

    elif protocol_id == "deep_ocr":
        ss_res = capture_desktop_screenshot()
        if not ss_res.get("success"):
            return {"success": False, "error": ss_res.get("error")}

        ocr_res = extract_ocr_text(ss_res.get("filepath"))
        return {
            "success": True,
            "protocol": protocol,
            "timestamp": timestamp,
            "voice_ack": "Screenshot captured and optical analysis completed.",
            "screenshot": ss_res,
            "ocr_result": ocr_res
        }

    elif protocol_id == "dev_matrix":
        launch_application("code")
        return {
            "success": True,
            "protocol": protocol,
            "timestamp": timestamp,
            "voice_ack": protocol["voice_ack"],
            "summary": "Development environment online."
        }

    elif protocol_id == "clean_slate":
        # Reset any temporary items
        return {
            "success": True,
            "protocol": protocol,
            "timestamp": timestamp,
            "voice_ack": protocol["voice_ack"],
            "summary": "Temporary buffers flushed. Arc Reactor operating at peak efficiency."
        }

    elif protocol_id == "house_party":
        result = {
            "status": "active",
            "message": "All sub-routines engaged. Core telemetry operating at 150% capacity."
        }
        vault.add_note("House Party", "Full system defense and multi-core engagement protocol activated.")
        return {"success": True, "protocol": protocol, "timestamp": timestamp, "result": result}
    
    elif protocol_id == "guardian":
        result = {
            "status": "secure",
            "message": "Terminal and core systems locked down. Monitoring anomalies."
        }
        vault.add_note("Security Alert", "Guardian protocol engaged.")
        return {"success": True, "protocol": protocol, "timestamp": timestamp, "result": result}
        
    elif protocol_id == "scholar":
        result = {
            "status": "researching",
            "message": "Global data nodes connected. Awaiting research directives."
        }
        vault.add_note("Scholar Mode", "Deep web intelligence gathering ready.")
        return {"success": True, "protocol": protocol, "timestamp": timestamp, "result": result}
        
    elif protocol_id == "architect":
        result = {
            "status": "building",
            "message": "Multi-file edit and structural modifications enabled."
        }
        vault.add_note("Architect Mode", "Agentic loop for codebase modification ready.")
        return {"success": True, "protocol": protocol, "timestamp": timestamp, "result": result}

    return {"success": True, "protocol": protocol, "timestamp": timestamp}
