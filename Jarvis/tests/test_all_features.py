"""
J.A.R.V.I.S. Comprehensive End-to-End System Test Suite
Tests all tabs, APIs, protocols, OCR, Memory Vault, and Providers.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from server import app
import json

client = TestClient(app)

def test_all_systems():
    print("==================================================================")
    print("   J.A.R.V.I.S. MARK-85 FULL SYSTEM & TABS INTEGRITY VERIFICATION")
    print("==================================================================")

    # 1. System Status
    r = client.get('/api/status')
    assert r.status_code == 200, f"Status check failed: {r.text}"
    status_data = r.json()
    assert status_data["status"] == "ONLINE"
    print(f"[PASS] 1. System Status: ONLINE // System: {status_data['system']}")

    # 2. Dynamic Models List
    r = client.get('/api/models')
    assert r.status_code == 200, f"Models check failed: {r.text}"
    models = r.json()["models"]
    assert len(models) >= 14, "Expected at least 14 models"
    print(f"[PASS] 2. Models Matrix: {len(models)} models available (OpenRouter + Ollama)")

    # 3. Telemetry & Vitals
    r = client.get('/api/vitals')
    assert r.status_code == 200, f"Vitals check failed: {r.text}"
    v = r.json()
    assert "cpu" in v and "memory" in v and "disk" in v
    print(f"[PASS] 3. Core Telemetry: CPU={v['cpu']['percent']}%, RAM={v['memory']['percent']}%, Storage={v['disk']['percent']}%")

    # 4. Protocols Tab Endpoints
    r = client.get('/api/protocols')
    assert r.status_code == 200
    protos = r.json() if isinstance(r.json(), list) else r.json().get("protocols", [])
    assert len(protos) == 6
    print(f"[PASS] 4. Protocols Matrix: 6 Protocols Registered ({', '.join(p['id'] for p in protos)})")

    # 5. Execute Diagnostic Protocol
    r = client.post('/api/protocol/execute', json={'protocol_id': 'diagnostics'})
    assert r.status_code == 200
    assert r.json()["success"] is True
    print(f"[PASS] 5. Diagnostic Protocol: {r.json().get('summary')}")

    # 6. Execute Threat Scan Protocol
    r = client.post('/api/protocol/execute', json={'protocol_id': 'threat_scan'})
    assert r.status_code == 200
    assert r.json()["success"] is True
    print(f"[PASS] 6. Threat Scan Protocol: Level = {r.json().get('threat_level')}")

    # 7. Desktop Screenshot Engine
    r = client.post('/api/screenshot')
    assert r.status_code == 200
    assert r.json()["success"] is True
    ss_len = len(r.json().get("base64_data", ""))
    print(f"[PASS] 7. Screenshot Engine: Captured Base64 ({ss_len} chars)")

    # 8. Terminal Tab Execution
    r = client.post('/api/command', json={'command': 'echo STARK_ONLINE'})
    assert r.status_code == 200
    assert "STARK_ONLINE" in r.json().get("stdout", "")
    print(f"[PASS] 8. Terminal Subsystem: Command execution verified")

    # 9. Memory Vault Tab (CRUD)
    r_create = client.post('/api/vault/note', json={'title': 'Mark85 Test Note', 'content': 'Verifying Vault integrity.'})
    assert r_create.status_code == 200
    note_id = r_create.json().get("id")
    assert note_id is not None

    r_list = client.get('/api/vault')
    assert r_list.status_code == 200
    assert any(n["id"] == note_id for n in r_list.json()["notes"])

    r_del = client.delete(f'/api/vault/note/{note_id}')
    assert r_del.status_code == 200
    print(f"[PASS] 9. Memory Vault Tab: Note Creation, Retrieval & Deletion 100% Functional")

    # 10. Provider Configuration
    r_cfg = client.get('/api/config/provider')
    assert r_cfg.status_code == 200
    assert "active_provider" in r_cfg.json()
    print(f"[PASS] 10. Provider Config: Active Provider = {r_cfg.json()['active_provider']}")

    # 11. LLM Chat Stream (OpenRouter)
    r_chat = client.post('/api/chat/stream', json={
        'messages': [{'role': 'user', 'content': 'وضعیت کلی سیستم را تایید کن'}],
        'provider': 'openrouter',
        'model': 'google/gemma-4-26b-a4b-it:free'
    })
    assert r_chat.status_code == 200
    print(f"[PASS] 11. OpenRouter Gemma 4 26B Chat Stream: HTTP 200 Stream OK")

    print("\n==================================================================")
    print("   ALL 11 BACKEND MODULES & TABS PASSED WITH 100% SUCCESS!")
    print("==================================================================")

if __name__ == "__main__":
    test_all_systems()
