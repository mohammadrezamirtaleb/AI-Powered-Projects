"""
J.A.R.V.I.S. Neural Memory Vault
Author: Stark Industries Mark-85 AI Core
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.security import encrypt_key, decrypt_key

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
VAULT_FILE = DATA_DIR / "memory_vault.json"

DEFAULT_VAULT = {
    "user_profile": {
        "callsign": "Sir",
        "title": "Chief Architect & Stark Commander",
        "primary_language": "Persian & English",
        "system_version": "Mark LXXXV",
        "theme": "Stark Arc Blue & Gold"
    },
    "provider_config": {
        "active_provider": "openrouter",  # "ollama" or "openrouter"
        "ollama_model": "qwen3.5:4b",
        "openrouter_api_key": "",
        "openrouter_model": "google/gemma-4-26b-a4b-it:free"
    },
    "notes": [
        {
            "id": 1,
            "title": "Mark 85 Reactor Core Calibration",
            "content": "Optimal plasma frequency maintained at 12.4 THz. Telemetry auto-refresh every 2.5s.",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    ],
    "learned_facts": [
        "J.A.R.V.I.S. operates on hybrid local neural models (Ollama qwen3.5:4b) & OpenRouter cloud models (Gemma-4-26B).",
        "Protocols can be triggered via prompt directives or HUD matrix buttons."
    ],
    "protocols": []
}

class MemoryVault:
    def __init__(self):
        self.file_path = VAULT_FILE
        self._load()

    def _load(self):
        if not self.file_path.exists():
            self.data = DEFAULT_VAULT.copy()
            self._save()
        else:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                    # Ensure provider_config exists in loaded data
                    if "provider_config" not in self.data:
                        self.data["provider_config"] = DEFAULT_VAULT["provider_config"].copy()
                        self._save()
                    elif "openrouter_api_key" in self.data["provider_config"]:
                        # Decrypt it in memory
                        enc_key = self.data["provider_config"]["openrouter_api_key"]
                        if enc_key.startswith("encrypted:"):
                            self.data["provider_config"]["openrouter_api_key"] = decrypt_key(enc_key)
            except Exception:
                self.data = DEFAULT_VAULT.copy()
                self._save()

    def _save(self):
        save_data = self.data.copy()
        # Encrypt the key before saving
        if "provider_config" in save_data and "openrouter_api_key" in save_data["provider_config"]:
            plain_key = save_data["provider_config"]["openrouter_api_key"]
            if plain_key and not plain_key.startswith("encrypted:"):
                # We need a deep copy of provider_config so we don't encrypt the in-memory version
                save_data["provider_config"] = save_data["provider_config"].copy()
                save_data["provider_config"]["openrouter_api_key"] = encrypt_key(plain_key)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def get_all(self) -> Dict[str, Any]:
        return self.data

    def get_user_profile(self) -> Dict[str, Any]:
        return self.data.get("user_profile", {})

    def update_user_profile(self, profile: Dict[str, Any]):
        self.data["user_profile"].update(profile)
        self._save()

    def get_provider_config(self) -> Dict[str, Any]:
        return self.data.get("provider_config", DEFAULT_VAULT["provider_config"])

    def update_provider_config(self, config: Dict[str, Any]):
        prov_cfg = self.data.setdefault("provider_config", DEFAULT_VAULT["provider_config"].copy())
        prov_cfg.update(config)
        self._save()

    def add_note(self, title: str, content: str) -> Dict[str, Any]:
        notes = self.data.get("notes", [])
        new_id = max([n.get("id", 0) for n in notes], default=0) + 1
        note_entry = {
            "id": new_id,
            "title": title,
            "content": content,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        notes.append(note_entry)
        self.data["notes"] = notes
        self._save()
        return note_entry

    def delete_note(self, note_id: int) -> bool:
        notes = self.data.get("notes", [])
        initial_len = len(notes)
        self.data["notes"] = [n for n in notes if n.get("id") != note_id]
        self._save()
        return len(self.data["notes"]) < initial_len

    def add_fact(self, fact: str) -> None:
        facts = self.data.setdefault("learned_facts", [])
        if fact not in facts:
            facts.append(fact)
            self._save()

    def get_context_summary(self) -> str:
        """Produce a compact context block to inject into LLM system prompt."""
        profile = self.data.get("user_profile", {})
        facts = self.data.get("learned_facts", [])
        notes = self.data.get("notes", [])
        
        ctx = [
            f"Commander Callsign: {profile.get('callsign', 'Sir')}",
            f"Suit Model: {profile.get('system_version', 'Mark 85')}",
            "Known Directives & Facts:"
        ]
        for f in facts[-5:]:
            ctx.append(f"- {f}")
        if notes:
            ctx.append("Active Memoranda / Notes:")
            for n in notes[-3:]:
                ctx.append(f"- [{n.get('title')}]: {n.get('content')}")
        return "\n".join(ctx)

vault = MemoryVault()
