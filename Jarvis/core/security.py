import os
from cryptography.fernet import Fernet
import base64

# Simple key derivation for JARVIS
_SECRET_KEY = os.environ.get("JARVIS_SECRET_KEY", "J.A.R.V.I.S.-Mark-86-Secret-Key-32")
# Ensure the key is url-safe base64-encoded 32-byte key
_fernet_key = base64.urlsafe_b64encode(_SECRET_KEY.encode('utf-8')[:32].ljust(32, b'0'))
_cipher_suite = Fernet(_fernet_key)

def encrypt_key(plain_text: str) -> str:
    """Encrypts an API key or sensitive string."""
    if not plain_text or plain_text.startswith("encrypted:"):
        return plain_text
    encrypted_bytes = _cipher_suite.encrypt(plain_text.encode('utf-8'))
    return "encrypted:" + encrypted_bytes.decode('utf-8')

def decrypt_key(cipher_text: str) -> str:
    """Decrypts an API key or sensitive string."""
    if not cipher_text or not cipher_text.startswith("encrypted:"):
        return cipher_text
    
    actual_cipher = cipher_text[len("encrypted:"):]
    try:
        decrypted_bytes = _cipher_suite.decrypt(actual_cipher.encode('utf-8'))
        return decrypted_bytes.decode('utf-8')
    except Exception:
        return "" # On error, return empty string

def validate_command(command: str) -> bool:
    """
    Validates if a command is safe to run.
    Uses a strict whitelist of allowed commands/patterns.
    """
    cmd = command.strip().lower()
    
    # Whitelist of allowed basic commands
    allowed_starts = [
        "ipconfig", "ping", "echo", "dir", "systeminfo",
        "tasklist", "netstat"
    ]
    
    # If the command starts with any allowed safe command, it's fine
    if any(cmd.startswith(safe_cmd) for safe_cmd in allowed_starts):
        return True
        
    return False
