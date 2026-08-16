import asyncio
import os
from core.tts_engine import generate_tts, download_piper_model

async def test():
    print("Testing Piper Model Download...")
    await download_piper_model("en_US-lessac-medium")
    await download_piper_model("fa_IR-amir-medium")
    print("Models downloaded.")
    
    print("Testing Edge TTS (English)...")
    path = await generate_tts("Hello Commander. Systems are online.", "edge", "en_US-Male")
    print(f"Generated: {path}, size: {os.path.getsize(path)}")
    
    print("Testing Piper TTS (Persian)...")
    path2 = await generate_tts("سلام فرمانده. سیستم ها آماده کار هستند.", "piper", "fa_IR-Male")
    print(f"Generated: {path2}, size: {os.path.getsize(path2)}")
    
    print("All tests passed.")

if __name__ == "__main__":
    asyncio.run(test())
