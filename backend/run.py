"""
Quick start script for IELTS Speaking Generator
"""

import subprocess
import sys
import os

def check_models():
    """Check if models are downloaded"""
    
    print("🔍 Checking models...")
    
    # Check if transformers cache exists
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    tts_cache = os.path.expanduser("~/.local/share/tts")
    
    models_exist = os.path.exists(cache_dir) and os.path.exists(tts_cache)
    
    if not models_exist:
        print("📥 Models not found. Downloading...")
        subprocess.run([sys.executable, "model_downloader.py"])
    else:
        print("✅ Models found")

def start_server():
    """Start the FastAPI server"""
    
    print("🚀 Starting IELTS Speaking Generator server...")
    
    try:
        subprocess.run([
            "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    print("🎯 IELTS Speaking Generator")
    print("=" * 40)
    
    check_models()
    start_server()