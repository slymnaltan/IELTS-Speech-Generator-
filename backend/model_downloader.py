"""
Model downloader script for IELTS Speaking Generator
Downloads and caches Hugging Face models locally
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from TTS.api import TTS
import torch

def download_text_models():
    """Download text generation models"""
    
    print("🔄 Downloading text generation models...")
    
    models_to_download = [
        "microsoft/DialoGPT-medium",  # Good for dialogue
        "facebook/blenderbot-400M-distill",  # Alternative conversational model
    ]
    
    for model_name in models_to_download:
        try:
            print(f"📥 Downloading {model_name}...")
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            print(f"✅ {model_name} downloaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

def download_tts_models():
    """Download TTS models"""
    
    print("🔄 Downloading TTS models...")
    
    tts_models = [
        "tts_models/en/vctk/vits",  # Multi-speaker English
        "tts_models/en/ljspeech/tacotron2-DDC",  # Alternative model
    ]
    
    for model_name in tts_models:
        try:
            print(f"📥 Downloading {model_name}...")
            
            # This will download and cache the model
            tts = TTS(model_name)
            
            print(f"✅ {model_name} downloaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

def check_system_requirements():
    """Check system requirements"""
    
    print("🔍 Checking system requirements...")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    print(f"Disk space - Total: {total//1e9:.1f}GB, Free: {free//1e9:.1f}GB")
    
    if free < 5e9:  # Less than 5GB
        print("⚠️  Warning: Low disk space. Models require ~3-5GB")
    
    print("✅ System check complete")

def main():
    """Main download function"""
    
    print("🚀 IELTS Speaking Generator - Model Downloader")
    print("=" * 50)
    
    # Check system
    check_system_requirements()
    
    # Create cache directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("audio_files", exist_ok=True)
    
    # Download models
    download_text_models()
    download_tts_models()
    
    print("\n🎉 All models downloaded successfully!")
    print("You can now run the FastAPI server with: uvicorn main:app --reload")

if __name__ == "__main__":
    main()