"""
Local TTS Service using Hugging Face models
No API keys required - everything runs locally
"""

import os
import torch
import torchaudio
from TTS.api import TTS
from typing import Optional, Dict
import uuid

class LocalTTSService:
    """Local Text-to-Speech service using Coqui TTS"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_dir = "audio_files"
        os.makedirs(self.audio_dir, exist_ok=True)
        
    def load_model(self):
        """Load the TTS model"""
        try:
            # Using VITS model with multiple speakers
            model_name = "tts_models/en/vctk/vits"
            self.model = TTS(model_name).to(self.device)
            print(f"✅ TTS model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"❌ Failed to load TTS model: {e}")
            return False
    
    def get_available_speakers(self) -> Dict[str, str]:
        """Get available speaker voices"""
        return {
            "examiner": "p225",  # Male British accent
            "candidate": "p231", # Female British accent
            "examiner_alt": "p226", # Male alternative
            "candidate_alt": "p232" # Female alternative
        }
    
    async def generate_speech(self, text: str, voice_type: str) -> Optional[str]:
        """Generate speech and return audio file path"""
        
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            speakers = self.get_available_speakers()
            speaker = speakers.get(voice_type, speakers["examiner"])
            
            # Generate unique filename
            audio_id = str(uuid.uuid4())
            audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav")
            
            # Generate speech
            self.model.tts_to_file(
                text=text,
                speaker=speaker,
                file_path=audio_path
            )
            
            return audio_path
            
        except Exception as e:
            print(f"TTS generation error: {e}")
            return None
    
    def cleanup_old_files(self, max_files: int = 50):
        """Clean up old audio files to save space"""
        try:
            files = os.listdir(self.audio_dir)
            if len(files) > max_files:
                # Sort by creation time and remove oldest
                files_with_time = [(f, os.path.getctime(os.path.join(self.audio_dir, f))) for f in files]
                files_with_time.sort(key=lambda x: x[1])
                
                for file_name, _ in files_with_time[:-max_files]:
                    os.remove(os.path.join(self.audio_dir, file_name))
                    
        except Exception as e:
            print(f"Cleanup error: {e}")

# Alternative lightweight TTS using transformers
class TransformersTTSService:
    """Alternative TTS using transformers library"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load SpeechT5 model for TTS"""
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            self.processor = processor
            self.model = model
            self.vocoder = vocoder
            
            print("✅ SpeechT5 TTS model loaded")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load SpeechT5: {e}")
            return False