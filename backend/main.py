from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from typing import List, Dict
import asyncio
from transformers import pipeline
import torch
from gtts import gTTS
import uuid
import json
import time
import subprocess
import tempfile

app = FastAPI(title="IELTS Speaking Generator")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
text_generator = None
tts_model = None

# Initialize models on startup
@app.on_event("startup")
async def load_models():
    global text_generator, tts_model
    
    print("🔄 Loading models...")
    
    # Load Qwen2.5-1.5B for lightweight instruction following
    try:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"📥 Loading model: {model_name}")
        text_generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            trust_remote_code=True
        )
        print("✅ Qwen2.5-1.5B model loaded successfully")
    except Exception as e:
        print(f"❌ Qwen model error: {e}")
        # Fallback to GPT-2
        try:
            model_name = "gpt2"
            print(f"📥 Loading fallback model: {model_name}")
            text_generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float32,
                pad_token_id=50256
            )
            print("✅ GPT-2 fallback model loaded successfully")
        except Exception as e2:
            print(f"❌ All models failed: {e2}")
            text_generator = None
    
    # TTS will use Google TTS (gTTS) - no model loading needed
    print("✅ TTS ready (using gTTS)")
    
    # Create audio directory
    os.makedirs("audio_files", exist_ok=True)

class TopicRequest(BaseModel):
    topic: str
    difficulty: str = "intermediate"  # beginner, intermediate, advanced

class DialogueResponse(BaseModel):
    examiner_lines: List[str]
    candidate_lines: List[str]
    full_dialogue: List[Dict[str, str]]

class TTSRequest(BaseModel):
    text: str
    voice_type: str  # "examiner" or "candidate"

class PodcastRequest(BaseModel):
    dialogue: List[Dict[str, str]]  # [{"speaker": "EXAMINER", "text": "..."}, ...]

@app.get("/")
async def root():
    return {"message": "IELTS Speaking Generator API"}

@app.post("/generate-dialogue", response_model=DialogueResponse)
async def generate_dialogue(request: TopicRequest):
    global text_generator
    
    if not text_generator:
        raise HTTPException(status_code=503, detail="Text generation model not loaded")
    
    try:
        print(f"🎯 Generating dialogue for topic: {request.topic} (Level: {request.difficulty})")
        
        # Create detailed IELTS-specific prompt for better generation
        difficulty_instructions = {
            "beginner": "The candidate should use clear, simple language with detailed explanations and personal examples. Each response should be 4-5 sentences with specific details, reasons, and complete thoughts. Avoid complex grammar but include rich content.",
            "intermediate": "The candidate should demonstrate good vocabulary range and grammatical variety. Responses should be 5-6 sentences each, including detailed examples, clear explanations, personal experiences, and well-structured arguments with smooth transitions.",
            "advanced": "The candidate should use sophisticated vocabulary, complex sentence structures, and analytical thinking. Responses should be 6-7 sentences each with in-depth analysis, multiple examples, abstract concepts, and nuanced perspectives that show critical thinking."
        }
        
        instruction = difficulty_instructions.get(request.difficulty, difficulty_instructions["intermediate"])
        
        # Use Phi-3 with perfect IELTS instruction prompt
        # Level-specific prompts for better results
        if request.difficulty == "beginner":
            prompt = f"""Generate a simple IELTS Speaking interview about {request.topic} for BEGINNER level (IELTS Band 4.0-5.5).

Use simple vocabulary and clear sentences. Candidate should give basic but complete answers.

Create exactly 5 question-answer pairs:

Examiner: I'd like you to describe {request.topic}. Please tell me about it.

Candidate: [Simple 2-3 sentence answer using basic vocabulary]

Examiner: [Simple question about personal experience]

Candidate: [Basic answer with simple examples]

Examiner: [Easy question about good points]

Candidate: [Simple answer about benefits]

Examiner: [Basic question about problems]

Candidate: [Simple answer about difficulties]

Examiner: [Easy question about the future]

Candidate: [Basic answer about future]

Generate the interview:"""
            
        elif request.difficulty == "advanced":
            prompt = f"""Generate a sophisticated IELTS Speaking interview about {request.topic} for ADVANCED level (IELTS Band 7.0-8.5).

Use complex vocabulary and analytical thinking. Candidate should give detailed, nuanced answers.

Create exactly 5 question-answer pairs:

Examiner: I'd like you to analyze {request.topic}. Please discuss its significance and implications.

Candidate: [Sophisticated 3-4 sentence answer with complex vocabulary and analysis]

Examiner: [Complex question requiring critical thinking]

Candidate: [Detailed analytical answer with multiple perspectives]

Examiner: [Question about broader implications]

Candidate: [Nuanced answer showing deep understanding]

Examiner: [Question about challenges and solutions]

Candidate: [Comprehensive answer with sophisticated reasoning]

Examiner: [Question about future trends and predictions]

Candidate: [Insightful answer with predictions and analysis]

Generate the interview:"""
            
        else:  # intermediate
            prompt = f"""Generate a complete IELTS Speaking interview about {request.topic} for INTERMEDIATE level (IELTS Band 6.0-6.5).

Use good vocabulary and clear explanations. Candidate should give detailed but accessible answers.

Create exactly 5 question-answer pairs:

Examiner: I'd like you to describe {request.topic}. You have one minute to prepare and speak for up to two minutes.

Candidate: [2-3 sentence detailed answer about the topic]

Examiner: [Follow-up question about personal experience]

Candidate: [2-3 sentence answer with personal examples]

Examiner: [Question about advantages or benefits]

Candidate: [2-3 sentence answer explaining advantages]

Examiner: [Question about challenges or disadvantages]

Candidate: [2-3 sentence answer about challenges]

Examiner: [Question about future or recommendations]

Candidate: [2-3 sentence answer about future/recommendations]

Generate the interview:"""

        try:
            generated = text_generator(
                prompt,
                max_new_tokens=1200,  # Much longer for complete interview with full answers
                temperature=0.7,      # Balanced creativity
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False,
                pad_token_id=text_generator.tokenizer.eos_token_id if hasattr(text_generator.tokenizer, 'eos_token_id') else None
            )
            
            generated_text = generated[0]['generated_text']
            print(f"📝 Phi-3 generated text length: {len(generated_text)} characters")
            print(f"🔍 Generated text preview: {generated_text[:300]}...")
            
            # Parse the Qwen generated dialogue
            best_dialogue = parse_qwen_dialogue(generated_text)
            
        except Exception as e:
            print(f"❌ Qwen generation failed: {e}")
            # Fallback to simple approach
            best_dialogue = create_simple_fallback(request.topic, request.difficulty)
        
        # Generate step by step - no loop needed
        print(f"🔄 Generating IELTS dialogue step by step for {request.topic}")
        
        # The dialogue is already generated above in the step-by-step process
        
        # Accept ANYTHING the AI generates - no restrictions at all
        if not best_dialogue:
            # If somehow no dialogue was parsed, create minimal from raw text
            print("🚨 No dialogue parsed, using raw AI output")
            best_dialogue = [
                {"speaker": "EXAMINER", "text": f"Please tell me about {request.topic}."},
                {"speaker": "CANDIDATE", "text": "I'd like to discuss this topic with you."}
            ]
        
        examiner_lines = [line["text"] for line in best_dialogue if line["speaker"] == "EXAMINER"]
        candidate_lines = [line["text"] for line in best_dialogue if line["speaker"] == "CANDIDATE"]
        
        print(f"🎉 Final dialogue: {len(examiner_lines)} examiner + {len(candidate_lines)} candidate lines")
        
        return DialogueResponse(
            examiner_lines=examiner_lines,
            candidate_lines=candidate_lines,
            full_dialogue=best_dialogue
        )
        
    except Exception as e:
        print(f"❌ Dialogue generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_qwen_dialogue(generated_text: str) -> List[Dict[str, str]]:
    """Parse Qwen generated IELTS dialogue - SIMPLE APPROACH"""
    print(f"📋 Parsing Qwen generated dialogue ({len(generated_text)} chars)")
    print(f"🔍 Full text to parse: {generated_text[:500]}...")
    
    # BASIT YAKLAŞIM: Metni cümlelere böl ve alternatif yap
    dialogue = []
    
    # Metni temizle ve cümlelere böl
    text = generated_text.strip()
    
    # Noktalama işaretlerine göre böl
    sentences = []
    for sent in text.split('.'):
        sent = sent.strip()
        if sent and len(sent) > 10:  # Çok kısa cümleler atla
            sentences.append(sent + '.')
    
    # İlk cümle examiner sorusu olsun
    dialogue.append({
        "speaker": "EXAMINER",
        "text": f"I'd like you to describe key of the good presentation. You have one minute to prepare and speak for up to two minutes."
    })
    
    # Sonraki cümleleri alternatif olarak dağıt
    is_examiner = False  # Bir sonraki candidate olsun
    
    for i, sentence in enumerate(sentences[:12]):  # Max 12 cümle al (daha fazla içerik)
        if sentence:
            speaker = "EXAMINER" if is_examiner else "CANDIDATE"
            
            # Examiner cümlelerini soru haline getir
            if is_examiner and not sentence.endswith('?'):
                sentence = sentence.rstrip('.') + '?'
            
            # Candidate cevaplarını kontrol et - çok kısa ise genişlet
            if not is_examiner and len(sentence) < 50:
                sentence = sentence + " This is something I have experience with and find quite important."
            
            dialogue.append({
                "speaker": speaker,
                "text": sentence
            })
            
            is_examiner = not is_examiner  # Alternatif yap
            
            print(f"✅ Added {speaker}: {sentence[:50]}...")
    
    # Son cevabın tam olduğundan emin ol
    if len(dialogue) > 0 and dialogue[-1]["speaker"] == "CANDIDATE":
        last_answer = dialogue[-1]["text"]
        if not last_answer.endswith('.') and not last_answer.endswith('!'):
            dialogue[-1]["text"] = last_answer + "."
    
    print(f"✅ Created {len(dialogue)} dialogue exchanges")
    return dialogue

def create_simple_fallback(topic: str, difficulty: str) -> List[Dict[str, str]]:
    """Simple fallback if Phi-3 fails"""
    return [
        {
            "speaker": "EXAMINER",
            "text": f"I'd like you to describe {topic}. You have one minute to prepare and speak for up to two minutes."
        },
        {
            "speaker": "CANDIDATE", 
            "text": f"I'd like to talk about {topic}. This is something I find quite interesting and relevant to my daily life."
        },
        {
            "speaker": "EXAMINER",
            "text": f"Can you tell me about your personal experience with {topic}?"
        },
        {
            "speaker": "CANDIDATE",
            "text": f"Well, I have quite a bit of experience with {topic}. It has been part of my life for several years now."
        }
    ]

def generate_ielts_answer(question: str, topic: str, difficulty: str) -> str:
    """Generate a proper IELTS answer using simple prompting"""
    global text_generator
    
    try:
        # Create context-aware prompts based on difficulty
        if difficulty == "beginner":
            prompt = f"Answer this IELTS question using simple English: {question}\n\nAnswer: I think {topic}"
        elif difficulty == "intermediate":
            prompt = f"Give a detailed IELTS answer with good vocabulary: {question}\n\nAnswer: Well, regarding {topic}, I would say"
        else:  # advanced
            prompt = f"Provide a sophisticated IELTS answer with complex ideas: {question}\n\nAnswer: From my perspective on {topic},"
            
        generated = text_generator(
            prompt,
            max_new_tokens=80,  # Shorter, focused responses
            temperature=0.7,    # Less randomness for coherence
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.1,
            pad_token_id=50256,
            return_full_text=False
        )
        
        response = generated[0]['generated_text'].strip()
        
        # Clean and format response
        response = response.split('\n')[0]  # First line only
        response = response.replace('"', '').replace("'", "")
        
        # Add starter if response is too short
        if len(response) < 30:
            starters = [
                f"I believe {topic} is very important because",
                f"In my experience with {topic}, I have found that",
                f"When it comes to {topic}, I think",
                f"From what I know about {topic},"
            ]
            response = random.choice(starters) + " " + response
        
        # Ensure proper ending
        if not response.endswith('.'):
            response += "."
            
        return response
        
    except Exception as e:
        print(f"❌ IELTS answer generation error: {e}")
        # Simple fallback answers
        fallbacks = [
            f"I think {topic} is very interesting and important in today's world.",
            f"From my personal experience, {topic} has had a significant impact on my life.",
            f"In my opinion, {topic} offers both advantages and challenges that we need to consider.",
            f"I believe {topic} will continue to be relevant and important in the future."
        ]
        return random.choice(fallbacks)

def generate_single_response(prompt: str, difficulty: str, speaker_type: str) -> str:
    """Generate a single response using AI"""
    global text_generator
    
    try:
        # Adjust prompt based on difficulty and speaker type
        if speaker_type == "examiner":
            if difficulty == "beginner":
                prompt += " Ask a simple, clear question."
            elif difficulty == "intermediate": 
                prompt += " Ask a thoughtful question that requires explanation."
            else:  # advanced
                prompt += " Ask an analytical question that requires critical thinking."
        else:  # candidate
            if difficulty == "beginner":
                prompt += " Give a simple, clear answer with personal examples."
            elif difficulty == "intermediate": 
                prompt += " Give a detailed answer with good vocabulary and examples."
            else:  # advanced
                prompt += " Give a sophisticated answer with complex ideas and analysis."
            
        generated = text_generator(
            prompt,
            max_new_tokens=100 if speaker_type == "examiner" else 200,  # Shorter questions, longer answers
            temperature=0.9,  # Higher creativity for variety
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.2,  # Prevent repetition
            pad_token_id=50256,
            return_full_text=False
        )
        
        response = generated[0]['generated_text'].strip()
        
        # Clean up the response
        response = response.split('\n')[0]  # Take first line only
        response = response.replace('"', '').replace("'", "")  # Remove quotes
        response = response.strip('.,!?')  # Remove trailing punctuation
        
        # Add proper punctuation
        if speaker_type == "examiner" and not response.endswith('?'):
            response += "?"
        elif speaker_type == "candidate" and not response.endswith('.'):
            response += "."
        
        # Ensure minimum length with better fallbacks
        if len(response) < 15:
            if speaker_type == "candidate":
                response = f"I believe this topic is quite significant and I have personal experience with it that I'd like to share."
            else:
                response = "What are your thoughts on this particular aspect?"
                
        return response
        
    except Exception as e:
        print(f"❌ Single response generation error: {e}")
        # Better fallback responses
        if speaker_type == "candidate":
            return f"This is definitely something I find meaningful and I have quite a bit to say about it."
        else:
            return "How would you describe your experience with this?"

def parse_dialogue(generated_text: str, topic: str, difficulty: str) -> List[Dict[str, str]]:
    """Parse generated text - accept EVERYTHING from AI"""
    print(f"📋 Parsing AI-generated text ({len(generated_text)} chars)")
    print(f"🔍 Full generated text: {generated_text}")
    
    lines = generated_text.split('\n')
    dialogue = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('EXAMINER:'):
            text = line.replace('EXAMINER:', '').strip()
            # NO RESTRICTIONS - accept any text
            if text:
                dialogue.append({
                    "speaker": "EXAMINER",
                    "text": text
                })
                print(f"✅ Found EXAMINER: {text}")
        elif line.startswith('CANDIDATE:'):
            text = line.replace('CANDIDATE:', '').strip()
            # NO RESTRICTIONS - accept any text
            if text:
                dialogue.append({
                    "speaker": "CANDIDATE", 
                    "text": text
                })
                print(f"✅ Found CANDIDATE: {text}")
    
    print(f"📊 Parsed {len(dialogue)} dialogue lines - accepting all")
    
    # If no proper format found, try to create dialogue from raw text
    if len(dialogue) == 0:
        print("🔧 No EXAMINER/CANDIDATE format found, creating from raw text")
        # Split the text and create alternating dialogue
        sentences = [s.strip() for s in generated_text.split('.') if s.strip()]
        for i, sentence in enumerate(sentences[:10]):  # Max 10 sentences
            if sentence:
                speaker = "EXAMINER" if i % 2 == 0 else "CANDIDATE"
                dialogue.append({
                    "speaker": speaker,
                    "text": sentence + "."
                })
    
    examiner_count = len([d for d in dialogue if d["speaker"] == "EXAMINER"])
    candidate_count = len([d for d in dialogue if d["speaker"] == "CANDIDATE"])
    
    print(f"✅ Final dialogue: {examiner_count} examiner questions, {candidate_count} candidate responses")
    return dialogue

# ALL TEMPLATE FUNCTIONS REMOVED - PURE AI GENERATION ONLY

@app.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    try:
        # Voice configuration for gTTS
        voice_config = {
            "examiner": {"lang": "en", "tld": "co.uk", "slow": False},  # British male-ish
            "candidate": {"lang": "en", "tld": "com", "slow": False}    # American female-ish
        }
        
        config = voice_config.get(request.voice_type, voice_config["examiner"])
        
        # Generate unique filename
        audio_id = str(uuid.uuid4())
        audio_path = f"audio_files/{audio_id}.mp3"
        
        # Generate speech using gTTS
        tts = gTTS(
            text=request.text,
            lang=config["lang"],
            tld=config["tld"],
            slow=config["slow"]
        )
        
        tts.save(audio_path)
        
        return {
            "audio_id": audio_id,
            "audio_url": f"/audio/{audio_id}.mp3",
            "voice_type": request.voice_type,
            "config": config,
            "text": request.text
        }
        
    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/generate-podcast")
async def generate_podcast(request: PodcastRequest):
    """Generate a complete podcast from dialogue"""
    try:
        podcast_id = str(uuid.uuid4())
        temp_files = []
        
        # Voice configuration
        voice_config = {
            "EXAMINER": {"lang": "en", "tld": "co.uk", "slow": False},  # British male-ish
            "CANDIDATE": {"lang": "en", "tld": "com", "slow": False}    # American female-ish
        }
        
        # Generate individual audio files with natural pauses
        for i, line in enumerate(request.dialogue):
            speaker = line["speaker"]
            text = line["text"]
            
            # Clean text for better TTS but ensure it's not empty
            text = text.replace("...", "").strip()
            if not text or len(text) < 3:  # Skip if text is too short or empty
                continue
                
            print(f"Generating TTS for {speaker}: {text[:50]}...")  # Debug log
            
            # Generate TTS for this line
            config = voice_config.get(speaker, voice_config["EXAMINER"])
            
            temp_file = f"audio_files/temp_{podcast_id}_{i}.mp3"
            temp_files.append(temp_file)
            
            try:
                tts = gTTS(
                    text=text,
                    lang=config["lang"],
                    tld=config["tld"],
                    slow=config["slow"]
                )
                
                tts.save(temp_file)
                print(f"✅ Generated audio for {speaker}")
                
            except Exception as tts_error:
                print(f"❌ TTS error for {speaker}: {tts_error}")
                # Remove the failed file from temp_files list
                temp_files.remove(temp_file)
                continue
            
            # Add a pause between speakers using a short word
            if i < len(request.dialogue) - 1:
                pause_file = f"audio_files/pause_{podcast_id}_{i}.mp3"
                temp_files.append(pause_file)
                
                try:
                    # Create a short pause using a brief word
                    pause_tts = gTTS(text="hmm", lang="en", slow=True)
                    pause_tts.save(pause_file)
                except Exception as pause_error:
                    print(f"❌ Pause generation error: {pause_error}")
                    # Remove pause file from list if it failed
                    temp_files.remove(pause_file)
        
        # Combine all audio files using a simple approach
        podcast_path = f"audio_files/podcast_{podcast_id}.mp3"
        
        # Check if we have any audio files to combine
        if not temp_files:
            raise HTTPException(status_code=400, detail="No audio files were generated")
        
        # Create a combined audio by concatenating all files
        combined_audio_data = b""
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    audio_data = f.read()
                    combined_audio_data += audio_data
                print(f"✅ Added {temp_file} to podcast")
            else:
                print(f"❌ File not found: {temp_file}")
        
        # Write combined audio
        with open(podcast_path, 'wb') as f:
            f.write(combined_audio_data)
            
        print(f"✅ Podcast created: {podcast_path} ({len(combined_audio_data)} bytes)")
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Estimate duration (rough calculation)
        total_chars = sum(len(line["text"]) for line in request.dialogue)
        estimated_duration = total_chars * 0.1  # Rough estimate: 0.1 seconds per character
        
        return {
            "podcast_id": f"podcast_{podcast_id}.mp3",
            "podcast_url": f"/audio/podcast_{podcast_id}.mp3",
            "duration": estimated_duration,
            "segments": len(request.dialogue),
            "message": "Podcast generated successfully"
        }
        
    except Exception as e:
        print(f"Podcast generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Podcast generation failed: {str(e)}")

@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Serve generated audio files"""
    audio_path = f"audio_files/{audio_id}"
    
    if os.path.exists(audio_path):
        media_type = "audio/mpeg" if audio_id.endswith('.mp3') else "audio/wav"
        return FileResponse(
            audio_path,
            media_type=media_type,
            filename=f"{audio_id}"
        )
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)