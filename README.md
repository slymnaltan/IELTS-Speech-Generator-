# IELTS Speaking Generator

An AI-powered IELTS Speaking test simulator that generates realistic dialogues based on user-provided topics.

<img width="1920" height="1080" alt="Ekran Görüntüsü (21)" src="https://github.com/user-attachments/assets/b040f621-f702-4630-8fe4-13a980fccedc" />

## Features
- Topic-based speaking dialogue generation
- Examiner and Candidate role simulation
- Text-to-Speech conversion with different voices
- FastAPI backend + TypeScript React frontend
- Podcast-style audio generation

## Installation

### 1. Backend (FastAPI + Local AI Models)
```bash
cd backend

# Install required packages
pip install -r requirements.txt

# Start the server (models will download automatically on first run)
python main.py
```

### 2. Frontend (TypeScript React)
```bash
cd frontend
npm install
npm run dev
```

## Usage
1. Open http://localhost:3000 in your browser
2. Enter a topic (e.g., "Technology in education")
3. Select difficulty level (Beginner/Intermediate/Advanced)
4. Click "Generate Dialogue" to create IELTS-style conversation
5. Click "Generate Podcast" to create audio version

## Key Features
- 🤖 **Local AI Models**: Uses Qwen2.5-1.5B-Instruct (no API keys required)
- 🎯 **Smart Generation**: Creates 5-6 examiner questions with detailed candidate responses
- 🔊 **Text-to-Speech**: Google TTS with British/American accents
- 📱 **Modern UI**: Responsive design with audio player
- ⚡ **Fast & Offline**: No internet dependency after initial setup
- 🎚️ **Difficulty Levels**: Adapts vocabulary and complexity to IELTS bands

## API Endpoints
- `POST /generate-dialogue` - Generate IELTS dialogue
- `POST /text-to-speech` - Convert text to speech
- `POST /generate-podcast` - Create combined audio
- `GET /audio/{audio_id}` - Serve audio files

## Model Information
- **Text Generation**: Qwen/Qwen2.5-1.5B-Instruct (~3GB)
- **TTS**: Google Text-to-Speech (gTTS)
- **Total**: ~3GB disk space required

## Difficulty Levels
- **Beginner (4.0-5.5)**: Simple vocabulary, basic sentence structures
- **Intermediate (6.0-6.5)**: Good vocabulary range, detailed explanations
- **Advanced (7.0-8.5)**: Sophisticated vocabulary, analytical thinking

## Technology Stack
- **Backend**: FastAPI, Transformers, PyTorch, gTTS
- **Frontend**: React, TypeScript, Vite
- **AI Model**: Qwen2.5-1.5B-Instruct
- **Audio**: Google Text-to-Speech
