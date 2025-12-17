# IELTS Speaking Generator

Bu proje, kullanıcıların girdiği konulara göre IELTS Speaking sınavı simülasyonu oluşturur.

## Özellikler
- Konu bazlı speaking diyalog üretimi
- Examiner (erkek ses) ve Candidate (kadın ses) rolleri
- Text-to-Speech ile ses dönüştürme
- FastAPI backend + TypeScript frontend

## Kurulum

### 1. Backend (FastAPI + Local Models)
```bash
cd backend

# Gerekli paketleri yükle
pip install -r requirements.txt

# Modelleri indir (ilk çalıştırmada)
python model_downloader.py

# Sunucuyu başlat
uvicorn main:app --reload
```

### 2. Frontend (TypeScript)
```bash
cd frontend
npm install
npm run dev
```

## Özellikler
- 🤖 **Local AI Models**: Hugging Face modellerini kullanır (API key gerekmez)
- 🎯 **Text Generation**: DialoGPT ile gerçekçi diyaloglar
- 🔊 **Text-to-Speech**: Coqui TTS ile erkek/kadın sesler
- 📱 **Modern UI**: React + TypeScript ile responsive tasarım
- ⚡ **Hızlı**: Local modeller, internet bağımlılığı yok

## API Endpoints
- POST `/generate-dialogue` - Diyalog üretimi
- POST `/text-to-speech` - Ses dönüştürme
- GET `/audio/{audio_id}` - Ses dosyası servisi

## Model Bilgileri
- **Text Generation**: microsoft/DialoGPT-medium (~1.5GB)
- **TTS**: tts_models/en/vctk/vits (~500MB)
- **Toplam**: ~2GB disk alanı gerekir