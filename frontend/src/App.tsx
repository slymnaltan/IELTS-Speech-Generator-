import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

interface DialogueLine {
  speaker: string;
  text: string;
}

interface DialogueResponse {
  examiner_lines: string[];
  candidate_lines: string[];
  full_dialogue: DialogueLine[];
}

function App() {
  const [topic, setTopic] = useState('');
  const [difficulty, setDifficulty] = useState('intermediate');
  const [dialogue, setDialogue] = useState<DialogueLine[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentPlaying, setCurrentPlaying] = useState<number | null>(null);
  
  // Podcast player state
  const [podcastUrl, setPodcastUrl] = useState<string>('');
  const [podcastLoading, setPodcastLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);

  const generateDialogue = async () => {
    if (!topic.trim()) {
      alert('Lütfen bir konu girin!');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post<DialogueResponse>('http://localhost:8000/generate-dialogue', {
        topic,
        difficulty
      });
      
      setDialogue(response.data.full_dialogue);
    } catch (error) {
      console.error('Diyalog oluşturulurken hata:', error);
      alert('Diyalog oluşturulurken bir hata oluştu!');
    } finally {
      setLoading(false);
    }
  };

  // Audio player functions
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [podcastUrl]);

  const generatePodcast = async () => {
    if (dialogue.length === 0) {
      alert('Önce bir diyalog oluşturun!');
      return;
    }

    setPodcastLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/generate-podcast', {
        dialogue
      });
      
      setPodcastUrl(`http://localhost:8000${response.data.podcast_url}`);
      alert('Podcast başarıyla oluşturuldu! 🎧');
    } catch (error) {
      console.error('Podcast oluşturulurken hata:', error);
      alert('Podcast oluşturulurken bir hata oluştu!');
    } finally {
      setPodcastLoading(false);
    }
  };

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const newTime = parseFloat(e.target.value);
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const playAudio = async (text: string, voiceType: string, index: number) => {
    setCurrentPlaying(index);
    
    try {
      // TTS API çağrısı
      const response = await axios.post('http://localhost:8000/text-to-speech', {
        text,
        voice_type: voiceType.toLowerCase()
      });
      
      // Ses dosyasını çal
      if (response.data.audio_url) {
        const audio = new Audio(`http://localhost:8000${response.data.audio_url}`);
        
        audio.onended = () => {
          setCurrentPlaying(null);
        };
        
        audio.onerror = () => {
          console.error('Ses çalma hatası');
          setCurrentPlaying(null);
        };
        
        await audio.play();
      }
      
    } catch (error) {
      console.error('Ses çalarken hata:', error);
      alert('Ses oluşturulurken hata oluştu. TTS modeli yüklü olduğundan emin olun.');
      setCurrentPlaying(null);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎯 IELTS Speaking Generator</h1>
        <p>Konu girin ve gerçekçi IELTS speaking pratiği yapın!</p>
      </header>

      <main className="main-content">
        <div className="input-section">
          <div className="form-group">
            <label htmlFor="topic">Konu:</label>
            <input
              id="topic"
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Örn: Favorite hobby, Travel experience, Technology..."
              className="topic-input"
            />
          </div>

          <div className="form-group">
            <label htmlFor="difficulty">Zorluk Seviyesi:</label>
            <select
              id="difficulty"
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              className="difficulty-select"
            >
              <option value="beginner">Başlangıç</option>
              <option value="intermediate">Orta</option>
              <option value="advanced">İleri</option>
            </select>
          </div>

          <button 
            onClick={generateDialogue}
            disabled={loading}
            className="generate-btn"
          >
            {loading ? '🔄 Oluşturuluyor...' : '✨ Diyalog Oluştur'}
          </button>
        </div>

        <div className="content-layout">
          {dialogue.length > 0 && (
            <div className="dialogue-section">
              <div className="section-header">
                <h2>📝 Generated Dialogue</h2>
                <button 
                  onClick={generatePodcast}
                  disabled={podcastLoading}
                  className="podcast-btn"
                >
                  {podcastLoading ? '🔄 Creating Podcast...' : '🎧 Create Podcast'}
                </button>
              </div>
              
              <div className="dialogue-container">
                {dialogue.map((line, index) => (
                  <div 
                    key={index} 
                    className={`dialogue-line ${line.speaker.toLowerCase()}`}
                  >
                    <div className="speaker-info">
                      <span className="speaker-name">
                        {line.speaker === 'EXAMINER' ? '👨‍🏫 Examiner' : '👩‍🎓 Candidate'}
                      </span>
                      <button
                        onClick={() => playAudio(line.text, line.speaker, index)}
                        disabled={currentPlaying === index}
                        className="play-btn"
                      >
                        {currentPlaying === index ? '🔊' : '▶️'}
                      </button>
                    </div>
                    <p className="dialogue-text">{line.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {podcastUrl && (
            <div className="podcast-player">
              <h2>🎧 IELTS Speaking Podcast</h2>
              
              <div className="player-container">
                <audio ref={audioRef} src={podcastUrl} preload="metadata" />
                
                <div className="player-controls">
                  <button 
                    onClick={togglePlayPause}
                    className="play-pause-btn"
                  >
                    {isPlaying ? '⏸️' : '▶️'}
                  </button>
                  
                  <div className="time-display">
                    {formatTime(currentTime)}
                  </div>
                </div>

                <div className="progress-container">
                  <input
                    type="range"
                    min="0"
                    max={duration || 0}
                    value={currentTime}
                    onChange={handleSeek}
                    className="progress-slider"
                  />
                  <div className="duration-display">
                    {formatTime(duration)}
                  </div>
                </div>

                <div className="podcast-info">
                  <p>🎯 Topic: <strong>{topic}</strong></p>
                  <p>📊 Level: <strong>{difficulty}</strong></p>
                  <p>💬 Exchanges: <strong>{dialogue.length}</strong></p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;