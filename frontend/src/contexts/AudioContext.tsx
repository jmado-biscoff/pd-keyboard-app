import React, { createContext, useContext, useEffect, useRef, useState } from 'react';

// Import audio files
import backgroundMusic from '@/assets/sounds/background_music.mp3';
import click1 from '@/assets/sounds/click_1.mp3';
import click2 from '@/assets/sounds/click_2.mp3';
import click3 from '@/assets/sounds/click_3.mp3';
import errorSound from '@/assets/sounds/error.mp3';

interface AudioContextType {
  musicVolume: number;
  soundVolume: number;
  setMusicVolume: (volume: number) => void;
  setSoundVolume: (volume: number) => void;
  playCorrectSound: () => void;
  playErrorSound: () => void;
  muteMusicTemporarily: (mute: boolean) => void;
}

const AudioContext = createContext<AudioContextType | undefined>(undefined);

export const useAudio = () => {
  const context = useContext(AudioContext);
  if (!context) {
    throw new Error('useAudio must be used within AudioProvider');
  }
  return context;
};

export const AudioProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Load saved volumes from localStorage or use defaults
  const [musicVolume, setMusicVolumeState] = useState(() => {
    const saved = localStorage.getItem('musicVolume');
    return saved ? parseFloat(saved) : 0.3; // Default 30%
  });

  const [soundVolume, setSoundVolumeState] = useState(() => {
    const saved = localStorage.getItem('soundVolume');
    return saved ? parseFloat(saved) : 0.5; // Default 50%
  });

  // Audio element refs
  const musicRef = useRef<HTMLAudioElement | null>(null);
  const clickSoundsRef = useRef<HTMLAudioElement[]>([]);
  const errorSoundRef = useRef<HTMLAudioElement | null>(null);
  const isMutedTemporarilyRef = useRef(false);

  // Initialize audio elements
  useEffect(() => {
    // Background music
    musicRef.current = new Audio(backgroundMusic);
    musicRef.current.loop = true;
    musicRef.current.volume = musicVolume;

    // Click sounds (create 3 instances for random selection)
    clickSoundsRef.current = [
      new Audio(click1),
      new Audio(click2),
      new Audio(click3),
    ];
    clickSoundsRef.current.forEach(audio => {
      audio.volume = soundVolume;
    });

    // Error sound
    errorSoundRef.current = new Audio(errorSound);
    errorSoundRef.current.volume = soundVolume;

    // Start playing background music
    const playMusic = async () => {
      try {
        await musicRef.current?.play();
      } catch (error) {
        console.log('Music autoplay blocked - will play on first user interaction');
      }
    };
    playMusic();

    // Cleanup on unmount
    return () => {
      musicRef.current?.pause();
      musicRef.current = null;
      clickSoundsRef.current = [];
      errorSoundRef.current = null;
    };
  }, []);

  // Update music volume
  useEffect(() => {
    if (musicRef.current) {
      musicRef.current.volume = isMutedTemporarilyRef.current ? 0 : musicVolume;
    }
    localStorage.setItem('musicVolume', musicVolume.toString());
  }, [musicVolume]);

  // Update sound effects volume
  useEffect(() => {
    clickSoundsRef.current.forEach(audio => {
      audio.volume = soundVolume;
    });
    if (errorSoundRef.current) {
      errorSoundRef.current.volume = soundVolume;
    }
    localStorage.setItem('soundVolume', soundVolume.toString());
  }, [soundVolume]);

  const setMusicVolume = (volume: number) => {
    setMusicVolumeState(Math.max(0, Math.min(1, volume)));
  };

  const setSoundVolume = (volume: number) => {
    setSoundVolumeState(Math.max(0, Math.min(1, volume)));
  };

  const playCorrectSound = () => {
    // Randomly select one of the three click sounds
    const randomIndex = Math.floor(Math.random() * 3);
    const sound = clickSoundsRef.current[randomIndex];

    if (sound) {
      sound.currentTime = 0; // Reset to start
      sound.play().catch(err => console.log('Sound play failed:', err));
    }
  };

  const playErrorSound = () => {
    if (errorSoundRef.current) {
      errorSoundRef.current.currentTime = 0;
      errorSoundRef.current.play().catch(err => console.log('Error sound play failed:', err));
    }
  };

  const muteMusicTemporarily = (mute: boolean) => {
    isMutedTemporarilyRef.current = mute;
    if (musicRef.current) {
      musicRef.current.volume = mute ? 0 : musicVolume;
    }
  };

  return (
    <AudioContext.Provider
      value={{
        musicVolume,
        soundVolume,
        setMusicVolume,
        setSoundVolume,
        playCorrectSound,
        playErrorSound,
        muteMusicTemporarily,
      }}
    >
      {children}
    </AudioContext.Provider>
  );
};
