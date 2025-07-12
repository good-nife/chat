"""
TextToSpeech Library - A flexible Python library for converting text to speech
"""

import os
import sys
import tempfile
import threading
import queue
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSEngine(ABC):
    """Abstract base class for TTS engines"""
    
    @abstractmethod
    def speak(self, text: str, **kwargs) -> bool:
        """Speak text aloud"""
        pass
    
    @abstractmethod
    def save_to_file(self, text: str, filepath: str, **kwargs) -> bool:
        """Save text as audio file"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine is available on system"""
        pass

class PyttxEngine(TTSEngine):
    """pyttsx3 engine wrapper"""
    
    def __init__(self):
        self.engine = None
        self._init_engine()
    
    def _init_engine(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            logger.info("pyttsx3 engine initialized")
        except ImportError:
            logger.warning("pyttsx3 not available. Install with: pip install pyttsx3")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
    
    def speak(self, text: str, **kwargs) -> bool:
        if not self.engine:
            return False
        
        try:
            # Set properties if provided
            if 'rate' in kwargs:
                self.engine.setProperty('rate', kwargs['rate'])
            if 'volume' in kwargs:
                self.engine.setProperty('volume', kwargs['volume'])
            if 'voice' in kwargs:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if kwargs['voice'].lower() in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Error speaking with pyttsx3: {e}")
            return False
    
    def save_to_file(self, text: str, filepath: str, **kwargs) -> bool:
        if not self.engine:
            return False
        
        try:
            # Set properties if provided
            if 'rate' in kwargs:
                self.engine.setProperty('rate', kwargs['rate'])
            if 'volume' in kwargs:
                self.engine.setProperty('volume', kwargs['volume'])
            
            self.engine.save_to_file(text, filepath)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Error saving to file with pyttsx3: {e}")
            return False
    
    def is_available(self) -> bool:
        return self.engine is not None

class GTTSEngine(TTSEngine):
    """Google Text-to-Speech engine wrapper"""
    
    def __init__(self):
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        try:
            import gtts
            import pygame
            return True
        except ImportError:
            logger.warning("gTTS or pygame not available. Install with: pip install gtts pygame")
            return False
    
    def speak(self, text: str, **kwargs) -> bool:
        if not self.available:
            return False
        
        try:
            import gtts
            import pygame
            import time
            
            lang = kwargs.get('lang', 'en')
            slow = kwargs.get('slow', False)
            
            tts = gtts.gTTS(text=text, lang=lang, slow=slow)
            
            # Create temp file with a more explicit approach
            temp_dir = tempfile.gettempdir()
            temp_filename = f"tts_temp_{int(time.time() * 1000)}.mp3"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            try:
                # Save the audio file
                tts.save(temp_path)
                
                # Verify file exists and has content
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    raise Exception("Failed to create audio file")
                
                # Initialize pygame mixer
                pygame.mixer.quit()  # Ensure clean state
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                # Load and play the audio file
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                # Small delay to ensure playback is complete
                time.sleep(0.1)
                
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass  # Ignore cleanup errors
            
            return True
        except Exception as e:
            logger.error(f"Error speaking with gTTS: {e}")
            return False
    
    def save_to_file(self, text: str, filepath: str, **kwargs) -> bool:
        if not self.available:
            return False
        
        try:
            import gtts
            
            lang = kwargs.get('lang', 'en')
            slow = kwargs.get('slow', False)
            
            tts = gtts.gTTS(text=text, lang=lang, slow=slow)
            tts.save(filepath)
            return True
        except Exception as e:
            logger.error(f"Error saving to file with gTTS: {e}")
            return False
    
    def is_available(self) -> bool:
        return self.available

class SystemTTSEngine(TTSEngine):
    """System-specific TTS engine (espeak, say, SAPI)"""
    
    def __init__(self):
        self.command = self._detect_system_tts()
    
    def _detect_system_tts(self) -> Optional[str]:
        """Detect available system TTS command"""
        if sys.platform == "darwin":  # macOS
            return "say"
        elif sys.platform.startswith("linux"):  # Linux
            # Check for espeak
            if os.system("which espeak > /dev/null 2>&1") == 0:
                return "espeak"
            elif os.system("which festival > /dev/null 2>&1") == 0:
                return "festival"
        elif sys.platform.startswith("win"):  # Windows
            # Windows has built-in SAPI
            return "powershell"
        return None
    
    def speak(self, text: str, **kwargs) -> bool:
        if not self.command:
            return False
        
        try:
            if self.command == "say":  # macOS
                voice = kwargs.get('voice', '')
                rate = kwargs.get('rate', 200)
                cmd = f'say -r {rate}'
                if voice:
                    cmd += f' -v "{voice}"'
                cmd += f' "{text}"'
                
            elif self.command == "espeak":  # Linux
                speed = kwargs.get('speed', 175)
                pitch = kwargs.get('pitch', 50)
                cmd = f'espeak -s {speed} -p {pitch} "{text}"'
                
            elif self.command == "festival":  # Linux
                cmd = f'echo "{text}" | festival --tts'
                
            elif self.command == "powershell":  # Windows
                cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak(\'{text}\')"'
            
            return os.system(cmd) == 0
        except Exception as e:
            logger.error(f"Error with system TTS: {e}")
            return False
    
    def save_to_file(self, text: str, filepath: str, **kwargs) -> bool:
        if not self.command:
            return False
        
        try:
            if self.command == "say":  # macOS
                voice = kwargs.get('voice', '')
                rate = kwargs.get('rate', 200)
                cmd = f'say -r {rate}'
                if voice:
                    cmd += f' -v "{voice}"'
                cmd += f' -o "{filepath}" "{text}"'
                
            elif self.command == "espeak":  # Linux
                speed = kwargs.get('speed', 175)
                pitch = kwargs.get('pitch', 50)
                cmd = f'espeak -s {speed} -p {pitch} -w "{filepath}" "{text}"'
                
            else:
                # For other systems, we'd need additional implementation
                return False
            
            return os.system(cmd) == 0
        except Exception as e:
            logger.error(f"Error saving with system TTS: {e}")
            return False
    
    def is_available(self) -> bool:
        return self.command is not None

class TextToSpeech:
    """Main TTS class that manages multiple engines"""
    
    def __init__(self, preferred_engine: Optional[str] = None):
        self.engines = {
            'gtts': GTTSEngine(),
            'pyttsx3': PyttxEngine(),
            'system': SystemTTSEngine()
        }
        
        self.preferred_engine = preferred_engine
        self.active_engine = self._get_best_engine()
        
        # Queue for async operations
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
    
    def _get_best_engine(self) -> TTSEngine:
        """Get the best available engine"""
        if self.preferred_engine and self.preferred_engine in self.engines:
            engine = self.engines[self.preferred_engine]
            if engine.is_available():
                logger.info(f"Using preferred engine: {self.preferred_engine}")
                return engine
        
        # Try engines in order of preference
        for name, engine in self.engines.items():
            if engine.is_available():
                logger.info(f"Using engine: {name}")
                return engine
        
        raise RuntimeError("No TTS engine available")
    
    def speak(self, text: str, **kwargs) -> bool:
        """Speak text using the active engine"""
        if not text.strip():
            return False
        
        return self.active_engine.speak(text, **kwargs)
    
    def save_to_file(self, text: str, filepath: str, **kwargs) -> bool:
        """Save text as audio file"""
        if not text.strip():
            return False
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        return self.active_engine.save_to_file(text, filepath, **kwargs)
    
    def speak_async(self, text: str, **kwargs):
        """Add text to speech queue for async processing"""
        self.speech_queue.put((text, kwargs))
        
        if not self.is_running:
            self._start_worker()
    
    def _start_worker(self):
        """Start the background worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Background worker for async speech"""
        while self.is_running:
            try:
                text, kwargs = self.speech_queue.get(timeout=1)
                self.speak(text, **kwargs)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in speech worker: {e}")
    
    def stop_async(self):
        """Stop async speech processing"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engines"""
        return [name for name, engine in self.engines.items() if engine.is_available()]
    
    def set_engine(self, engine_name: str) -> bool:
        """Switch to a different engine"""
        if engine_name not in self.engines:
            return False
        
        engine = self.engines[engine_name]
        if not engine.is_available():
            return False
        
        self.active_engine = engine
        self.preferred_engine = engine_name
        logger.info(f"Switched to engine: {engine_name}")
        return True
    
    def get_voices(self) -> List[str]:
        """Get available voices (if supported by engine)"""
        if hasattr(self.active_engine, 'engine') and self.active_engine.engine:
            try:
                voices = self.active_engine.engine.getProperty('voices')
                return [voice.name for voice in voices] if voices else []
            except:
                pass
        return []

# Convenience functions
def speak(text: str, engine: Optional[str] = None, **kwargs) -> bool:
    """Quick function to speak text"""
    tts = TextToSpeech(preferred_engine=engine)
    return tts.speak(text, **kwargs)

def save_speech(text: str, filepath: str, engine: Optional[str] = None, **kwargs) -> bool:
    """Quick function to save speech to file"""
    tts = TextToSpeech(preferred_engine=engine)
    return tts.save_to_file(text, filepath, **kwargs)

# Example usage
if __name__ == "__main__":
    # Initialize TTS
    tts = TextToSpeech()
    
    print("Available engines:", tts.get_available_engines())
    print("Available voices:", tts.get_voices())
    
    # Test speaking
    print("Testing speech...")
    tts.speak("Hello! This is a test of the text to speech library.")
    
    # Test saving to file
    print("Saving to file...")
    tts.save_to_file("This will be saved as an audio file.", "output.wav")
    
    # Test async speech
    print("Testing async speech...")
    tts.speak_async("This is async speech number 1")
    tts.speak_async("This is async speech number 2")
    
    # Wait for async operations to complete
    import time
    time.sleep(3)
    tts.stop_async()
    
    print("Done!")