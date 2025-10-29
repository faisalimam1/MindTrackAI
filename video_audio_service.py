#!/usr/bin/env python3
"""
Video and Audio Analysis Service for Mental Health Assessment
Analyzes facial expressions, voice patterns, and speech for depression/confidence detection

UPDATED VERSION with:
- Better audio format conversion (FFmpeg support)
- Improved pitch detection (pYIN algorithm)
- Enhanced speech recognition (Whisper support)
- Clinical-grade depression scoring
- Proper preprocessing and validation
"""

from __future__ import annotations

import os
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FFmpeg configuration with auto-detection
def find_ffmpeg():
    """
    Find FFmpeg executable using multiple methods:
    1. Environment variable FFMPEG_PATH
    2. System PATH
    3. Common installation locations
    """
    # Check environment variable first
    ffmpeg_path = os.getenv('FFMPEG_PATH')
    if ffmpeg_path and os.path.isfile(ffmpeg_path):
        return ffmpeg_path

    # Check if ffmpeg is in system PATH
    ffmpeg_in_path = shutil.which('ffmpeg')
    if ffmpeg_in_path:
        return ffmpeg_in_path

    # Check common installation locations (Windows)
    common_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
    ]
    for path in common_paths:
        if os.path.isfile(path):
            return path

    return None

def find_ffprobe():
    """Find FFprobe executable"""
    ffprobe_path = os.getenv('FFPROBE_PATH')
    if ffprobe_path and os.path.isfile(ffprobe_path):
        return ffprobe_path

    ffprobe_in_path = shutil.which('ffprobe')
    if ffprobe_in_path:
        return ffprobe_in_path

    common_paths = [
        r'C:\ffmpeg\bin\ffprobe.exe',
        r'C:\Program Files\ffmpeg\bin\ffprobe.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe',
    ]
    for path in common_paths:
        if os.path.isfile(path):
            return path

    return None

# Detect FFmpeg
FFMPEG_PATH = find_ffmpeg()
FFPROBE_PATH = find_ffprobe()
FFMPEG_AVAILABLE = FFMPEG_PATH is not None

if FFMPEG_AVAILABLE:
    # Add FFmpeg directory to PATH for pydub
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    print(f"FFmpeg found at: {FFMPEG_PATH}")
else:
    print("Warning: FFmpeg not found. Audio conversion will be limited. Please install FFmpeg or set FFMPEG_PATH environment variable.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Video analysis will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # Define np as None for type hints when numpy not available
    print("Warning: NumPy not available. Analysis will be limited.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: Librosa not available. Audio analysis will be limited.")

try:
    from moviepy.editor import AudioFileClip
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Video audio extraction will be limited.")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Warning: SpeechRecognition not available. Speech transcription will be limited.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Sentiment analysis will be limited.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Advanced features disabled.")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False
    print("Info: DeepFace not available.")

try:
    from fer import FER
    FER_AVAILABLE = True
except Exception:
    FER_AVAILABLE = False
    print("Info: FER not available.")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Info: noisereduce not available. Audio preprocessing limited.")

from datetime import datetime
from typing import Dict, List, Tuple, TYPE_CHECKING, Any
import logging
import tempfile
import os
import subprocess

# TYPE_CHECKING allows type hints to work even if imports fail at runtime
if TYPE_CHECKING:
    import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoAudioAnalysisService:
    """Service for analyzing video and audio data for mental health indicators."""
    
    def __init__(self):
        """Initialize the analysis service with ML models."""
        self.face_cascade = None
        self.emotion_model = None
        self.speech_recognizer = None
        if SPEECH_RECOGNITION_AVAILABLE:
            self.speech_recognizer = sr.Recognizer()
        self._load_models()
    
    def _load_models(self):
        """Load required ML models for analysis."""
        try:
            # Load OpenCV face detection if available
            if CV2_AVAILABLE:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                logger.info("Face detection model loaded successfully")
            else:
                logger.warning("OpenCV not available - face detection disabled")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def analyze_video_audio(self, video_data: bytes, audio_data: bytes) -> Dict:
        """
        Analyze both video and audio data for mental health indicators.
        
        Args:
            video_data: Raw video data (60 seconds)
            audio_data: Raw audio data (60 seconds)
            
        Returns:
            Dict containing analysis results
        """
        try:
            logger.info(f"Starting video/audio analysis - Video size: {len(video_data)} bytes, Audio size: {len(audio_data)} bytes")
            
            # Validate input data
            if not video_data or len(video_data) < 1000:
                logger.warning("Video data is too small or empty")
                video_analysis = self._get_fallback_video_analysis()
            else:
                # Analyze video for facial expressions
                video_analysis = self._analyze_video_emotions(video_data)
                logger.info(f"Video analysis completed - Depression: {video_analysis.get('depression_indicators', {}).get('score', 'N/A')}")
            
            if not audio_data or len(audio_data) < 1000:
                logger.warning("Audio data is too small or empty")
                audio_analysis = self._get_fallback_audio_analysis()
            else:
                # Analyze audio for voice patterns and speech
                audio_analysis = self._analyze_audio_sentiment(audio_data)
                logger.info(f"Audio analysis completed - Depression: {audio_analysis.get('depression_indicators', {}).get('score', 'N/A')}")
            
            # Combine results for comprehensive assessment
            combined_assessment = self._combine_assessments(video_analysis, audio_analysis)
            logger.info(f"Combined assessment - Depression: {combined_assessment.get('depression_score', 'N/A')}, Confidence: {combined_assessment.get('confidence_score', 'N/A')}")
            
            combined = {
                'timestamp': datetime.now().isoformat(),
                'assessment_type': 'video_audio',
                'video_analysis': video_analysis,
                'audio_analysis': audio_analysis,
                'combined_assessment': combined_assessment,
                'confidence_score': combined_assessment.get('confidence', 0.5),
                'recommendations': self._generate_recommendations(combined_assessment),
                'processing_info': {
                    'video_processed': len(video_data) >= 1000,
                    'audio_processed': len(audio_data) >= 1000,
                    'libraries_available': {
                        'opencv': CV2_AVAILABLE,
                        'librosa': LIBROSA_AVAILABLE,
                        'moviepy': MOVIEPY_AVAILABLE,
                        'speech_recognition': SPEECH_RECOGNITION_AVAILABLE,
                        'textblob': TEXTBLOB_AVAILABLE,
                        'noisereduce': NOISEREDUCE_AVAILABLE
                    }
                }
            }
            
            # Flatten key combined metrics for frontend consumption
            combined.update({
                'depression_score': combined_assessment.get('depression_score', 0.5),
                'depression_level': combined_assessment.get('depression_level', 'moderate'),
                'confidence_score': combined_assessment.get('confidence_score', combined_assessment.get('confidence', 0.5)),
                'confidence_level': combined_assessment.get('confidence_level', 'moderate'),
                'overall_wellbeing': combined_assessment.get('overall_wellbeing', 'moderate')
            })
            
            logger.info("Video/audio analysis completed successfully")
            return combined
            
        except Exception as e:
            logger.error(f"Error in video/audio analysis: {e}", exc_info=True)
            return self._get_fallback_assessment('video_audio')
    
    def analyze_audio_only(self, audio_data: bytes) -> Dict:
        """
        Analyze audio-only data for mental health indicators.
        
        Args:
            audio_data: Raw audio data (60 seconds)
            
        Returns:
            Dict containing analysis results
        """
        try:
            logger.info(f"Starting audio-only analysis - Audio size: {len(audio_data)} bytes")
            
            # Validate audio data
            if not audio_data or len(audio_data) < 1000:
                logger.warning("Audio data is too small or empty")
                return self._get_fallback_assessment('audio_only')
            
            # Analyze audio for voice patterns and speech
            audio_analysis = self._analyze_audio_sentiment(audio_data)
            
            # Generate assessment based on audio only
            audio_assessment = self._generate_audio_assessment(audio_analysis)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'assessment_type': 'audio_only',
                'audio_analysis': audio_analysis,
                'assessment': audio_assessment,
                'confidence_score': audio_assessment.get('confidence', 0.5),
                'recommendations': self._generate_recommendations(audio_assessment)
            }
            
            # Flatten key metrics from audio-only assessment
            result.update({
                'depression_score': audio_assessment.get('depression_score', 0.5),
                'depression_level': audio_assessment.get('depression_level', 'moderate'),
                'confidence_score': audio_assessment.get('confidence_score', 0.5),
                'confidence_level': audio_assessment.get('confidence_level', 'moderate'),
                'overall_wellbeing': audio_assessment.get('overall_wellbeing', 'moderate')
            })
            
            logger.info("Audio-only analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}", exc_info=True)
            return self._get_fallback_assessment('audio_only')
    
    def _preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio to improve feature extraction quality.
        
        Steps:
        - Noise reduction
        - Normalization
        - Pre-emphasis filtering
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Preprocessed audio signal
        """
        try:
            # 1. Noise reduction
            if NOISEREDUCE_AVAILABLE:
                try:
                    y_clean = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.8)
                    logger.info("Noise reduction applied")
                except Exception as e:
                    logger.warning(f"Noise reduction failed: {e}, using raw audio")
                    y_clean = y
            else:
                y_clean = y
            
            # 2. Normalize amplitude
            max_val = np.max(np.abs(y_clean))
            if max_val > 0:
                y_clean = y_clean / max_val
                logger.info(f"Audio normalized (max: {max_val:.3f})")
            else:
                logger.warning("Audio is silent or has zero amplitude")
            
            # 3. Apply pre-emphasis filter (boost high frequencies)
            # This helps with pitch detection and reduces noise
            pre_emphasis = 0.97
            y_clean = np.append(y_clean[0], y_clean[1:] - pre_emphasis * y_clean[:-1])
            
            return y_clean
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}, using raw audio")
            return y
    
    def _write_audio_wav(self, audio_bytes: bytes, out_wav_path: str) -> None:
        """
        Convert browser recording bytes to WAV format using multiple methods.

        Priority order:
        1. FFmpeg (most reliable)
        2. Librosa + soundfile (direct conversion)
        3. MoviePy
        4. Pydub
        5. Raw copy (last resort)

        Args:
            audio_bytes: Raw audio data from browser
            out_wav_path: Output WAV file path
        """
        tmp_in = None
        conversion_successful = False

        try:
            # Save raw bytes to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as f:
                f.write(audio_bytes)
                f.flush()
                tmp_in = f.name

            logger.info(f"Temp input file: {tmp_in} ({len(audio_bytes)} bytes)")

            # Method 1: Try FFmpeg (most reliable for WebM/Opus conversion)
            if FFMPEG_AVAILABLE and FFMPEG_PATH:
                try:
                    result = subprocess.run([
                        FFMPEG_PATH, '-loglevel', 'error', '-i', tmp_in,
                        '-acodec', 'pcm_s16le',  # 16-bit PCM
                        '-ar', '16000',           # 16kHz sample rate
                        '-ac', '1',               # Mono
                        '-y',                     # Overwrite
                        out_wav_path
                    ], check=True, capture_output=True, timeout=30, text=True)

                    # Verify output file
                    if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1000:
                        logger.info(f"✓ FFmpeg conversion successful: {os.path.getsize(out_wav_path)} bytes")
                        conversion_successful = True
                        return
                    else:
                        logger.warning("FFmpeg output file invalid or too small")

                except subprocess.CalledProcessError as e:
                    logger.warning(f"FFmpeg failed: {e.stderr if e.stderr else 'Unknown error'}")
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg conversion timed out")
                except Exception as e:
                    logger.warning(f"FFmpeg error: {e}")
            else:
                logger.info("FFmpeg not available, skipping to alternative methods")
            
            # Method 2: Try Librosa + soundfile (direct conversion)
            if LIBROSA_AVAILABLE and not conversion_successful:
                try:
                    import soundfile as sf
                    logger.info("Trying librosa + soundfile conversion...")
                    y, sr = librosa.load(tmp_in, sr=16000, mono=True)

                    if len(y) > 0:
                        sf.write(out_wav_path, y, 16000, subtype='PCM_16')

                        if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1000:
                            logger.info(f"✓ Librosa conversion successful: {os.path.getsize(out_wav_path)} bytes")
                            conversion_successful = True
                            return

                except Exception as e:
                    logger.warning(f"Librosa conversion failed: {e}")

            # Method 3: Try Pydub
            if not conversion_successful:
                try:
                    from pydub import AudioSegment
                    logger.info("Trying pydub conversion...")

                    # Pydub can handle WebM
                    audio = AudioSegment.from_file(tmp_in)
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    audio.export(out_wav_path, format='wav')

                    if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1000:
                        logger.info(f"✓ Pydub conversion successful: {os.path.getsize(out_wav_path)} bytes")
                        conversion_successful = True
                        return

                except Exception as e:
                    logger.warning(f"Pydub conversion failed: {e}")

            # Method 4: Try MoviePy
            if MOVIEPY_AVAILABLE and not conversion_successful:
                try:
                    logger.info("Trying MoviePy conversion...")
                    audio_clip = AudioFileClip(tmp_in)
                    audio_clip.write_audiofile(
                        out_wav_path,
                        fps=16000,
                        nbytes=2,
                        codec='pcm_s16le',
                        verbose=False,
                        logger=None
                    )
                    audio_clip.close()

                    if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1000:
                        logger.info(f"✓ MoviePy conversion successful: {os.path.getsize(out_wav_path)} bytes")
                        conversion_successful = True
                        return

                except Exception as e:
                    logger.warning(f"MoviePy conversion failed: {e}")

            # If all methods failed
            if not conversion_successful:
                logger.error("❌ All conversion methods failed! Audio analysis will use fallback values.")
                # Create an empty WAV file with proper headers to prevent crashes
                import wave
                with wave.open(out_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(b'\x00' * 16000)  # 1 second of silence

        except Exception as e:
            logger.error(f"Fatal error in audio conversion: {e}", exc_info=True)
            # Write minimal WAV file to prevent crashes downstream
            try:
                import wave
                with wave.open(out_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(b'\x00' * 16000)
            except Exception:
                pass
                
        finally:
            # Cleanup temp input file
            if tmp_in and os.path.exists(tmp_in):
                try:
                    os.remove(tmp_in)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
    
    def _extract_audio_features(self, audio_data: bytes) -> Dict:
        """
        Extract comprehensive audio features for voice pattern analysis.
        
        Features extracted:
        - Pitch (F0) statistics using pYIN algorithm
        - Energy (RMS) statistics
        - Speaking rate and pause frequency
        - Spectral features (centroid, rolloff)
        - Voice quality (jitter, shimmer)
        - Zero-crossing rate
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Dict of normalized audio features
        """
        try:
            if not LIBROSA_AVAILABLE or not NUMPY_AVAILABLE:
                logger.warning("Librosa/NumPy not available")
                return self._get_fallback_audio_features()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                try:
                    # Convert to WAV
                    self._write_audio_wav(audio_data, tmp.name)
                    
                    # Validate file exists and has content
                    if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) < 5000:
                        logger.error(f"Invalid audio file for feature extraction (size: {os.path.getsize(tmp.name) if os.path.exists(tmp.name) else 0} bytes)")
                        return self._get_fallback_audio_features()
                    
                    # Load audio with librosa
                    try:
                        y, sr = librosa.load(tmp.name, sr=16000, mono=True)
                    except Exception as e:
                        logger.error(f"Librosa load failed: {e}")
                        return self._get_fallback_audio_features()
                    
                    # Check for silent or very quiet audio
                    max_amplitude = np.max(np.abs(y))
                    if max_amplitude < 0.01:
                        logger.warning(f"Audio is too quiet or silent (max amplitude: {max_amplitude})")
                        return {
                            'pitch_std': 0.1,
                            'pitch_mean': 0.1,
                            'pitch_cv': 0.1,
                            'energy_mean': 0.1,
                            'energy_std': 0.05,
                            'energy_max': 0.1,
                            'speaking_rate': 0.2,
                            'pause_frequency': 0.9,
                            'spectral_centroid_mean': 0.3,
                            'spectral_rolloff_mean': 0.3,
                            'zcr_mean': 0.2,
                            'jitter': 0.5,
                            'shimmer': 0.5,
                            'is_silent': True,
                            'total_duration': len(y) / sr,
                            'speech_duration': 0.0
                        }
                    
                    # Preprocess audio
                    y = self._preprocess_audio(y, sr)
                    
                    logger.info(f"Audio loaded: {len(y)} samples, {sr} Hz, duration: {len(y)/sr:.2f}s")
                    
                    # ===== PITCH EXTRACTION (F0) =====
                    # Use pYIN algorithm (more robust than piptrack)
                    try:
                        f0, voiced_flag, voiced_probs = librosa.pyin(
                            y,
                            fmin=librosa.note_to_hz('C2'),  # ~65 Hz (male voice low)
                            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (female voice high)
                            sr=sr,
                            frame_length=2048
                        )
                        
                        # Filter out unvoiced frames
                        pitch_values = f0[voiced_flag & ~np.isnan(f0)]
                        
                        if len(pitch_values) > 5:
                            pitch_std = np.std(pitch_values)
                            pitch_mean = np.mean(pitch_values)
                            pitch_cv = pitch_std / (pitch_mean + 1e-8)  # Coefficient of variation
                            logger.info(f"pYIN pitch: mean={pitch_mean:.1f}Hz, std={pitch_std:.1f}Hz, cv={pitch_cv:.3f}")
                        else:
                            pitch_std = 0.0
                            pitch_mean = 0.0
                            pitch_cv = 0.0
                            logger.warning("Insufficient pitch values from pYIN")
                            
                    except Exception as e:
                        logger.warning(f"pYIN pitch extraction failed: {e}, trying piptrack")
                        
                        # Fallback to piptrack
                        try:
                            pitches, magnitudes = librosa.piptrack(
                                y=y, sr=sr,
                                threshold=0.1,
                                fmin=65,
                                fmax=2093
                            )
                            
                            pitch_values = []
                            for t in range(pitches.shape[1]):
                                index = magnitudes[:, t].argmax()
                                pitch = pitches[index, t]
                                if pitch > 0:
                                    pitch_values.append(pitch)
                            
                            if pitch_values:
                                pitch_std = np.std(pitch_values)
                                pitch_mean = np.mean(pitch_values)
                                pitch_cv = pitch_std / (pitch_mean + 1e-8)
                                logger.info(f"piptrack pitch: mean={pitch_mean:.1f}Hz, std={pitch_std:.1f}Hz")
                            else:
                                pitch_std = 0.0
                                pitch_mean = 0.0
                                pitch_cv = 0.0
                                logger.warning("No pitch detected with piptrack")
                        except Exception as e2:
                            logger.error(f"Both pitch extraction methods failed: {e2}")
                            pitch_std = 0.0
                            pitch_mean = 0.0
                            pitch_cv = 0.0
                    
                    # ===== ENERGY ANALYSIS =====
                    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
                    energy_mean = np.mean(energy)
                    energy_std = np.std(energy)
                    energy_max = np.max(energy)
                    
                    # ===== SPECTRAL FEATURES =====
                    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
                    zcr = librosa.feature.zero_crossing_rate(y)[0]  # Voice activity indicator
                    
                    # ===== SPEAKING RATE & PAUSES =====
                    # Adaptive energy threshold based on audio characteristics
                    energy_threshold = np.percentile(energy, 30)  # 30th percentile
                    speech_frames = energy > energy_threshold
                    
                    # Calculate speaking rate
                    hop_length = 512
                    frame_duration = hop_length / sr
                    speech_duration = np.sum(speech_frames) * frame_duration
                    total_duration = len(y) / sr
                    speaking_rate = speech_duration / total_duration if total_duration > 0 else 0.0
                    
                    # Calculate pause frequency
                    pause_threshold = np.percentile(energy, 15)  # Lower threshold for pauses
                    pauses = energy < pause_threshold
                    
                    # Count pause transitions (silence -> speech)
                    pause_transitions = np.sum(np.diff(pauses.astype(int)) == 1)
                    pause_frequency = pause_transitions / total_duration if total_duration > 0 else 0.0
                    
                    # ===== ADVANCED VOICE QUALITY FEATURES =====
                    # Jitter (pitch variation between consecutive frames)
                    if len(pitch_values) > 1:
                        pitch_diffs = np.abs(np.diff(pitch_values))
                        jitter = np.mean(pitch_diffs) / (pitch_mean + 1e-8)
                    else:
                        jitter = 0.0
                    
                    # Shimmer (amplitude variation)
                    if len(energy) > 1:
                        energy_diffs = np.abs(np.diff(energy))
                        shimmer = np.mean(energy_diffs) / (energy_mean + 1e-8)
                    else:
                        shimmer = 0.0
                    
                    # ===== NORMALIZE FEATURES =====
                    # Calibrated to actual human voice ranges
                    normalized_features = {
                        # Pitch features (calibrated to human voice)
                        'pitch_std': min(pitch_std / 50.0, 1.0),  # 50 Hz std is high variation
                        'pitch_mean': min(max(pitch_mean / 200.0, 0.0), 1.0),  # 200 Hz average
                        'pitch_cv': min(pitch_cv / 0.3, 1.0),  # Coefficient of variation
                        
                        # Energy features
                        'energy_mean': min(energy_mean * 20.0, 1.0),  # Adjusted scaling
                        'energy_std': min(energy_std * 20.0, 1.0),
                        'energy_max': min(energy_max * 10.0, 1.0),
                        
                        # Speaking patterns
                        'speaking_rate': min(speaking_rate, 1.0),
                        'pause_frequency': min(pause_frequency * 2.0, 1.0),  # Scale up for sensitivity
                        
                        # Spectral features
                        'spectral_centroid_mean': min(np.mean(spectral_centroids) / 3000.0, 1.0),
                        'spectral_rolloff_mean': min(np.mean(spectral_rolloff) / 6000.0, 1.0),
                        'zcr_mean': min(np.mean(zcr) * 10.0, 1.0),
                        
                        # Advanced features
                        'jitter': min(jitter * 5.0, 1.0),
                        'shimmer': min(shimmer * 5.0, 1.0),
                        
                        # Metadata
                        'is_silent': False,
                        'total_duration': total_duration,
                        'speech_duration': speech_duration
                    }
                    
                    logger.info(f"Extracted features: pitch={pitch_mean:.1f}Hz (std={pitch_std:.1f}), "
                              f"energy={energy_mean:.3f}, rate={speaking_rate:.2f}, pauses={pause_frequency:.2f}")
                    
                    return normalized_features
                    
                except Exception as e:
                    logger.error(f"Feature extraction failed: {e}", exc_info=True)
                    return self._get_fallback_audio_features()
                finally:
                    try:
                        if os.path.exists(tmp.name):
                            os.unlink(tmp.name)
                    except (OSError, PermissionError):
                        pass
                        
        except Exception as e:
            logger.error(f"Fatal feature extraction error: {e}")
            return self._get_fallback_audio_features()
    
    def _get_fallback_audio_features(self) -> Dict:
        """
        Fallback features when extraction fails.

        NOTE: These are neutral baseline values that should trigger
        further manual review in the frontend.
        """
        logger.warning("⚠️  Using fallback audio features - extraction failed")
        logger.warning("⚠️  Results may not be accurate - please try re-recording")

        return {
            'pitch_std': 0.5,
            'pitch_mean': 0.5,
            'pitch_cv': 0.5,
            'energy_mean': 0.5,
            'energy_std': 0.5,
            'energy_max': 0.5,
            'speaking_rate': 0.5,
            'pause_frequency': 0.5,
            'spectral_centroid_mean': 0.5,
            'spectral_rolloff_mean': 0.5,
            'zcr_mean': 0.5,
            'jitter': 0.5,
            'shimmer': 0.5,
            'is_silent': False,
            'extraction_failed': True,
            'extraction_error': 'Audio conversion or feature extraction failed'
        }
    
    def _transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio to text using multiple speech recognition services.
        
        Priority order:
        1. Google Speech Recognition
        2. OpenAI Whisper (if available)
        3. Sphinx (offline fallback)
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text
        """
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                logger.warning("Speech recognition not available - skipping transcription")
                return ""  # Return empty string instead of error message

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                try:
                    # Convert to WAV
                    self._write_audio_wav(audio_data, tmp.name)

                    # Verify WAV file is valid
                    if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) < 1000:
                        logger.error("Invalid WAV file generated for transcription")
                        return ""  # Return empty string instead of error message
                    
                    logger.info(f"Transcribing audio file: {tmp.name} ({os.path.getsize(tmp.name)} bytes)")
                    
                    with sr.AudioFile(tmp.name) as source:
                        # Adjust for ambient noise and record
                        self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = self.speech_recognizer.record(source, duration=60)
                    
                    # Method 1: Try Google Speech Recognition (free, good accuracy)
                    try:
                        text = self.speech_recognizer.recognize_google(
                            audio,
                            language='en-US',
                            show_all=False
                        )
                        if text and len(text) > 3:
                            logger.info(f"Google transcription successful: {text[:100]}...")
                            return text
                    except sr.UnknownValueError:
                        logger.warning("Google couldn't understand audio")
                    except sr.RequestError as e:
                        logger.warning(f"Google API error: {e}")
                    
                    # Method 2: Try Whisper if available (OpenAI's model, very accurate)
                    if TRANSFORMERS_AVAILABLE:
                        try:
                            logger.info("Trying Whisper transcription...")
                            transcriber = pipeline(
                                "automatic-speech-recognition",
                                model="openai/whisper-tiny",  # Use tiny for speed
                                device=-1  # CPU
                            )
                            result = transcriber(tmp.name)
                            text = result.get('text', '')
                            if text and len(text) > 3:
                                logger.info(f"Whisper transcription successful: {text[:100]}...")
                                return text
                        except Exception as e:
                            logger.warning(f"Whisper transcription failed: {e}")
                    
                    # Method 3: Try Sphinx as last resort (offline but less accurate)
                    try:
                        logger.info("Trying Sphinx transcription...")
                        text = self.speech_recognizer.recognize_sphinx(audio)
                        if text and len(text) > 3:
                            logger.info(f"Sphinx transcription successful: {text[:100]}...")
                            return text
                    except sr.UnknownValueError:
                        logger.warning("Sphinx couldn't understand audio")
                    except Exception as e:
                        logger.warning(f"Sphinx error: {e}")
                    
                    # If all methods fail but audio seems valid
                    if os.path.getsize(tmp.name) > 10000:
                        logger.warning("All transcription methods failed for valid audio - will use voice features only")
                    else:
                        logger.warning("Could not understand audio clearly")

                    return ""  # Return empty string - analysis will rely on voice features only
                    
                except Exception as e:
                    logger.error(f"Transcription error: {e}", exc_info=True)
                    return ""  # Return empty string on error
                finally:
                    try:
                        if os.path.exists(tmp.name):
                            os.unlink(tmp.name)
                    except (OSError, PermissionError):
                        pass
                        
        except Exception as e:
            logger.error(f"Fatal transcription error: {e}")
            return ""  # Return empty string on fatal error
    
    def _analyze_speech_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of transcribed speech.
        
        Args:
            text: Transcribed text
            
        Returns:
            Dict with sentiment analysis results
        """
        try:
            if TEXTBLOB_AVAILABLE and text and len(text) > 3:
                blob = TextBlob(text)
                sentiment = blob.sentiment
            else:
                # Fallback sentiment analysis
                sentiment = type('obj', (object,), {'polarity': 0.0, 'subjectivity': 0.5})()
            
            return {
                'polarity': sentiment.polarity,  # -1 to 1
                'subjectivity': sentiment.subjectivity,  # 0 to 1
                'sentiment_label': self._get_sentiment_label(sentiment.polarity),
                'emotional_words': self._extract_emotional_words(text),
                'transcribed_text': text  # Add transcribed text for keyword detection
            }
        except Exception as e:
            logger.error(f"Error analyzing speech sentiment: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'sentiment_label': 'neutral',
                'emotional_words': [],
                'transcribed_text': ''  # Empty string on error
            }
    
    def _calculate_voice_depression_score(self, voice_analysis: Dict, sentiment_analysis: Dict) -> float:
        """
        Calculate depression score from voice patterns and speech sentiment.
        
        Based on clinical research showing depression manifests as:
        - Reduced pitch variability (monotone)
        - Lower pitch overall
        - Reduced energy/loudness
        - Slower speaking rate
        - More pauses/hesitations
        - Increased jitter/shimmer (voice quality deterioration)
        - Negative sentiment in speech
        
        Args:
            voice_analysis: Dict of voice features
            sentiment_analysis: Dict of sentiment features
            
        Returns:
            Depression score (0-1, higher = more depressed)
        """
        try:
            # Check if audio was silent or failed extraction
            if voice_analysis.get('is_silent', False):
                logger.error("Audio is too quiet or silent - cannot analyze")
                return None  # Return None to indicate failure

            if voice_analysis.get('extraction_failed', False):
                logger.error("Feature extraction failed - cannot analyze")
                return None  # Return None to indicate failure
            
            # ===== VOICE FEATURE INDICATORS =====
            pitch_std = voice_analysis.get('pitch_std', 0.5)
            pitch_cv = voice_analysis.get('pitch_cv', 0.5)
            energy_mean = voice_analysis.get('energy_mean', 0.5)
            energy_std = voice_analysis.get('energy_std', 0.5)
            speaking_rate = voice_analysis.get('speaking_rate', 0.5)
            pause_frequency = voice_analysis.get('pause_frequency', 0.5)
            spectral_centroid = voice_analysis.get('spectral_centroid_mean', 0.5)
            jitter = voice_analysis.get('jitter', 0.5)
            shimmer = voice_analysis.get('shimmer', 0.5)
            
            # ===== CLINICAL WEIGHT DISTRIBUTION =====
            # Based on meta-analysis of voice depression research
            voice_score = (
                (1 - pitch_std) * 0.18 +          # Reduced pitch variability (strong indicator)
                (1 - pitch_cv) * 0.12 +            # Monotone coefficient
                (1 - energy_mean) * 0.15 +         # Reduced vocal energy
                (1 - energy_std) * 0.08 +          # Flat energy profile
                pause_frequency * 0.15 +           # Hesitations/pauses
                (1 - speaking_rate) * 0.12 +       # Psychomotor retardation
                (1 - spectral_centroid) * 0.08 +   # Darker/duller voice tone
                jitter * 0.06 +                    # Voice quality deterioration
                shimmer * 0.06                     # Amplitude instability
            )
            
            # ===== SENTIMENT ANALYSIS CONTRIBUTION =====
            polarity = sentiment_analysis.get('polarity', 0)
            subjectivity = sentiment_analysis.get('subjectivity', 0.5)
            emotional_words = sentiment_analysis.get('emotional_words', [])

            # Get transcribed text for keyword detection
            transcribed_text = sentiment_analysis.get('transcribed_text', '').lower()

            # POSITIVE KEYWORD DETECTION (with negation context awareness)
            positive_keywords = [
                'happy', 'joy', 'joyful', 'love', 'wonderful', 'great', 'amazing',
                'excited', 'fantastic', 'birthday', 'celebration', 'party', 'fun',
                'good', 'excellent', 'awesome', 'beautiful', 'enjoying', 'glad',
                'blessed', 'grateful', 'thankful', 'proud', 'thrilled', 'delighted'
            ]

            # Context-aware positive keyword counting - exclude negated contexts
            negation_words = ['not', 'no', 'never', 'hardly', 'barely', 'nothing', 'neither']
            positive_keyword_count = 0

            for kw in positive_keywords:
                if kw in transcribed_text:
                    # Check if keyword appears in negative context
                    is_negated = False
                    kw_index = transcribed_text.find(kw)

                    # Get 20 characters before the keyword to check for negation
                    context_before = transcribed_text[max(0, kw_index - 20):kw_index].lower()

                    # Check if any negation word appears in the context
                    for neg_word in negation_words:
                        if neg_word in context_before.split():
                            is_negated = True
                            logger.info(f"Excluded positive keyword '{kw}' due to negation context: '{context_before.strip()} {kw}'")
                            break

                    # Only count if not negated
                    if not is_negated:
                        positive_keyword_count += 1

            # NEGATIVE/DEPRESSION KEYWORD DETECTION (Expanded)
            depression_keywords = [
                # Core depression words
                'sad', 'hopeless', 'worthless', 'tired', 'depressed', 'depression',
                'anxious', 'worried', 'alone', 'empty', 'numb', 'anxiety',
                'exhausted', 'helpless', 'miserable', 'unhappy', 'hopelessness',

                # Stress and pressure indicators
                'pressure', 'stressed', 'stress', 'overwhelming', 'overwhelmed',
                'burden', 'struggling', 'difficult', 'hard time', 'tough',

                # Negative mood states
                'not in a good mood', 'bad mood', 'terrible', 'awful',
                'frustrated', 'angry', 'upset', 'crying', 'cry',

                # Inability/helplessness phrases
                'not able to', 'unable to', 'cannot', 'can\'t cope',
                'giving up', 'no hope', 'no energy', 'no motivation',

                # Social/isolation
                'lonely', 'isolated', 'nobody', 'no one', 'abandoned',

                # Physical symptoms
                'fatigue', 'insomnia', 'sleepless', 'pain', 'headache',

                # Work/academic stress
                'work pressure', 'job stress', 'academic pressure', 'deadlines',
                'failing', 'failure', 'unemployed', 'jobless'
            ]

            # Check for multi-word phrases first (more specific)
            multi_word_phrases = [
                'not in a good mood', 'not able to', 'hard time', 'work pressure',
                'academic pressure', 'family pressure', 'no hope', 'no energy',
                'can\'t cope', 'giving up'
            ]

            depression_keyword_count = 0
            for phrase in multi_word_phrases:
                if phrase in transcribed_text:
                    depression_keyword_count += 2  # Count multi-word phrases more heavily

            # Then check single words
            for kw in depression_keywords:
                if len(kw.split()) == 1 and kw in transcribed_text:  # Single words only
                    depression_keyword_count += 1

            # Calculate sentiment score
            if polarity < -0.3:
                sentiment_score = 0.7
            elif polarity < -0.1:
                sentiment_score = 0.55
            elif polarity < 0.1:
                sentiment_score = 0.45
            elif polarity < 0.3:
                sentiment_score = 0.25
            elif polarity >= 0.3:
                # Strong positive sentiment
                sentiment_score = max(0.0, 0.15 - (polarity - 0.3) * 0.3)
            else:
                sentiment_score = max(0, -polarity) * 0.4

            # High subjectivity with negative sentiment suggests emotional distress
            if subjectivity > 0.7 and polarity < 0:
                sentiment_score += 0.15

            # Depression keywords boost (INCREASED SENSITIVITY)
            # Each keyword adds more weight, especially multi-word phrases
            if depression_keyword_count >= 5:
                depression_keyword_boost = 0.40  # Very high distress
            elif depression_keyword_count >= 3:
                depression_keyword_boost = 0.30  # High distress
            elif depression_keyword_count >= 2:
                depression_keyword_boost = 0.20  # Moderate distress
            elif depression_keyword_count >= 1:
                depression_keyword_boost = 0.15  # Mild distress
            else:
                depression_keyword_boost = 0.0

            sentiment_score += depression_keyword_boost

            logger.info(f"Depression keywords found: {depression_keyword_count}, boost: +{depression_keyword_boost:.2f}")

            # Positive keywords reduction
            positive_keyword_reduction = min(positive_keyword_count * 0.08, 0.30)
            sentiment_score = max(0.0, sentiment_score - positive_keyword_reduction)
            
            # ===== COMBINE VOICE AND SENTIMENT =====
            # ADAPTIVE WEIGHT DISTRIBUTION based on sentiment strength
            # When sentiment is strongly positive, give it more weight (user is clearly happy)
            # When sentiment is strongly negative, voice features are more reliable

            if polarity >= 0.4 or positive_keyword_count >= 2:
                # Strong positive sentiment - increase sentiment weight dramatically
                voice_weight = 0.40  # Reduce from 0.65
                sentiment_weight = 0.60  # Increase from 0.35
                logger.info(f"✓ Strong positive speech detected (polarity={polarity:.2f}, pos_keywords={positive_keyword_count})")
            elif polarity >= 0.2 or positive_keyword_count >= 1:
                # Moderate positive sentiment - moderately increase sentiment weight
                voice_weight = 0.50
                sentiment_weight = 0.50
            elif polarity <= -0.3 or depression_keyword_count >= 2:
                # Strong negative sentiment - voice is more reliable
                voice_weight = 0.70
                sentiment_weight = 0.30
            else:
                # Neutral/unclear - use default balanced weights
                voice_weight = 0.65
                sentiment_weight = 0.35

            combined_score = (voice_score * voice_weight + sentiment_score * sentiment_weight)
            
            # ===== APPLY CLINICAL THRESHOLDS =====
            # Research shows clear thresholds for depression voice markers
            
            # Severe indicators boost
            severe_indicators = 0
            if pitch_std < 0.2:
                severe_indicators += 1
            if energy_mean < 0.25:
                severe_indicators += 1
            if speaking_rate < 0.3:
                severe_indicators += 1
            if pause_frequency > 0.7:
                severe_indicators += 1
            if polarity < -0.4:
                severe_indicators += 1
            
            if severe_indicators >= 3:
                combined_score = min(combined_score * 1.25, 1.0)
                logger.info(f"Severe depression indicators detected: {severe_indicators}")
            elif severe_indicators >= 2:
                combined_score = min(combined_score * 1.15, 1.0)
            
            # IMPROVED Mild indicators reduction with more aggressive reductions
            mild_indicators = 0
            if pitch_std > 0.5:  # Lowered threshold from 0.6
                mild_indicators += 1
            if energy_mean > 0.6:  # Lowered threshold from 0.7
                mild_indicators += 1
            if speaking_rate > 0.6:  # Lowered threshold from 0.7
                mild_indicators += 1
            if polarity > 0.3:
                mild_indicators += 1
            if positive_keyword_count >= 1:  # NEW: Add positive keywords as indicator
                mild_indicators += 1

            # More aggressive reductions for positive indicators
            if mild_indicators >= 4:
                combined_score = max(combined_score * 0.45, 0.0)  # Very strong reduction
                logger.info(f"✓ Excellent positive indicators ({mild_indicators}/5)")
            elif mild_indicators >= 3:
                combined_score = max(combined_score * 0.55, 0.0)  # Strong reduction (was 0.75)
                logger.info(f"✓ Strong positive indicators ({mild_indicators}/5)")
            elif mild_indicators >= 2:
                combined_score = max(combined_score * 0.70, 0.0)  # Moderate reduction (was 0.85)
                logger.info(f"✓ Positive indicators detected ({mild_indicators}/5)")

            # POSITIVE KEYWORD OVERRIDE
            # If user says 2+ strong positive words, apply additional reduction
            if positive_keyword_count >= 3:
                combined_score = max(combined_score * 0.50, 0.0)
                logger.info(f"✓ POSITIVE KEYWORD OVERRIDE: {positive_keyword_count} positive words detected")
            elif positive_keyword_count >= 2 and polarity >= 0.3:
                combined_score = max(combined_score * 0.60, 0.0)
                logger.info(f"✓ Positive keyword boost: {positive_keyword_count} words + positive sentiment")
            
            # Final bounds check
            final_score = np.clip(combined_score, 0.0, 1.0)

            # IMPROVED LOGGING with voice features and keywords
            logger.info(f"Depression score: {final_score:.3f} "
                       f"(voice={voice_score:.3f}, sentiment={sentiment_score:.3f}, "
                       f"severe={severe_indicators}, mild={mild_indicators})")
            logger.info(f"Voice features: pitch_std={pitch_std:.2f}, energy={energy_mean:.2f}, "
                       f"rate={speaking_rate:.2f}, pauses={pause_frequency:.2f}")
            logger.info(f"Sentiment: polarity={polarity:.2f}, "
                       f"pos_keywords={positive_keyword_count}, dep_keywords={depression_keyword_count}")
            logger.info(f"Weights used: voice={voice_weight:.2f}, sentiment={sentiment_weight:.2f}")

            return float(final_score)
            
        except Exception as e:
            logger.error(f"Error calculating depression score: {e}", exc_info=True)
            return 0.5
    
    def _calculate_voice_confidence_score(self, voice_analysis: Dict, sentiment_analysis: Dict) -> float:
        """
        Calculate confidence score from voice patterns and speech sentiment.
        
        Voice indicators of confidence:
        - High energy level
        - Good speaking rate (not too fast/slow)
        - Moderate pitch variation (not monotone)
        - Low pause frequency
        - Higher spectral centroid (brighter voice)
        - Low jitter/shimmer (stable voice)
        
        Args:
            voice_analysis: Dict of voice features
            sentiment_analysis: Dict of sentiment features
            
        Returns:
            Confidence score (0-1, higher = more confident)
        """
        try:
            # Check if audio was silent or failed extraction
            if voice_analysis.get('is_silent', False):
                logger.error("Audio is too quiet or silent - cannot calculate confidence")
                return None  # Return None to indicate failure

            if voice_analysis.get('extraction_failed', False):
                logger.error("Feature extraction failed - cannot calculate confidence")
                return None  # Return None to indicate failure
            
            # ===== VOICE FEATURE INDICATORS =====
            energy_mean = voice_analysis.get('energy_mean', 0.5)
            speaking_rate = voice_analysis.get('speaking_rate', 0.5)
            pitch_variation = voice_analysis.get('pitch_std', 0.5)
            pitch_cv = voice_analysis.get('pitch_cv', 0.5)
            pause_frequency = voice_analysis.get('pause_frequency', 0.5)
            spectral_centroid = voice_analysis.get('spectral_centroid_mean', 0.5)
            jitter = voice_analysis.get('jitter', 0.5)
            shimmer = voice_analysis.get('shimmer', 0.5)
            
            # ===== CALCULATE VOICE CONFIDENCE INDICATORS =====
            voice_score = (
                energy_mean * 0.20 +                    # High energy = confident
                speaking_rate * 0.15 +                  # Good speaking rate
                pitch_variation * 0.15 +                # Good pitch variation
                pitch_cv * 0.10 +                       # Dynamic speech
                (1 - pause_frequency) * 0.15 +          # Low pause frequency
                spectral_centroid * 0.10 +              # Brighter voice tone
                (1 - jitter) * 0.075 +                  # Stable pitch
                (1 - shimmer) * 0.075                   # Stable amplitude
            )
            
            # ===== SENTIMENT ANALYSIS CONTRIBUTION =====
            polarity = sentiment_analysis.get('polarity', 0)
            subjectivity = sentiment_analysis.get('subjectivity', 0.5)
            
            # Positive sentiment increases confidence
            if polarity > 0.3:
                sentiment_score = 0.7
            elif polarity > 0.1:
                sentiment_score = 0.6
            elif polarity > -0.1:
                sentiment_score = 0.5
            else:
                sentiment_score = max(0, polarity + 0.5)
            
            # Moderate subjectivity might indicate self-assurance
            if 0.3 <= subjectivity <= 0.7:
                sentiment_score += 0.1
            
            # ===== COMBINE VOICE AND SENTIMENT =====
            combined_score = (voice_score * 0.75 + sentiment_score * 0.25)
            
            # ===== APPLY CONFIDENCE THRESHOLDS =====
            
            # High confidence indicators boost
            high_conf_indicators = 0
            if energy_mean > 0.7:
                high_conf_indicators += 1
            if speaking_rate > 0.6 and speaking_rate < 0.9:
                high_conf_indicators += 1
            if pause_frequency < 0.3:
                high_conf_indicators += 1
            if polarity > 0.3:
                high_conf_indicators += 1
            
            if high_conf_indicators >= 3:
                combined_score = min(combined_score * 1.2, 1.0)
            elif high_conf_indicators >= 2:
                combined_score = min(combined_score * 1.1, 1.0)
            
            # Low confidence indicators reduction
            low_conf_indicators = 0
            if energy_mean < 0.3:
                low_conf_indicators += 1
            if pause_frequency > 0.7:
                low_conf_indicators += 1
            if jitter > 0.6 or shimmer > 0.6:
                low_conf_indicators += 1
            if polarity < -0.2:
                low_conf_indicators += 1
            
            if low_conf_indicators >= 3:
                combined_score = max(combined_score * 0.7, 0.0)
            elif low_conf_indicators >= 2:
                combined_score = max(combined_score * 0.85, 0.0)
            
            # Final bounds check
            final_score = np.clip(combined_score, 0.0, 1.0)
            
            logger.info(f"Confidence score: {final_score:.3f} "
                       f"(voice={voice_score:.3f}, sentiment={sentiment_score:.3f})")
            
            return float(final_score)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}", exc_info=True)
            return 0.5
    
    def _analyze_audio_sentiment(self, audio_data: bytes) -> Dict:
        """
        Analyze voice patterns and speech sentiment from audio data.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Dict containing audio analysis results
        """
        try:
            logger.info("Starting audio sentiment analysis")
            
            # Extract audio features
            audio_features = self._extract_audio_features(audio_data)
            
            # Transcribe audio to text
            speech_text = self._transcribe_audio(audio_data)
            
            # Voice pattern analysis
            voice_analysis = {
                'pitch_variation': audio_features.get('pitch_std', 0.5),
                'pitch_mean': audio_features.get('pitch_mean', 0.5),
                'energy_level': audio_features.get('energy_mean', 0.5),
                'speaking_rate': audio_features.get('speaking_rate', 0.5),
                'pause_frequency': audio_features.get('pause_frequency', 0.5),
                'jitter': audio_features.get('jitter', 0.5),
                'shimmer': audio_features.get('shimmer', 0.5)
            }
            
            # Speech sentiment analysis
            sentiment_analysis = self._analyze_speech_sentiment(speech_text)

            # Calculate depression indicators from voice
            depression_score = self._calculate_voice_depression_score(audio_features, sentiment_analysis)
            confidence_score = self._calculate_voice_confidence_score(audio_features, sentiment_analysis)

            # Check if scoring failed due to silent/invalid audio
            if depression_score is None or confidence_score is None:
                error_msg = "Audio is too quiet or silent. Please record again in a quieter environment and speak clearly."
                logger.error(error_msg)
                return {
                    'error': error_msg,
                    'is_silent': True,
                    'voice_features': audio_features,
                    'analysis_failed': True
                }

            result = {
                'voice_features': voice_analysis,
                'speech_sentiment': sentiment_analysis,
                'transcribed_text': speech_text[:200] + "..." if len(speech_text) > 200 else speech_text,
                'depression_indicators': {
                    'score': depression_score,
                    'level': self._categorize_score(depression_score),
                    'voice_factors': self._identify_voice_depression_factors(audio_features)
                },
                'confidence_indicators': {
                    'score': confidence_score,
                    'level': self._categorize_score(confidence_score),
                    'voice_factors': self._identify_voice_confidence_factors(audio_features)
                },
                'speech_quality': 'good' if len(speech_text) > 50 else 'limited',
                'is_silent': audio_features.get('is_silent', False)
            }
            
            logger.info(f"Audio sentiment analysis completed: "
                       f"depression={depression_score:.3f}, confidence={confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in audio sentiment analysis: {e}", exc_info=True)
            return self._get_fallback_audio_analysis()
    
    def _analyze_video_emotions(self, video_data: bytes) -> Dict:
        """Analyze facial expressions from video data."""
        try:
            # Convert bytes to video frames
            frames = self._extract_frames_from_bytes(video_data)
            
            if not frames:
                logger.warning("No frames extracted from video")
                return self._get_fallback_video_analysis()
            
            emotion_scores = {
                'happiness': 0.0,
                'sadness': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'surprise': 0.0,
                'neutral': 0.0
            }
            
            face_detection_count = 0
            total_frames = len(frames)
            
            logger.info(f"Analyzing {total_frames} video frames")
            
            # Preferred: Use DeepFace or FER if available for more accurate emotions
            if DEEPFACE_AVAILABLE and frames:
                step = max(1, len(frames) // 20)
                for idx in range(0, len(frames), step):
                    frame = frames[idx]
                    try:
                        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        emo = analysis[0]['emotion'] if isinstance(analysis, list) else analysis.get('emotion', {})
                        if emo:
                            total = sum(float(v) for v in emo.values()) or 1.0
                            norm = {k.lower(): float(v) / total for k, v in emo.items()}
                            face_detection_count += 1
                            for key, val in norm.items():
                                if key in emotion_scores:
                                    emotion_scores[key] += val
                    except Exception:
                        continue
                        
            elif FER_AVAILABLE and frames:
                try:
                    detector = FER()
                    step = max(1, len(frames) // 20)
                    for idx in range(0, len(frames), step):
                        frame = frames[idx]
                        emo = detector.detect_emotions(frame)
                        if emo:
                            face_detection_count += 1
                            scores = emo[0].get('emotions', {})
                            for key, val in scores.items():
                                k = key.lower()
                                if k in emotion_scores:
                                    emotion_scores[k] += float(val)
                except Exception as e:
                    logger.warning(f"FER detection failed: {e}")
                    
            else:
                # Use basic OpenCV detection
                for frame in frames:
                    try:
                        if not CV2_AVAILABLE or self.face_cascade is None:
                            break
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                        
                        if len(faces) > 0:
                            face_detection_count += 1
                            emotions = self._detect_emotions_from_face(gray, faces[0])
                            for emotion, score in emotions.items():
                                emotion_scores[emotion] += score
                    except Exception:
                        continue
            
            # Average the scores
            if face_detection_count > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= face_detection_count
            else:
                logger.warning("No faces detected in video")
            
            # Calculate depression and confidence indicators
            depression_score = (
                emotion_scores['sadness'] * 0.4 +
                emotion_scores['fear'] * 0.3 +
                (1 - emotion_scores['happiness']) * 0.3
            )
            
            confidence_score = (
                emotion_scores['happiness'] * 0.4 +
                emotion_scores['neutral'] * 0.3 +
                (1 - emotion_scores['fear']) * 0.3
            )
            
            result = {
                'emotion_scores': emotion_scores,
                'depression_indicators': {
                    'score': min(depression_score, 1.0),
                    'level': self._categorize_score(depression_score),
                    'factors': self._identify_depression_factors(emotion_scores)
                },
                'confidence_indicators': {
                    'score': min(confidence_score, 1.0),
                    'level': self._categorize_score(confidence_score),
                    'factors': self._identify_confidence_factors(emotion_scores)
                },
                'face_detection_rate': face_detection_count / total_frames if total_frames > 0 else 0,
                'analysis_quality': 'good' if face_detection_count > total_frames * 0.5 else 'moderate'
            }
            
            logger.info(f"Video emotion analysis completed: "
                       f"depression={depression_score:.3f}, confidence={confidence_score:.3f}, "
                       f"faces_detected={face_detection_count}/{total_frames}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in video emotion analysis: {e}", exc_info=True)
            return self._get_fallback_video_analysis()
    
    def _extract_frames_from_bytes(self, video_data: bytes) -> List:
        """Extract frames from video bytes data using cv2 and temporary file."""
        frames = []
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, cannot extract frames")
            return frames
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                tmp.write(video_data)
                tmp.flush()
                tmp_path = tmp.name
            
            logger.info(f"Extracting frames from: {tmp_path}")
            
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logger.error("Failed to open video file")
                return frames
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
            
            logger.info(f"Video: {frame_count} frames, {fps} fps")
            
            # Sample frames (max 60 frames for efficiency)
            sample_every = max(1, frame_count // 60) if frame_count > 0 else 5
            
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % sample_every == 0:
                    frames.append(frame)
                idx += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}", exc_info=True)
            return frames
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    
    def _detect_emotions_from_face(self, gray_frame, face_coords: Tuple) -> Dict:
        """Detect emotions from facial features using basic computer vision."""
        try:
            x, y, w, h = face_coords
            face_roi = gray_frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return {
                    'happiness': 0.5,
                    'sadness': 0.5,
                    'anger': 0.5,
                    'fear': 0.5,
                    'surprise': 0.5,
                    'neutral': 0.5
                }
            
            emotions = {
                'happiness': 0.5,
                'sadness': 0.5,
                'anger': 0.5,
                'fear': 0.5,
                'surprise': 0.5,
                'neutral': 0.5
            }
            
            try:
                # Detect eyes and mouth
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
                
                eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
                smiles = smile_cascade.detectMultiScale(face_roi, 1.1, 3)
                
                # Analyze eye openness
                if len(eyes) >= 2:
                    eye_areas = [w*h for (x, y, w, h) in eyes]
                    avg_eye_area = np.mean(eye_areas) if eye_areas else 0
                    
                    if avg_eye_area > (w*h) * 0.02:
                        emotions['surprise'] = min(emotions['surprise'] + 0.3, 1.0)
                        emotions['fear'] = min(emotions['fear'] + 0.2, 1.0)
                    else:
                        emotions['neutral'] = min(emotions['neutral'] + 0.2, 1.0)
                
                # Analyze smile detection
                if len(smiles) > 0:
                    emotions['happiness'] = min(emotions['happiness'] + 0.4, 1.0)
                    emotions['sadness'] = max(emotions['sadness'] - 0.3, 0.0)
                else:
                    emotions['sadness'] = min(emotions['sadness'] + 0.2, 1.0)
                    emotions['neutral'] = min(emotions['neutral'] + 0.1, 1.0)
                
                # Analyze facial symmetry
                mid_x = w // 2
                left_half = face_roi[:, :mid_x]
                right_half = face_roi[:, mid_x:]
                
                if left_half.size > 0 and right_half.size > 0:
                    left_brightness = np.mean(left_half)
                    right_brightness = np.mean(right_half)
                    brightness_diff = abs(left_brightness - right_brightness)
                    
                    if brightness_diff > 20:
                        emotions['anger'] = min(emotions['anger'] + 0.2, 1.0)
                        emotions['fear'] = min(emotions['fear'] + 0.1, 1.0)
                    else:
                        emotions['neutral'] = min(emotions['neutral'] + 0.1, 1.0)
                
                # Analyze overall brightness
                overall_brightness = np.mean(face_roi)
                if overall_brightness < 100:
                    emotions['sadness'] = min(emotions['sadness'] + 0.2, 1.0)
                elif overall_brightness > 150:
                    emotions['happiness'] = min(emotions['happiness'] + 0.1, 1.0)
                    
            except Exception as e:
                logger.warning(f"Error in advanced face analysis: {e}")
            
            # Normalize emotions
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total * 3.0 for k, v in emotions.items()}
                emotions = {k: min(v, 1.0) for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return {
                'happiness': 0.5,
                'sadness': 0.5,
                'anger': 0.5,
                'fear': 0.5,
                'surprise': 0.5,
                'neutral': 0.5
            }
    
    def _combine_assessments(self, video_analysis: Dict, audio_analysis: Dict) -> Dict:
        """Combine video and audio analysis for comprehensive assessment."""
        try:
            video_depression = video_analysis.get('depression_indicators', {}).get('score', 0.5)
            audio_depression = audio_analysis.get('depression_indicators', {}).get('score', 0.5)
            
            video_confidence = video_analysis.get('confidence_indicators', {}).get('score', 0.5)
            audio_confidence = audio_analysis.get('confidence_indicators', {}).get('score', 0.5)
            
            # Weighted combination (audio is often more reliable than video for depression)
            combined_depression = video_depression * 0.4 + audio_depression * 0.6
            combined_confidence = video_confidence * 0.45 + audio_confidence * 0.55
            
            return {
                'depression_score': combined_depression,
                'depression_level': self._categorize_score(combined_depression),
                'confidence_score': combined_confidence,
                'confidence_level': self._categorize_score(combined_confidence),
                'overall_wellbeing': self._calculate_overall_wellbeing(combined_depression, combined_confidence),
                'confidence': min((video_analysis.get('analysis_quality', 'moderate') == 'good') * 0.5 +
                                (audio_analysis.get('speech_quality', 'limited') == 'good') * 0.5, 1.0)
            }
        except Exception as e:
            logger.error(f"Error combining assessments: {e}")
            return {
                'depression_score': 0.5,
                'depression_level': 'moderate',
                'confidence_score': 0.5,
                'confidence_level': 'moderate',
                'overall_wellbeing': 'moderate',
                'confidence': 0.5
            }
    
    def _generate_audio_assessment(self, audio_analysis: Dict) -> Dict:
        """Generate assessment based on audio-only analysis."""
        try:
            depression_score = audio_analysis.get('depression_indicators', {}).get('score', 0.5)
            confidence_score = audio_analysis.get('confidence_indicators', {}).get('score', 0.5)
            
            return {
                'depression_score': depression_score,
                'depression_level': self._categorize_score(depression_score),
                'confidence_score': confidence_score,
                'confidence_level': self._categorize_score(confidence_score),
                'overall_wellbeing': self._calculate_overall_wellbeing(depression_score, confidence_score),
                'confidence': 0.7 if audio_analysis.get('speech_quality') == 'good' else 0.5
            }
        except Exception as e:
            logger.error(f"Error generating audio assessment: {e}")
            return {
                'depression_score': 0.5,
                'depression_level': 'moderate',
                'confidence_score': 0.5,
                'confidence_level': 'moderate',
                'overall_wellbeing': 'moderate',
                'confidence': 0.5
            }
    
    def _categorize_score(self, score: float) -> str:
        """Categorize numerical score into descriptive level."""
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'moderate'
        else:
            return 'high'
    
    def _calculate_overall_wellbeing(self, depression_score: float, confidence_score: float) -> str:
        """Calculate overall wellbeing assessment."""
        wellbeing_score = (1 - depression_score) * 0.6 + confidence_score * 0.4
        
        if wellbeing_score > 0.7:
            return 'good'
        elif wellbeing_score > 0.4:
            return 'moderate'
        else:
            return 'concerning'
    
    def _generate_recommendations(self, assessment: Dict) -> List[str]:
        """Generate personalized recommendations based on assessment."""
        recommendations = []
        
        depression_level = assessment.get('depression_level', 'moderate')
        confidence_level = assessment.get('confidence_level', 'moderate')
        
        if depression_level == 'high':
            recommendations.extend([
                "Consider speaking with a mental health professional",
                "Practice daily mindfulness or meditation",
                "Engage in regular physical activity",
                "Maintain social connections with supportive people"
            ])
        elif depression_level == 'moderate':
            recommendations.extend([
                "Try journaling your thoughts and feelings",
                "Connect with supportive friends or family",
                "Maintain a regular sleep schedule",
                "Consider therapy or counseling if symptoms persist"
            ])
        else:
            recommendations.append("Continue maintaining your current positive mental health practices")
        
        if confidence_level == 'low':
            recommendations.extend([
                "Practice positive self-talk and affirmations",
                "Set small, achievable daily goals",
                "Consider confidence-building activities or workshops"
            ])
        
        return recommendations[:5]
    
    def _identify_depression_factors(self, emotion_scores: Dict) -> List[str]:
        """Identify factors contributing to depression indicators."""
        factors = []
        if emotion_scores.get('sadness', 0) > 0.6:
            factors.append("High sadness levels detected in facial expressions")
        if emotion_scores.get('fear', 0) > 0.5:
            factors.append("Elevated fear/anxiety indicators in facial expressions")
        if emotion_scores.get('happiness', 0) < 0.3:
            factors.append("Low positive emotion expression")
        if not factors:
            factors.append("No significant depression factors detected")
        return factors
    
    def _identify_confidence_factors(self, emotion_scores: Dict) -> List[str]:
        """Identify factors affecting confidence levels."""
        factors = []
        if emotion_scores.get('fear', 0) > 0.5:
            factors.append("High anxiety/fear levels may affect confidence")
        if emotion_scores.get('neutral', 0) < 0.3:
            factors.append("Limited emotional expression detected")
        if emotion_scores.get('happiness', 0) > 0.6:
            factors.append("Positive emotional expression supports confidence")
        if not factors:
            factors.append("Moderate confidence indicators")
        return factors
    
    def _identify_voice_depression_factors(self, voice_analysis: Dict) -> List[str]:
        """Identify voice factors indicating depression."""
        factors = []
        
        if voice_analysis.get('energy_mean', 0.5) < 0.3:
            factors.append("Low vocal energy detected")
        if voice_analysis.get('pitch_std', 0.5) < 0.3:
            factors.append("Monotone speech pattern (reduced pitch variation)")
        if voice_analysis.get('speaking_rate', 0.5) < 0.3:
            factors.append("Slow speaking rate (possible psychomotor retardation)")
        if voice_analysis.get('pause_frequency', 0.5) > 0.7:
            factors.append("Frequent pauses and hesitations")
        if voice_analysis.get('jitter', 0.5) > 0.6 or voice_analysis.get('shimmer', 0.5) > 0.6:
            factors.append("Voice quality instability detected")
        
        if not factors:
            factors.append("No significant voice-based depression indicators")
        
        return factors
    
    def _identify_voice_confidence_factors(self, voice_analysis: Dict) -> List[str]:
        """Identify voice factors affecting confidence."""
        factors = []
        
        if voice_analysis.get('pause_frequency', 0.5) > 0.7:
            factors.append("Frequent hesitations in speech")
        if voice_analysis.get('energy_mean', 0.5) < 0.4:
            factors.append("Low vocal confidence and projection")
        if voice_analysis.get('speaking_rate', 0.5) < 0.4:
            factors.append("Hesitant speaking pace")
        if voice_analysis.get('energy_mean', 0.5) > 0.7:
            factors.append("Strong vocal projection supports confidence")
        
        if not factors:
            factors.append("Moderate voice-based confidence indicators")
        
        return factors
    
    def _get_sentiment_label(self, polarity: float) -> str:
        """Convert polarity score to sentiment label."""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_emotional_words(self, text: str) -> List[str]:
        """Extract emotionally significant words from text."""
        if not text or len(text) < 3:
            return []
        
        emotional_keywords = [
            'sad', 'happy', 'worried', 'anxious', 'confident', 'afraid',
            'hopeful', 'depressed', 'excited', 'nervous', 'calm', 'angry',
            'frustrated', 'tired', 'exhausted', 'energetic', 'miserable',
            'joyful', 'stressed', 'relaxed', 'overwhelmed', 'peaceful',
            'hopeless', 'optimistic', 'scared', 'brave', 'weak', 'strong'
        ]
        
        words = text.lower().split()
        emotional_words = []
        
        for word in words:
            for keyword in emotional_keywords:
                if keyword in word:
                    emotional_words.append(word)
                    break
        
        return emotional_words
    
    def _get_fallback_assessment(self, assessment_type: str) -> Dict:
        """Return fallback assessment when analysis fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'assessment_type': assessment_type,
            'error': 'Analysis failed, using fallback assessment',
            'depression_score': 0.5,
            'depression_level': 'moderate',
            'confidence_score': 0.5,
            'confidence_level': 'moderate',
            'overall_wellbeing': 'moderate',
            'recommendations': [
                'Please try the assessment again with clear audio/video',
                'Ensure good lighting and minimal background noise',
                'Consider speaking with a mental health professional if you continue to experience difficulties'
            ],
            'confidence': 0.0,
            'processing_info': {
                'error': 'Processing failed',
                'fallback_used': True
            }
        }
    
    def _get_fallback_video_analysis(self) -> Dict:
        """Return fallback video analysis."""
        return {
            'emotion_scores': {
                'happiness': 0.5,
                'sadness': 0.5,
                'anger': 0.5,
                'fear': 0.5,
                'surprise': 0.5,
                'neutral': 0.5
            },
            'depression_indicators': {
                'score': 0.5,
                'level': 'moderate',
                'factors': ['Video analysis unavailable']
            },
            'confidence_indicators': {
                'score': 0.5,
                'level': 'moderate',
                'factors': ['Video analysis unavailable']
            },
            'face_detection_rate': 0.0,
            'analysis_quality': 'insufficient'
        }
    
    def _get_fallback_audio_analysis(self) -> Dict:
        """Return fallback audio analysis."""
        return {
            'voice_features': {
                'pitch_variation': 0.5,
                'energy_level': 0.5,
                'speaking_rate': 0.5,
                'pause_frequency': 0.5
            },
            'speech_sentiment': {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'sentiment_label': 'neutral',
                'emotional_words': []
            },
            'transcribed_text': 'Audio analysis unavailable',
            'depression_indicators': {
                'score': 0.5,
                'level': 'moderate',
                'voice_factors': ['Audio analysis unavailable']
            },
            'confidence_indicators': {
                'score': 0.5,
                'level': 'moderate',
                'voice_factors': ['Audio analysis unavailable']
            },
            'speech_quality': 'insufficient',
            'is_silent': False
        }