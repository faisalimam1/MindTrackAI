#!/usr/bin/env python3
"""
Video and Audio Analysis Service for Mental Health Assessment
Analyzes facial expressions, voice patterns, and speech for depression/confidence detection
"""

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
    print("Warning: NumPy not available. Analysis will be limited.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: Librosa not available. Audio analysis will be limited.")

# Optional: moviepy for extracting audio from video containers (e.g., webm)
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Audio extraction from video will be limited.")

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

# Optional: Transformers-based audio emotion recognition
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Advanced audio emotion recognition disabled.")

# Optional: DeepFace or FER for video emotion analysis
try:
    from deepface import DeepFace  # type: ignore
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False
    print("Info: DeepFace not available.")

try:
    from fer import FER  # type: ignore
    FER_AVAILABLE = True
except Exception:
    FER_AVAILABLE = False
    print("Info: FER not available.")

import json
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import random

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
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
                        'textblob': TEXTBLOB_AVAILABLE
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
            
            logger.info(f"Video/audio analysis completed successfully")
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
            return result
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return self._get_fallback_assessment('audio_only')
    
    def _analyze_video_emotions(self, video_data: bytes) -> Dict:
        """Analyze facial expressions from video data."""
        try:
            # Convert bytes to video frames
            frames = self._extract_frames_from_bytes(video_data)
            
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
            
            # Preferred: Use DeepFace or FER if available for more accurate emotions
            if 'DeepFace' in globals() and DEEPFACE_AVAILABLE and frames:
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
            elif 'FER' in globals() and FER_AVAILABLE and frames:
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
                except Exception:
                    pass
            else:
                for frame in frames:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4) if self.face_cascade is not None else []
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
            
            # Calculate depression and confidence indicators
            depression_score = (emotion_scores['sadness'] * 0.4 + 
                              emotion_scores['fear'] * 0.3 + 
                              (1 - emotion_scores['happiness']) * 0.3)
            
            confidence_score = (emotion_scores['happiness'] * 0.4 + 
                               emotion_scores['neutral'] * 0.3 + 
                               (1 - emotion_scores['fear']) * 0.3)
            
            return {
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
            
        except Exception as e:
            logger.error(f"Error in video emotion analysis: {e}")
            return self._get_fallback_video_analysis()
    
    def _analyze_audio_sentiment(self, audio_data: bytes) -> Dict:
        """Analyze voice patterns and speech sentiment from audio data."""
        try:
            # Convert audio data to format suitable for analysis
            audio_features = self._extract_audio_features(audio_data)
            speech_text = self._transcribe_audio(audio_data)
            
            # Voice pattern analysis
            voice_analysis = {
                'pitch_variation': audio_features.get('pitch_std', 0.5),
                'energy_level': audio_features.get('energy_mean', 0.5),
                'speaking_rate': audio_features.get('speaking_rate', 0.5),
                'pause_frequency': audio_features.get('pause_frequency', 0.5)
            }
            
            # Speech sentiment analysis
            sentiment_analysis = self._analyze_speech_sentiment(speech_text)

            # Advanced: transformers-based speech emotion recognition if available
            audio_emotion = {}
            if 'pipeline' in globals() and TRANSFORMERS_AVAILABLE:
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmp:
                        self._write_audio_wav(audio_data, tmp.name)
                        emo_pipe = pipeline('audio-classification', model='superb/hubert-large-superb-er', top_k=5)
                        preds = emo_pipe(tmp.name)
                        if isinstance(preds, list):
                            for pred in preds:
                                if isinstance(pred, dict) and 'label' in pred and 'score' in pred:
                                    audio_emotion[pred['label'].lower()] = float(pred['score'])
                except Exception as e:
                    logger.warning(f"Audio emotion pipeline failed: {e}")
            
            # Calculate depression indicators from voice
            depression_score = self._calculate_voice_depression_score(voice_analysis, sentiment_analysis)
            confidence_score = self._calculate_voice_confidence_score(voice_analysis, sentiment_analysis)
            
            return {
                'voice_features': voice_analysis,
                'speech_sentiment': sentiment_analysis,
                'transcribed_text': speech_text[:200] + "..." if len(speech_text) > 200 else speech_text,
                'audio_emotion': audio_emotion or None,
                'depression_indicators': {
                    'score': depression_score,
                    'level': self._categorize_score(depression_score),
                    'voice_factors': self._identify_voice_depression_factors(voice_analysis)
                },
                'confidence_indicators': {
                    'score': confidence_score,
                    'level': self._categorize_score(confidence_score),
                    'voice_factors': self._identify_voice_confidence_factors(voice_analysis)
                },
                'speech_quality': 'good' if len(speech_text) > 50 else 'limited'
            }
            
        except Exception as e:
            logger.error(f"Error in audio sentiment analysis: {e}")
            return self._get_fallback_audio_analysis()
    
    def _extract_frames_from_bytes(self, video_data: bytes) -> List:
        """Extract frames from video bytes data using cv2 and temporary file."""
        frames: List = []
        if not CV2_AVAILABLE:
            return frames
        import tempfile, os
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                tmp.write(video_data)
                tmp.flush()
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
            return frames
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return frames
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _write_audio_wav(self, audio_bytes: bytes, out_wav_path: str) -> None:
        """Best-effort conversion of raw recording bytes to WAV using moviepy or librosa."""
        import tempfile, os
        tmp_in = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as f:
                f.write(audio_bytes)
                f.flush()
                tmp_in = f.name
            if 'VideoFileClip' in globals() and MOVIEPY_AVAILABLE:
                try:
                    clip = VideoFileClip(tmp_in)
                    audio = clip.audio
                    if audio is not None:
                        audio.write_audiofile(out_wav_path, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
                        clip.close()
                        return
                except Exception as e:
                    logger.warning(f"moviepy conversion failed: {e}")
            if LIBROSA_AVAILABLE:
                try:
                    import soundfile as sf  # type: ignore
                    y, sr = librosa.load(tmp_in, sr=16000, mono=True)
                    sf.write(out_wav_path, y, 16000)
                    return
                except Exception as e:
                    logger.warning(f"librosa conversion failed: {e}")
            with open(out_wav_path, 'wb') as f_out:
                f_out.write(audio_bytes)
        finally:
            try:
                if tmp_in and os.path.exists(tmp_in):
                    os.remove(tmp_in)
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
            
            # Basic emotion detection using facial geometry
            emotions = {
                'happiness': 0.5,
                'sadness': 0.5,
                'anger': 0.5,
                'fear': 0.5,
                'surprise': 0.5,
                'neutral': 0.5
            }
            
            # Analyze facial features using OpenCV
            try:
                # Detect eyes and mouth using Haar cascades
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
                
                # Detect eyes
                eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
                # Detect smiles
                smiles = smile_cascade.detectMultiScale(face_roi, 1.1, 3)
                
                # Analyze eye openness (simplified)
                if len(eyes) >= 2:
                    # Calculate average eye area
                    eye_areas = [w*h for (x, y, w, h) in eyes]
                    avg_eye_area = np.mean(eye_areas) if eye_areas else 0
                    
                    # Large eyes might indicate surprise or fear
                    if avg_eye_area > (w*h) * 0.02:  # Threshold for large eyes
                        emotions['surprise'] = min(emotions['surprise'] + 0.3, 1.0)
                        emotions['fear'] = min(emotions['fear'] + 0.2, 1.0)
                    else:
                        emotions['neutral'] = min(emotions['neutral'] + 0.2, 1.0)
                
                # Analyze smile detection
                if len(smiles) > 0:
                    emotions['happiness'] = min(emotions['happiness'] + 0.4, 1.0)
                    emotions['sadness'] = max(emotions['sadness'] - 0.3, 0.0)
                else:
                    # No smile detected, might indicate sadness or neutral
                    emotions['sadness'] = min(emotions['sadness'] + 0.2, 1.0)
                    emotions['neutral'] = min(emotions['neutral'] + 0.1, 1.0)
                
                # Analyze facial symmetry and brightness
                # Split face into left and right halves
                mid_x = w // 2
                left_half = face_roi[:, :mid_x]
                right_half = face_roi[:, mid_x:]
                
                if left_half.size > 0 and right_half.size > 0:
                    # Calculate brightness difference
                    left_brightness = np.mean(left_half)
                    right_brightness = np.mean(right_half)
                    brightness_diff = abs(left_brightness - right_brightness)
                    
                    # High asymmetry might indicate stress or anger
                    if brightness_diff > 20:  # Threshold for asymmetry
                        emotions['anger'] = min(emotions['anger'] + 0.2, 1.0)
                        emotions['fear'] = min(emotions['fear'] + 0.1, 1.0)
                    else:
                        emotions['neutral'] = min(emotions['neutral'] + 0.1, 1.0)
                
                # Analyze overall brightness (darker might indicate sadness)
                overall_brightness = np.mean(face_roi)
                if overall_brightness < 100:  # Dark threshold
                    emotions['sadness'] = min(emotions['sadness'] + 0.2, 1.0)
                elif overall_brightness > 150:  # Bright threshold
                    emotions['happiness'] = min(emotions['happiness'] + 0.1, 1.0)
                
            except Exception as e:
                logger.warning(f"Error in advanced face analysis: {e}")
                # Fallback to basic analysis
                pass
            
            # Normalize emotions to ensure they sum to a reasonable range
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total * 3.0 for k, v in emotions.items()}  # Scale to ~3.0 total
                emotions = {k: min(v, 1.0) for k, v in emotions.items()}  # Cap at 1.0
            
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
    
    def _extract_audio_features(self, audio_data: bytes) -> Dict:
        """Extract audio features for voice pattern analysis."""
        try:
            if not LIBROSA_AVAILABLE:
                logger.warning("Librosa not available, using fallback features")
                return {
                    'pitch_std': 0.5,
                    'energy_mean': 0.5,
                    'speaking_rate': 0.5,
                    'pause_frequency': 0.5
                }
            
            # Convert audio bytes to numpy array
            import tempfile
            import os
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                # Try to write audio data directly first
                try:
                    tmp.write(audio_data)
                    tmp.flush()
                    
                    # Load audio with librosa - try different formats
                    try:
                        y, sr = librosa.load(tmp.name, sr=16000, mono=True)
                    except Exception as e:
                        logger.warning(f"Failed to load with librosa: {e}")
                        # Try with different sample rate
                        y, sr = librosa.load(tmp.name, sr=None, mono=True)
                        # Resample if needed
                        if sr != 16000:
                            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                            sr = 16000
                    
                    # Extract pitch (F0) using librosa
                    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
                    pitch_values = []
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if pitch > 0:
                            pitch_values.append(pitch)
                    
                    pitch_std = np.std(pitch_values) if pitch_values else 0.0
                    pitch_mean = np.mean(pitch_values) if pitch_values else 0.0
                    
                    # Extract energy (RMS)
                    energy = librosa.feature.rms(y=y)[0]
                    energy_mean = np.mean(energy)
                    energy_std = np.std(energy)
                    
                    # Extract spectral features
                    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                    
                    # Calculate speaking rate (approximate)
                    # Find speech segments using energy
                    energy_threshold = np.mean(energy) * 0.1
                    speech_frames = energy > energy_threshold
                    speech_duration = np.sum(speech_frames) / sr
                    total_duration = len(y) / sr
                    speaking_rate = speech_duration / total_duration if total_duration > 0 else 0.0
                    
                    # Calculate pause frequency
                    # Find pauses (low energy regions)
                    pause_threshold = np.mean(energy) * 0.05
                    pauses = energy < pause_threshold
                    pause_count = np.sum(np.diff(pauses.astype(int)) == 1)  # Count pause starts
                    pause_frequency = pause_count / total_duration if total_duration > 0 else 0.0
                    
                    # Normalize features to 0-1 range
                    normalized_features = {
                        'pitch_std': min(pitch_std / 1000.0, 1.0),  # Normalize pitch std
                        'pitch_mean': min(pitch_mean / 500.0, 1.0),  # Normalize pitch mean
                        'energy_mean': min(energy_mean * 10.0, 1.0),  # Normalize energy
                        'energy_std': min(energy_std * 10.0, 1.0),
                        'speaking_rate': min(speaking_rate, 1.0),
                        'pause_frequency': min(pause_frequency, 1.0),
                        'spectral_centroid_mean': min(np.mean(spectral_centroids) / 5000.0, 1.0),
                        'spectral_rolloff_mean': min(np.mean(spectral_rolloff) / 10000.0, 1.0)
                    }
                    
                    return normalized_features
                    
                except Exception as e:
                    logger.error(f"Error processing audio with librosa: {e}")
                    # Fallback to basic analysis
                    return {
                        'pitch_std': 0.5,
                        'energy_mean': 0.5,
                        'speaking_rate': 0.5,
                        'pause_frequency': 0.5
                    }
                finally:
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {
                'pitch_std': 0.5,
                'energy_mean': 0.5,
                'speaking_rate': 0.5,
                'pause_frequency': 0.5
            }
    
    def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio to text for sentiment analysis."""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                logger.warning("Speech recognition not available, using fallback")
                return "Speech recognition not available"
            
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                try:
                    # Write audio data to temporary file
                    tmp.write(audio_data)
                    tmp.flush()
                    
                    # Use speech recognition to transcribe
                    with sr.AudioFile(tmp.name) as source:
                        # Adjust for ambient noise
                        self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.2)
                        audio = self.speech_recognizer.record(source)
                    
                    # Try multiple recognition engines
                    try:
                        # Try Google Speech Recognition first
                        text = self.speech_recognizer.recognize_google(audio)
                        logger.info(f"Transcribed text: {text[:100]}...")
                        return text
                    except sr.UnknownValueError:
                        logger.warning("Google Speech Recognition could not understand audio")
                        try:
                            # Try Sphinx as fallback
                            text = self.speech_recognizer.recognize_sphinx(audio)
                            logger.info(f"Transcribed text (Sphinx): {text[:100]}...")
                            return text
                        except sr.UnknownValueError:
                            logger.warning("Sphinx could not understand audio")
                            return "Could not understand audio"
                    except sr.RequestError as e:
                        logger.error(f"Speech recognition service error: {e}")
                        return "Speech recognition service unavailable"
                        
                except Exception as e:
                    logger.error(f"Error in speech recognition: {e}")
                    return "Error processing audio for transcription"
                finally:
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return "Unable to transcribe audio"
    
    def _analyze_speech_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of transcribed speech."""
        try:
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                sentiment = blob.sentiment
            else:
                # Fallback sentiment analysis
                sentiment = type('obj', (object,), {'polarity': 0.0, 'subjectivity': 0.5})()
            
            return {
                'polarity': sentiment.polarity,  # -1 to 1
                'subjectivity': sentiment.subjectivity,  # 0 to 1
                'sentiment_label': self._get_sentiment_label(sentiment.polarity),
                'emotional_words': self._extract_emotional_words(text)
            }
        except Exception as e:
            logger.error(f"Error analyzing speech sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5, 'sentiment_label': 'neutral', 'emotional_words': []}
    
    def _calculate_voice_depression_score(self, voice_analysis: Dict, sentiment_analysis: Dict) -> float:
        """Calculate depression score from voice patterns and speech sentiment."""
        # Voice indicators of depression:
        # - Low pitch variation (monotone)
        # - Low energy level
        # - High pause frequency (hesitation)
        # - Slow speaking rate
        # - Low spectral centroid (darker voice)
        
        pitch_variation = voice_analysis.get('pitch_std', 0.5)
        energy_level = voice_analysis.get('energy_mean', 0.5)
        pause_frequency = voice_analysis.get('pause_frequency', 0.5)
        speaking_rate = voice_analysis.get('speaking_rate', 0.5)
        spectral_centroid = voice_analysis.get('spectral_centroid_mean', 0.5)
        
        # Calculate voice depression indicators
        voice_score = (
            (1 - pitch_variation) * 0.25 +  # Monotone speech
            (1 - energy_level) * 0.25 +     # Low energy
            pause_frequency * 0.20 +        # Frequent pauses
            (1 - speaking_rate) * 0.15 +    # Slow speech
            (1 - spectral_centroid) * 0.15  # Darker voice tone
        )
        
        # Sentiment analysis contribution
        polarity = sentiment_analysis.get('polarity', 0)
        subjectivity = sentiment_analysis.get('subjectivity', 0.5)
        
        # Negative sentiment increases depression score
        sentiment_score = max(0, -polarity) * 0.6
        
        # High subjectivity might indicate emotional distress
        if subjectivity > 0.7:
            sentiment_score += 0.2
        
        # Combine voice and sentiment scores
        combined_score = (voice_score * 0.7 + sentiment_score * 0.3)
        
        # Apply some sensitivity to make it more responsive
        if combined_score > 0.6:
            combined_score = min(combined_score * 1.2, 1.0)  # Boost high scores
        elif combined_score < 0.3:
            combined_score = max(combined_score * 0.8, 0.0)  # Reduce low scores
        
        return min(combined_score, 1.0)
    
    def _calculate_voice_confidence_score(self, voice_analysis: Dict, sentiment_analysis: Dict) -> float:
        """Calculate confidence score from voice patterns and speech sentiment."""
        # Voice indicators of confidence:
        # - High energy level
        # - Good speaking rate (not too fast/slow)
        # - Moderate pitch variation (not monotone)
        # - Low pause frequency
        # - Higher spectral centroid (brighter voice)
        
        energy_level = voice_analysis.get('energy_mean', 0.5)
        speaking_rate = voice_analysis.get('speaking_rate', 0.5)
        pitch_variation = voice_analysis.get('pitch_std', 0.5)
        pause_frequency = voice_analysis.get('pause_frequency', 0.5)
        spectral_centroid = voice_analysis.get('spectral_centroid_mean', 0.5)
        
        # Calculate voice confidence indicators
        voice_score = (
            energy_level * 0.25 +                    # High energy
            speaking_rate * 0.20 +                   # Good speaking rate
            pitch_variation * 0.20 +                 # Good pitch variation
            (1 - pause_frequency) * 0.20 +           # Low pause frequency
            spectral_centroid * 0.15                 # Brighter voice tone
        )
        
        # Sentiment analysis contribution
        polarity = sentiment_analysis.get('polarity', 0)
        subjectivity = sentiment_analysis.get('subjectivity', 0.5)
        
        # Positive sentiment increases confidence
        sentiment_score = max(0, polarity) * 0.4
        
        # Moderate subjectivity might indicate self-assurance
        if 0.3 <= subjectivity <= 0.7:
            sentiment_score += 0.1
        
        # Combine voice and sentiment scores
        combined_score = (voice_score * 0.8 + sentiment_score * 0.2)
        
        # Apply some sensitivity to make it more responsive
        if combined_score > 0.7:
            combined_score = min(combined_score * 1.1, 1.0)  # Slight boost for high scores
        elif combined_score < 0.4:
            combined_score = max(combined_score * 0.9, 0.0)  # Slight reduction for low scores
        
        return min(combined_score, 1.0)
    
    def _combine_assessments(self, video_analysis: Dict, audio_analysis: Dict) -> Dict:
        """Combine video and audio analysis for comprehensive assessment."""
        video_depression = video_analysis.get('depression_indicators', {}).get('score', 0.5)
        audio_depression = audio_analysis.get('depression_indicators', {}).get('score', 0.5)
        
        video_confidence = video_analysis.get('confidence_indicators', {}).get('score', 0.5)
        audio_confidence = audio_analysis.get('confidence_indicators', {}).get('score', 0.5)
        
        # Weighted combination (video 60%, audio 40%)
        combined_depression = video_depression * 0.6 + audio_depression * 0.4
        combined_confidence = video_confidence * 0.6 + audio_confidence * 0.4
        
        return {
            'depression_score': combined_depression,
            'depression_level': self._categorize_score(combined_depression),
            'confidence_score': combined_confidence,
            'confidence_level': self._categorize_score(combined_confidence),
            'overall_wellbeing': self._calculate_overall_wellbeing(combined_depression, combined_confidence),
            'confidence': min((video_analysis.get('analysis_quality', 'moderate') == 'good') * 0.5 + 
                            (audio_analysis.get('speech_quality', 'limited') == 'good') * 0.5, 1.0)
        }
    
    def _generate_audio_assessment(self, audio_analysis: Dict) -> Dict:
        """Generate assessment based on audio-only analysis."""
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
                "Engage in regular physical activity"
            ])
        elif depression_level == 'moderate':
            recommendations.extend([
                "Try journaling your thoughts and feelings",
                "Connect with supportive friends or family",
                "Maintain a regular sleep schedule"
            ])
        
        if confidence_level == 'low':
            recommendations.extend([
                "Practice positive self-talk and affirmations",
                "Set small, achievable daily goals",
                "Consider confidence-building activities"
            ])
        
        if not recommendations:
            recommendations.append("Continue maintaining your current positive mental health practices")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _identify_depression_factors(self, emotion_scores: Dict) -> List[str]:
        """Identify factors contributing to depression indicators."""
        factors = []
        if emotion_scores.get('sadness', 0) > 0.6:
            factors.append("High sadness levels detected")
        if emotion_scores.get('fear', 0) > 0.5:
            factors.append("Elevated fear/anxiety indicators")
        if emotion_scores.get('happiness', 0) < 0.3:
            factors.append("Low positive emotion expression")
        return factors
    
    def _identify_confidence_factors(self, emotion_scores: Dict) -> List[str]:
        """Identify factors affecting confidence levels."""
        factors = []
        if emotion_scores.get('fear', 0) > 0.5:
            factors.append("High anxiety/fear levels")
        if emotion_scores.get('neutral', 0) < 0.3:
            factors.append("Limited emotional expression")
        return factors
    
    def _identify_voice_depression_factors(self, voice_analysis: Dict) -> List[str]:
        """Identify voice factors indicating depression."""
        factors = []
        if voice_analysis.get('energy_level', 0.5) < 0.3:
            factors.append("Low vocal energy")
        if voice_analysis.get('pitch_variation', 0.5) < 0.3:
            factors.append("Monotone speech pattern")
        if voice_analysis.get('speaking_rate', 0.5) < 0.3:
            factors.append("Slow speaking rate")
        return factors
    
    def _identify_voice_confidence_factors(self, voice_analysis: Dict) -> List[str]:
        """Identify voice factors affecting confidence."""
        factors = []
        if voice_analysis.get('pause_frequency', 0.5) > 0.7:
            factors.append("Frequent hesitations")
        if voice_analysis.get('energy_level', 0.5) < 0.4:
            factors.append("Low vocal confidence")
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
        emotional_words = ['sad', 'happy', 'worried', 'anxious', 'confident', 'afraid', 'hopeful', 'depressed']
        words = text.lower().split()
        return [word for word in words if any(emo in word for emo in emotional_words)]
    
    def _get_fallback_assessment(self, assessment_type: str) -> Dict:
        """Return fallback assessment when analysis fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'assessment_type': assessment_type,
            'error': 'Analysis failed, using fallback assessment',
            'depression_score': None,
            'confidence_score': None,
            'recommendations': ['Please try the assessment again', 'Consider speaking with a mental health professional'],
            'confidence': 0.0
        }
    
    def _get_fallback_video_analysis(self) -> Dict:
        """Return fallback video analysis."""
        return {
            'emotion_scores': None,
            'depression_indicators': {'score': None, 'level': 'unknown'},
            'confidence_indicators': {'score': None, 'level': 'unknown'},
            'analysis_quality': 'insufficient'
        }
    
    def _get_fallback_audio_analysis(self) -> Dict:
        """Return fallback audio analysis."""
        return {
            'voice_features': None,
            'speech_sentiment': {'polarity': 0.0, 'sentiment_label': 'neutral'},
            'depression_indicators': {'score': None, 'level': 'unknown'},
            'confidence_indicators': {'score': None, 'level': 'unknown'},
            'speech_quality': 'insufficient'
        }
