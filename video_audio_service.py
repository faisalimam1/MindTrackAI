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
            # Analyze video for facial expressions
            video_analysis = self._analyze_video_emotions(video_data)
            
            # Analyze audio for voice patterns and speech
            audio_analysis = self._analyze_audio_sentiment(audio_data)
            
            # Combine results for comprehensive assessment
            combined_assessment = self._combine_assessments(video_analysis, audio_analysis)
            
            combined = {
                'timestamp': datetime.now().isoformat(),
                'assessment_type': 'video_audio',
                'video_analysis': video_analysis,
                'audio_analysis': audio_analysis,
                'combined_assessment': combined_assessment,
                'confidence_score': combined_assessment.get('confidence', 0.5),
                'recommendations': self._generate_recommendations(combined_assessment)
            }
            # Flatten key combined metrics for frontend consumption
            combined.update({
                'depression_score': combined_assessment.get('depression_score', 0.5),
                'depression_level': combined_assessment.get('depression_level', 'moderate'),
                'confidence_score': combined_assessment.get('confidence_score', combined_assessment.get('confidence', 0.5)),
                'confidence_level': combined_assessment.get('confidence_level', 'moderate'),
                'overall_wellbeing': combined_assessment.get('overall_wellbeing', 'moderate')
            })
            return combined
            
        except Exception as e:
            logger.error(f"Error in video/audio analysis: {e}")
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
        """Detect emotions from facial features (simplified implementation)."""
        # Simplified emotion detection - in production, use proper ML models
        return {
            'happiness': random.uniform(0.2, 0.8),
            'sadness': random.uniform(0.1, 0.6),
            'anger': random.uniform(0.0, 0.4),
            'fear': random.uniform(0.0, 0.5),
            'surprise': random.uniform(0.0, 0.3),
            'neutral': random.uniform(0.3, 0.7)
        }
    
    def _extract_audio_features(self, audio_data: bytes) -> Dict:
        """Extract audio features for voice pattern analysis."""
        try:
            # Simplified audio feature extraction
            return {
                'pitch_std': random.uniform(0.2, 0.8),
                'energy_mean': random.uniform(0.3, 0.9),
                'speaking_rate': random.uniform(0.4, 0.8),
                'pause_frequency': random.uniform(0.2, 0.7)
            }
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
    
    def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio to text for sentiment analysis."""
        try:
            # Simplified transcription - in production, use proper speech recognition
            sample_texts = [
                "I feel okay today but sometimes I worry about things",
                "Life has been challenging lately and I'm not sure what to do",
                "I'm trying to stay positive but it's difficult sometimes",
                "I feel confident about some things but uncertain about others"
            ]
            return random.choice(sample_texts)
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
        voice_score = (
            (1 - voice_analysis.get('pitch_variation', 0.5)) * 0.3 +
            (1 - voice_analysis.get('energy_level', 0.5)) * 0.3 +
            voice_analysis.get('pause_frequency', 0.5) * 0.2 +
            (1 - voice_analysis.get('speaking_rate', 0.5)) * 0.2
        )
        
        sentiment_score = max(0, -sentiment_analysis.get('polarity', 0)) * 0.5
        
        return min((voice_score + sentiment_score) / 1.5, 1.0)
    
    def _calculate_voice_confidence_score(self, voice_analysis: Dict, sentiment_analysis: Dict) -> float:
        """Calculate confidence score from voice patterns and speech sentiment."""
        voice_score = (
            voice_analysis.get('energy_level', 0.5) * 0.4 +
            voice_analysis.get('speaking_rate', 0.5) * 0.3 +
            voice_analysis.get('pitch_variation', 0.5) * 0.3
        )
        
        sentiment_score = max(0, sentiment_analysis.get('polarity', 0)) * 0.5
        
        return min((voice_score + sentiment_score) / 1.5, 1.0)
    
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
            'depression_score': 0.5,
            'confidence_score': 0.5,
            'recommendations': ['Please try the assessment again', 'Consider speaking with a mental health professional'],
            'confidence': 0.3
        }
    
    def _get_fallback_video_analysis(self) -> Dict:
        """Return fallback video analysis."""
        return {
            'emotion_scores': {'happiness': 0.5, 'sadness': 0.3, 'neutral': 0.2},
            'depression_indicators': {'score': 0.5, 'level': 'moderate'},
            'confidence_indicators': {'score': 0.5, 'level': 'moderate'},
            'analysis_quality': 'limited'
        }
    
    def _get_fallback_audio_analysis(self) -> Dict:
        """Return fallback audio analysis."""
        return {
            'voice_features': {'energy_level': 0.5, 'speaking_rate': 0.5},
            'speech_sentiment': {'polarity': 0.0, 'sentiment_label': 'neutral'},
            'depression_indicators': {'score': 0.5, 'level': 'moderate'},
            'confidence_indicators': {'score': 0.5, 'level': 'moderate'},
            'speech_quality': 'limited'
        }
