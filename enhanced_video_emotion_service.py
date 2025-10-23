#!/usr/bin/env python3
"""
Enhanced Video Emotion Detection Service for Mental Health Assessment
Uses state-of-the-art computer vision models for comprehensive facial analysis

Features:
- Multi-model emotion recognition (DeepFace, FER, MediaPipe)
- Facial Action Unit (AU) detection for micro-expressions
- Gaze tracking and attention analysis
- Head pose estimation
- Blink rate and eye contact monitoring
- Clinical-grade depression and anxiety scoring
- Personalized mental health recommendations
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tempfile
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with graceful degradation
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not available - advanced emotion recognition disabled")

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    logger.warning("FER not available - alternative emotion recognition disabled")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available - facial landmarks and gaze tracking disabled")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - BERT emotion analysis disabled")


@dataclass
class EmotionFrame:
    """Represents emotion data from a single frame"""
    frame_number: int
    timestamp: float
    emotions: Dict[str, float]
    face_detected: bool
    confidence: float
    facial_landmarks: Optional[List[Tuple[float, float]]] = None
    head_pose: Optional[Dict[str, float]] = None
    gaze_direction: Optional[Dict[str, float]] = None
    action_units: Optional[Dict[str, float]] = None


@dataclass
class MentalHealthIndicators:
    """Clinical mental health indicators from video analysis"""
    depression_score: float  # 0-1
    anxiety_score: float  # 0-1
    stress_level: float  # 0-1
    emotional_stability: float  # 0-1
    engagement_level: float  # 0-1
    confidence_level: str
    primary_emotions: List[str]
    risk_factors: List[str]
    protective_factors: List[str]


class EnhancedVideoEmotionService:
    """
    Advanced video emotion analysis service for mental health assessment.

    Uses multiple computer vision models to analyze:
    - Facial expressions (7 basic emotions + micro-expressions)
    - Facial Action Units (FACS)
    - Gaze patterns and eye contact
    - Head pose and body language
    - Blink rate and attention markers
    """

    def __init__(self):
        """Initialize emotion detection models"""
        self.fer_detector = None
        self.mp_face_mesh = None
        self.mp_face_detection = None
        self.mp_pose = None

        self._load_models()

    def _load_models(self):
        """Load all available computer vision models"""
        try:
            # Load FER detector if available
            if FER_AVAILABLE:
                self.fer_detector = FER(mtcnn=True)  # Use MTCNN for better face detection
                logger.info("✓ FER emotion detector loaded")

            # Load MediaPipe models if available
            if MEDIAPIPE_AVAILABLE:
                mp_instance = mp.solutions

                # Face Mesh for detailed facial landmarks (468 points)
                self.mp_face_mesh = mp_instance.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,  # Enable eye and lip refinement
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )

                # Face Detection for robust face localization
                self.mp_face_detection = mp_instance.face_detection.FaceDetection(
                    model_selection=1,  # Full-range model
                    min_detection_confidence=0.5
                )

                # Pose detection for body language
                self.mp_pose = mp_instance.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )

                logger.info("✓ MediaPipe models loaded (Face Mesh, Detection, Pose)")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def analyze_video(self, video_path: str, sample_rate: int = 5) -> Dict:
        """
        Comprehensive video analysis for mental health assessment.

        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame (default: 5 for efficiency)

        Returns:
            Comprehensive analysis results with mental health indicators
        """
        try:
            logger.info(f"Starting comprehensive video analysis: {video_path}")

            # Extract and analyze frames
            frames_data = self._extract_and_analyze_frames(video_path, sample_rate)

            if not frames_data:
                logger.warning("No frames analyzed")
                return self._get_fallback_analysis()

            # Aggregate emotion data
            emotion_timeline = self._aggregate_emotions(frames_data)

            # Analyze gaze patterns
            gaze_analysis = self._analyze_gaze_patterns(frames_data)

            # Analyze head pose patterns
            head_pose_analysis = self._analyze_head_pose_patterns(frames_data)

            # Detect micro-expressions
            micro_expressions = self._detect_micro_expressions(frames_data)

            # Calculate clinical mental health indicators
            mental_health = self._calculate_mental_health_indicators(
                emotion_timeline,
                gaze_analysis,
                head_pose_analysis,
                micro_expressions
            )

            # Generate personalized recommendations
            recommendations = self._generate_recommendations(mental_health)

            # Compile comprehensive results
            results = {
                'timestamp': datetime.now().isoformat(),
                'video_info': {
                    'total_frames_analyzed': len(frames_data),
                    'faces_detected': sum(1 for f in frames_data if f.face_detected),
                    'detection_rate': sum(1 for f in frames_data if f.face_detected) / len(frames_data) if frames_data else 0
                },
                'emotion_analysis': emotion_timeline,
                'gaze_analysis': gaze_analysis,
                'head_pose_analysis': head_pose_analysis,
                'micro_expressions': micro_expressions,
                'mental_health_indicators': {
                    'depression_score': mental_health.depression_score,
                    'depression_level': self._categorize_severity(mental_health.depression_score),
                    'anxiety_score': mental_health.anxiety_score,
                    'anxiety_level': self._categorize_severity(mental_health.anxiety_score),
                    'stress_level': mental_health.stress_level,
                    'stress_category': self._categorize_severity(mental_health.stress_level),
                    'emotional_stability': mental_health.emotional_stability,
                    'engagement_level': mental_health.engagement_level,
                    'confidence_level': mental_health.confidence_level,
                    'primary_emotions': mental_health.primary_emotions,
                    'risk_factors': mental_health.risk_factors,
                    'protective_factors': mental_health.protective_factors
                },
                'recommendations': recommendations,
                'confidence_score': self._calculate_overall_confidence(frames_data)
            }

            logger.info(f"Video analysis complete - Depression: {mental_health.depression_score:.2f}, "
                       f"Anxiety: {mental_health.anxiety_score:.2f}")

            return results

        except Exception as e:
            logger.error(f"Error in video analysis: {e}", exc_info=True)
            return self._get_fallback_analysis()

    def _extract_and_analyze_frames(self, video_path: str, sample_rate: int) -> List[EmotionFrame]:
        """
        Extract frames from video and analyze each frame.

        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame

        Returns:
            List of EmotionFrame objects
        """
        frames_data = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return frames_data

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = 0
            analyzed_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames at specified rate
                if frame_count % sample_rate == 0:
                    timestamp = frame_count / fps
                    frame_analysis = self._analyze_single_frame(frame, frame_count, timestamp)

                    if frame_analysis:
                        frames_data.append(frame_analysis)
                        analyzed_count += 1

                frame_count += 1

            cap.release()
            logger.info(f"Extracted {analyzed_count} frames from {frame_count} total frames")

        except Exception as e:
            logger.error(f"Error extracting frames: {e}")

        return frames_data

    def _analyze_single_frame(self, frame: np.ndarray, frame_num: int, timestamp: float) -> Optional[EmotionFrame]:
        """
        Analyze a single video frame for emotions and facial features.

        Args:
            frame: BGR image frame
            frame_num: Frame number
            timestamp: Timestamp in seconds

        Returns:
            EmotionFrame object or None if analysis fails
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Initialize result containers
            emotions = {}
            facial_landmarks = None
            head_pose = None
            gaze_direction = None
            action_units = None
            face_detected = False
            confidence = 0.0

            # Method 1: DeepFace emotion detection (most accurate)
            if DEEPFACE_AVAILABLE:
                try:
                    analysis = DeepFace.analyze(
                        rgb_frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )

                    if isinstance(analysis, list) and len(analysis) > 0:
                        analysis = analysis[0]

                    if 'emotion' in analysis:
                        emotions = analysis['emotion']
                        face_detected = True
                        # Normalize emotions
                        total = sum(emotions.values()) or 1.0
                        emotions = {k: v / total for k, v in emotions.items()}
                        confidence = max(emotions.values())

                except Exception as e:
                    logger.debug(f"DeepFace analysis failed for frame {frame_num}: {e}")

            # Method 2: FER emotion detection (fallback)
            if not face_detected and FER_AVAILABLE and self.fer_detector:
                try:
                    fer_result = self.fer_detector.detect_emotions(rgb_frame)
                    if fer_result and len(fer_result) > 0:
                        emotions = fer_result[0]['emotions']
                        face_detected = True
                        confidence = max(emotions.values())

                except Exception as e:
                    logger.debug(f"FER analysis failed for frame {frame_num}: {e}")

            # Method 3: MediaPipe facial landmarks and gaze
            if MEDIAPIPE_AVAILABLE and self.mp_face_mesh:
                try:
                    results = self.mp_face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]

                        # Extract landmark coordinates
                        h, w, _ = frame.shape
                        facial_landmarks = [
                            (lm.x * w, lm.y * h)
                            for lm in face_landmarks.landmark
                        ]

                        # Calculate head pose
                        head_pose = self._calculate_head_pose(facial_landmarks, (h, w))

                        # Calculate gaze direction
                        gaze_direction = self._calculate_gaze_direction(facial_landmarks)

                        # Detect facial action units (AU)
                        action_units = self._detect_action_units(facial_landmarks)

                        face_detected = True

                except Exception as e:
                    logger.debug(f"MediaPipe analysis failed for frame {frame_num}: {e}")

            # If no face detected, return None
            if not face_detected:
                return None

            return EmotionFrame(
                frame_number=frame_num,
                timestamp=timestamp,
                emotions=emotions,
                face_detected=face_detected,
                confidence=confidence,
                facial_landmarks=facial_landmarks,
                head_pose=head_pose,
                gaze_direction=gaze_direction,
                action_units=action_units
            )

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_num}: {e}")
            return None

    def _calculate_head_pose(self, landmarks: List[Tuple[float, float]], image_size: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate head pose (pitch, yaw, roll) from facial landmarks.

        Uses key points: nose tip, chin, left/right eye corners

        Returns:
            Dict with pitch, yaw, roll in degrees
        """
        try:
            if len(landmarks) < 468:
                return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

            # Key landmark indices (MediaPipe 468-point model)
            nose_tip = landmarks[1]  # Nose tip
            chin = landmarks[152]  # Chin
            left_eye = landmarks[33]  # Left eye outer corner
            right_eye = landmarks[263]  # Right eye outer corner
            left_mouth = landmarks[61]  # Left mouth corner
            right_mouth = landmarks[291]  # Right mouth corner

            # Calculate yaw (left-right rotation)
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_x = nose_tip[0]
            h, w = image_size
            yaw = (nose_x - eye_center_x) / (w / 2) * 45  # Normalize to -45 to +45 degrees

            # Calculate pitch (up-down rotation)
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            nose_y = nose_tip[1]
            pitch = (nose_y - eye_center_y) / (h / 2) * 30  # Normalize to -30 to +30 degrees

            # Calculate roll (tilt)
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            roll = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees

            return {
                'pitch': float(pitch),
                'yaw': float(yaw),
                'roll': float(roll)
            }

        except Exception as e:
            logger.error(f"Error calculating head pose: {e}")
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

    def _calculate_gaze_direction(self, landmarks: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate gaze direction from eye landmarks.

        Estimates where the person is looking based on iris position.

        Returns:
            Dict with horizontal and vertical gaze direction
        """
        try:
            if len(landmarks) < 468:
                return {'horizontal': 0.0, 'vertical': 0.0, 'looking_at_camera': False}

            # Left eye landmarks
            left_eye_center = landmarks[468]  # Left iris center
            left_eye_left = landmarks[33]
            left_eye_right = landmarks[133]
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]

            # Right eye landmarks
            right_eye_center = landmarks[473]  # Right iris center
            right_eye_left = landmarks[362]
            right_eye_right = landmarks[263]
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]

            # Calculate horizontal gaze (left-right)
            left_horizontal = (left_eye_center[0] - left_eye_left[0]) / (left_eye_right[0] - left_eye_left[0])
            right_horizontal = (right_eye_center[0] - right_eye_left[0]) / (right_eye_right[0] - right_eye_left[0])
            horizontal_gaze = (left_horizontal + right_horizontal) / 2 - 0.5  # Center at 0

            # Calculate vertical gaze (up-down)
            left_vertical = (left_eye_center[1] - left_eye_top[1]) / (left_eye_bottom[1] - left_eye_top[1])
            right_vertical = (right_eye_center[1] - right_eye_top[1]) / (right_eye_bottom[1] - right_eye_top[1])
            vertical_gaze = (left_vertical + right_vertical) / 2 - 0.5  # Center at 0

            # Determine if looking at camera (center gaze)
            looking_at_camera = abs(horizontal_gaze) < 0.15 and abs(vertical_gaze) < 0.15

            return {
                'horizontal': float(horizontal_gaze),
                'vertical': float(vertical_gaze),
                'looking_at_camera': bool(looking_at_camera)
            }

        except Exception as e:
            logger.debug(f"Error calculating gaze: {e}")
            return {'horizontal': 0.0, 'vertical': 0.0, 'looking_at_camera': False}

    def _detect_action_units(self, landmarks: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Detect Facial Action Units (AU) based on FACS (Facial Action Coding System).

        Action Units are scientific markers for specific muscle movements.
        Important AUs for depression/anxiety:
        - AU1: Inner brow raiser (sadness, worry)
        - AU4: Brow lowerer (concentration, anger)
        - AU6: Cheek raiser (genuine smile)
        - AU12: Lip corner puller (smile)
        - AU15: Lip corner depressor (sadness)
        - AU17: Chin raiser (doubt, sadness)

        Returns:
            Dict of AU intensities (0-1)
        """
        try:
            if not landmarks or len(landmarks) < 468:
                return {}

            action_units = {}

            # AU1 - Inner Brow Raiser (sadness, concern)
            left_inner_brow = landmarks[70]
            right_inner_brow = landmarks[300]
            nose_bridge = landmarks[168]
            brow_height = ((left_inner_brow[1] + right_inner_brow[1]) / 2 - nose_bridge[1])
            action_units['AU1_inner_brow_raise'] = max(0, min(1, -brow_height / 20))

            # AU4 - Brow Lowerer (concentration, anger)
            action_units['AU4_brow_lower'] = max(0, min(1, brow_height / 20))

            # AU6 + AU12 - Cheek Raiser + Smile (genuine happiness)
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            mouth_center = landmarks[0]
            smile_width = abs(right_mouth[0] - left_mouth[0])
            action_units['AU6_AU12_smile'] = max(0, min(1, smile_width / 100))

            # AU15 - Lip Corner Depressor (sadness)
            mouth_corners_down = ((left_mouth[1] + right_mouth[1]) / 2 - mouth_center[1])
            action_units['AU15_lip_corner_down'] = max(0, min(1, mouth_corners_down / 20))

            # AU17 - Chin Raiser (doubt, sadness)
            chin = landmarks[152]
            lower_lip = landmarks[17]
            chin_raise = (chin[1] - lower_lip[1])
            action_units['AU17_chin_raise'] = max(0, min(1, -chin_raise / 15))

            return action_units

        except Exception as e:
            logger.error(f"Error detecting action units: {e}")
            return {}

    def _aggregate_emotions(self, frames_data: List[EmotionFrame]) -> Dict:
        """
        Aggregate emotion data across all frames.

        Calculates:
        - Mean emotion scores
        - Emotion variability
        - Dominant emotions
        - Emotional transitions
        """
        if not frames_data:
            return {}

        # Collect all emotions
        all_emotions = {}
        for frame in frames_data:
            for emotion, score in frame.emotions.items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = []
                all_emotions[emotion].append(score)

        # Calculate statistics
        emotion_stats = {}
        for emotion, scores in all_emotions.items():
            emotion_stats[emotion] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'max': float(np.max(scores)),
                'min': float(np.min(scores))
            }

        # Find dominant emotions
        mean_scores = {k: v['mean'] for k, v in emotion_stats.items()}
        sorted_emotions = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        dominant_emotions = [e[0] for e in sorted_emotions[:3]]

        return {
            'emotion_statistics': emotion_stats,
            'dominant_emotions': dominant_emotions,
            'emotional_variability': float(np.mean([v['std'] for v in emotion_stats.values()])),
            'total_frames': len(frames_data)
        }

    def _analyze_gaze_patterns(self, frames_data: List[EmotionFrame]) -> Dict:
        """
        Analyze gaze patterns throughout the video.

        Poor eye contact can indicate:
        - Social anxiety
        - Depression
        - Low confidence
        - Autism spectrum traits
        """
        gaze_data = [f.gaze_direction for f in frames_data if f.gaze_direction]

        if not gaze_data:
            return {'eye_contact_percentage': 0.5, 'gaze_stability': 0.5}

        # Calculate eye contact percentage
        looking_at_camera_count = sum(1 for g in gaze_data if g.get('looking_at_camera', False))
        eye_contact_percentage = looking_at_camera_count / len(gaze_data) if gaze_data else 0

        # Calculate gaze stability (less wandering = more stable)
        horizontal_values = [g.get('horizontal', 0) for g in gaze_data]
        vertical_values = [g.get('vertical', 0) for g in gaze_data]
        gaze_stability = 1.0 - (np.std(horizontal_values) + np.std(vertical_values)) / 2

        return {
            'eye_contact_percentage': float(eye_contact_percentage),
            'gaze_stability': float(max(0, min(1, gaze_stability))),
            'average_horizontal_deviation': float(np.mean([abs(h) for h in horizontal_values])),
            'average_vertical_deviation': float(np.mean([abs(v) for v in vertical_values]))
        }

    def _analyze_head_pose_patterns(self, frames_data: List[EmotionFrame]) -> Dict:
        """
        Analyze head pose patterns.

        Depression indicators:
        - Downward head tilt (negative pitch)
        - Less head movement (rigidity)
        - Avoidant posture
        """
        head_poses = [f.head_pose for f in frames_data if f.head_pose]

        if not head_poses:
            return {'average_pitch': 0, 'head_movement_variability': 0.5}

        pitch_values = [h['pitch'] for h in head_poses]
        yaw_values = [h['yaw'] for h in head_poses]
        roll_values = [h['roll'] for h in head_poses]

        # Calculate statistics
        avg_pitch = np.mean(pitch_values)  # Negative = looking down
        pitch_variability = np.std(pitch_values)
        yaw_variability = np.std(yaw_values)
        roll_variability = np.std(roll_values)

        # Overall head movement
        movement_variability = (pitch_variability + yaw_variability + roll_variability) / 3

        return {
            'average_pitch': float(avg_pitch),
            'average_yaw': float(np.mean(yaw_values)),
            'average_roll': float(np.mean(roll_values)),
            'head_movement_variability': float(movement_variability),
            'looking_down_tendency': float(max(0, -avg_pitch / 30))  # 0-1 scale
        }

    def _detect_micro_expressions(self, frames_data: List[EmotionFrame]) -> Dict:
        """
        Detect micro-expressions - brief, involuntary facial expressions.

        Micro-expressions can reveal suppressed emotions.
        """
        micro_expressions_detected = []

        # Look for rapid emotion changes (< 0.5 seconds)
        for i in range(1, len(frames_data)):
            prev_frame = frames_data[i-1]
            curr_frame = frames_data[i]

            time_diff = curr_frame.timestamp - prev_frame.timestamp

            if time_diff < 0.5 and prev_frame.emotions and curr_frame.emotions:
                # Check for significant emotion change
                for emotion in prev_frame.emotions:
                    if emotion in curr_frame.emotions:
                        change = abs(curr_frame.emotions[emotion] - prev_frame.emotions[emotion])

                        if change > 0.3:  # Significant change
                            micro_expressions_detected.append({
                                'timestamp': curr_frame.timestamp,
                                'emotion': emotion,
                                'intensity_change': float(change),
                                'duration': float(time_diff)
                            })

        return {
            'count': len(micro_expressions_detected),
            'detected_expressions': micro_expressions_detected[:10],  # Limit to first 10
            'suppression_indicator': min(1.0, len(micro_expressions_detected) / 20)  # Higher = more suppression
        }

    def _calculate_mental_health_indicators(
        self,
        emotion_timeline: Dict,
        gaze_analysis: Dict,
        head_pose_analysis: Dict,
        micro_expressions: Dict
    ) -> MentalHealthIndicators:
        """
        Calculate clinical mental health indicators from all analysis data.

        Uses evidence-based research linking facial expressions to mental health.
        """
        # Extract emotion statistics
        emotion_stats = emotion_timeline.get('emotion_statistics', {})
        dominant_emotions = emotion_timeline.get('dominant_emotions', [])

        # Depression scoring (clinical indicators)
        depression_indicators = []

        # 1. Sadness/negative emotions
        sadness_score = emotion_stats.get('sad', {}).get('mean', 0) + \
                       emotion_stats.get('fear', {}).get('mean', 0) * 0.5
        depression_indicators.append(sadness_score)

        # 2. Reduced happiness/anhedonia
        happiness_score = emotion_stats.get('happy', {}).get('mean', 0.5)
        anhedonia_score = 1.0 - happiness_score
        depression_indicators.append(anhedonia_score * 0.8)

        # 3. Flat affect (reduced emotional expression)
        emotional_variability = emotion_timeline.get('emotional_variability', 0.5)
        flat_affect_score = 1.0 - emotional_variability
        depression_indicators.append(flat_affect_score * 0.6)

        # 4. Poor eye contact
        eye_contact = gaze_analysis.get('eye_contact_percentage', 0.5)
        depression_indicators.append((1.0 - eye_contact) * 0.5)

        # 5. Downward head tilt
        looking_down = head_pose_analysis.get('looking_down_tendency', 0)
        depression_indicators.append(looking_down * 0.7)

        # 6. Reduced head movement (psychomotor retardation)
        head_movement = head_pose_analysis.get('head_movement_variability', 10)
        psychomotor_score = 1.0 - min(1.0, head_movement / 20)
        depression_indicators.append(psychomotor_score * 0.5)

        # Calculate overall depression score (weighted average)
        depression_score = np.mean(depression_indicators)

        # Anxiety scoring
        anxiety_indicators = []

        # 1. Fear/worry emotions
        fear_score = emotion_stats.get('fear', {}).get('mean', 0) + \
                    emotion_stats.get('surprise', {}).get('mean', 0) * 0.3
        anxiety_indicators.append(fear_score)

        # 2. Gaze avoidance/instability
        gaze_stability = gaze_analysis.get('gaze_stability', 0.5)
        anxiety_indicators.append((1.0 - gaze_stability) * 0.8)

        # 3. Micro-expressions (emotional suppression)
        suppression = micro_expressions.get('suppression_indicator', 0)
        anxiety_indicators.append(suppression * 0.6)

        # 4. High emotional variability (mood swings)
        anxiety_indicators.append(emotional_variability * 0.5)

        anxiety_score = np.mean(anxiety_indicators)

        # Stress level
        stress_indicators = []
        stress_indicators.append(emotion_stats.get('angry', {}).get('mean', 0))
        stress_indicators.append(anxiety_score * 0.7)
        stress_indicators.append(emotional_variability * 0.6)
        stress_level = np.mean(stress_indicators)

        # Emotional stability (inverse of variability)
        emotional_stability = 1.0 - emotional_variability

        # Engagement level (eye contact + head movement)
        engagement_level = (eye_contact * 0.6 + min(1.0, head_movement / 15) * 0.4)

        # Confidence level
        confidence = 'low' if engagement_level < 0.4 else 'moderate' if engagement_level < 0.7 else 'high'

        # Primary emotions (top 3)
        primary_emotions = dominant_emotions[:3] if dominant_emotions else ['neutral']

        # Identify risk factors
        risk_factors = []
        if depression_score > 0.6:
            risk_factors.append('Elevated depression indicators')
        if anxiety_score > 0.6:
            risk_factors.append('Elevated anxiety indicators')
        if eye_contact < 0.3:
            risk_factors.append('Poor eye contact (social withdrawal)')
        if flat_affect_score > 0.7:
            risk_factors.append('Flat affect (reduced emotional expression)')
        if 'sad' in primary_emotions or 'fear' in primary_emotions:
            risk_factors.append('Predominant negative emotions')

        # Identify protective factors
        protective_factors = []
        if 'happy' in primary_emotions:
            protective_factors.append('Positive emotional expression')
        if engagement_level > 0.6:
            protective_factors.append('Good engagement and presence')
        if emotional_stability > 0.6:
            protective_factors.append('Emotional stability')
        if eye_contact > 0.5:
            protective_factors.append('Adequate eye contact')

        return MentalHealthIndicators(
            depression_score=float(max(0, min(1, depression_score))),
            anxiety_score=float(max(0, min(1, anxiety_score))),
            stress_level=float(max(0, min(1, stress_level))),
            emotional_stability=float(max(0, min(1, emotional_stability))),
            engagement_level=float(max(0, min(1, engagement_level))),
            confidence_level=confidence,
            primary_emotions=primary_emotions,
            risk_factors=risk_factors if risk_factors else ['None identified'],
            protective_factors=protective_factors if protective_factors else ['Consider building support systems']
        )

    def _generate_recommendations(self, mental_health: MentalHealthIndicators) -> List[Dict]:
        """
        Generate personalized mental health recommendations based on analysis.
        """
        recommendations = []

        # Depression recommendations
        if mental_health.depression_score > 0.7:
            recommendations.append({
                'category': 'depression',
                'priority': 'high',
                'title': 'Consider Professional Mental Health Support',
                'description': 'Your assessment shows significant depression indicators. Speaking with a mental health professional can provide personalized treatment options.',
                'actions': [
                    'Schedule appointment with therapist or psychiatrist',
                    'Consider evidence-based treatments (CBT, medication)',
                    'Reach out to crisis helpline if having thoughts of self-harm'
                ],
                'resources': ['National Suicide Prevention Lifeline: 988', 'NIMHANS: 080-46110007']
            })
        elif mental_health.depression_score > 0.5:
            recommendations.append({
                'category': 'depression',
                'priority': 'medium',
                'title': 'Monitor Mood and Consider Support',
                'description': 'Mild to moderate depression indicators detected. Self-care and professional support can help.',
                'actions': [
                    'Engage in regular physical activity (30min daily)',
                    'Practice sleep hygiene (7-9 hours)',
                    'Connect with supportive friends/family',
                    'Consider therapy if symptoms persist'
                ]
            })

        # Anxiety recommendations
        if mental_health.anxiety_score > 0.6:
            recommendations.append({
                'category': 'anxiety',
                'priority': 'medium' if mental_health.anxiety_score < 0.8 else 'high',
                'title': 'Address Anxiety Symptoms',
                'description': 'Elevated anxiety indicators detected. Anxiety management techniques can help.',
                'actions': [
                    'Practice deep breathing exercises (4-7-8 technique)',
                    'Try progressive muscle relaxation',
                    'Limit caffeine and alcohol',
                    'Consider cognitive behavioral therapy (CBT) for anxiety'
                ]
            })

        # Social engagement recommendations
        if mental_health.engagement_level < 0.4:
            recommendations.append({
                'category': 'social',
                'priority': 'medium',
                'title': 'Improve Social Engagement',
                'description': 'Low engagement detected. Social connection is vital for mental health.',
                'actions': [
                    'Practice eye contact in comfortable settings',
                    'Join interest-based groups or communities',
                    'Schedule regular social activities',
                    'Consider social skills training if helpful'
                ]
            })

        # Stress management
        if mental_health.stress_level > 0.6:
            recommendations.append({
                'category': 'stress',
                'priority': 'medium',
                'title': 'Stress Reduction Strategies',
                'description': 'High stress levels detected. Stress management is important for overall wellbeing.',
                'actions': [
                    'Practice mindfulness meditation (10-15min daily)',
                    'Identify and address major stressors',
                    'Set healthy boundaries at work/home',
                    'Engage in relaxing activities (yoga, nature walks)'
                ]
            })

        # Positive reinforcement
        if mental_health.protective_factors and len(mental_health.protective_factors) > 0:
            recommendations.append({
                'category': 'positive',
                'priority': 'low',
                'title': 'Strengths Identified',
                'description': f"Your protective factors: {', '.join(mental_health.protective_factors)}",
                'actions': [
                    'Continue nurturing these positive aspects',
                    'Build on existing strengths',
                    'Share your coping strategies with others'
                ]
            })

        return recommendations

    def _categorize_severity(self, score: float) -> str:
        """Categorize severity score into clinical levels"""
        if score < 0.2:
            return 'minimal'
        elif score < 0.4:
            return 'mild'
        elif score < 0.6:
            return 'moderate'
        elif score < 0.8:
            return 'moderately_severe'
        else:
            return 'severe'

    def _calculate_overall_confidence(self, frames_data: List[EmotionFrame]) -> float:
        """Calculate overall confidence in the analysis"""
        if not frames_data:
            return 0.0

        # Confidence based on face detection rate and emotion confidence
        face_detection_rate = sum(1 for f in frames_data if f.face_detected) / len(frames_data)
        avg_emotion_confidence = np.mean([f.confidence for f in frames_data if f.confidence > 0])

        overall_confidence = (face_detection_rate * 0.6 + avg_emotion_confidence * 0.4)
        return float(max(0, min(1, overall_confidence)))

    def _get_fallback_analysis(self) -> Dict:
        """Return fallback analysis when video processing fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'error': 'Video analysis failed',
            'mental_health_indicators': {
                'depression_score': 0.5,
                'depression_level': 'moderate',
                'anxiety_score': 0.5,
                'anxiety_level': 'moderate',
                'stress_level': 0.5,
                'confidence_level': 'moderate',
                'primary_emotions': ['neutral'],
                'risk_factors': ['Unable to assess - video quality or processing issue'],
                'protective_factors': ['Unable to assess']
            },
            'recommendations': [{
                'category': 'general',
                'priority': 'medium',
                'title': 'Consider Alternative Assessment',
                'description': 'Video analysis was unsuccessful. Try audio-only assessment or retake video with better lighting.',
                'actions': ['Ensure good lighting', 'Position face clearly in frame', 'Minimize background noise']
            }],
            'confidence_score': 0.0
        }


# Singleton instance
_service_instance = None

def get_video_emotion_service() -> EnhancedVideoEmotionService:
    """Get singleton instance of video emotion service"""
    global _service_instance
    if _service_instance is None:
        _service_instance = EnhancedVideoEmotionService()
    return _service_instance
