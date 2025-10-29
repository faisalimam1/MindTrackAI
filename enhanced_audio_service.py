#!/usr/bin/env python3
"""
Enhanced Audio Analysis Service with Crisis Detection and Intent Analysis
Integrates advanced crisis detection, emotion intensity, and urgency analysis
"""

import logging
from typing import Dict, List
from datetime import datetime

# Import base video_audio_service for audio feature extraction and transcription
try:
    from video_audio_service import VideoAudioAnalysisService
    BASE_SERVICE_AVAILABLE = True
except ImportError:
    BASE_SERVICE_AVAILABLE = False
    logging.warning("Base video_audio_service not available")

# Import crisis detector
try:
    from audio_crisis_detector import (
        get_crisis_detector,
        detect_crisis_from_text,
        analyze_emotion_intensity,
        detect_speech_urgency
    )
    CRISIS_DETECTOR_AVAILABLE = True
except ImportError:
    CRISIS_DETECTOR_AVAILABLE = False
    logging.warning("Crisis detector not available")

# Import ML services for crisis detection integration
try:
    from ml_services import detect_crisis as ml_detect_crisis
    ML_SERVICES_AVAILABLE = True
except ImportError:
    ML_SERVICES_AVAILABLE = False
    logging.warning("ML services not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAudioAnalysisService:
    """
    Enhanced audio analysis service with comprehensive crisis detection
    and mental health assessment capabilities
    """

    def __init__(self):
        """Initialize the enhanced audio analysis service"""
        self.base_service = VideoAudioAnalysisService() if BASE_SERVICE_AVAILABLE else None
        self.crisis_detector = get_crisis_detector() if CRISIS_DETECTOR_AVAILABLE else None
        logger.info("Enhanced audio analysis service initialized")

    def analyze_audio_comprehensive(self, audio_data: bytes) -> Dict:
        """
        Comprehensive audio analysis with crisis detection, emotion intensity,
        and urgency detection

        Args:
            audio_data: Raw audio bytes

        Returns:
            Dict with comprehensive analysis results including safety recommendations
        """
        try:
            logger.info(f"Starting comprehensive audio analysis ({len(audio_data)} bytes)")

            # Initialize result structure
            result = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'enhanced_audio_comprehensive',
                'processing_successful': False
            }

            # Validate audio data
            if not audio_data or len(audio_data) < 1000:
                logger.error("Invalid or insufficient audio data")
                return self._get_error_response('insufficient_audio_data')

            # Step 1: Extract audio features and transcription
            if not self.base_service:
                logger.error("Base service not available")
                return self._get_error_response('service_unavailable')

            # Extract comprehensive audio features
            audio_features = self.base_service._extract_audio_features(audio_data)
            logger.info(f"Audio features extracted: {len(audio_features)} features")

            # Check if extraction actually worked
            if audio_features.get('extraction_failed', False):
                logger.error("⚠️  Audio feature extraction FAILED - using fallback values")
                logger.error("⚠️  Results will not be accurate")
                result.update({
                    'processing_successful': True,
                    'extraction_warning': True,
                    'warning_message': 'Audio processing partially failed. Results may not be accurate. Please try recording again with clear audio.',
                    'extraction_method_failed': True
                })

            # Transcribe audio to text
            transcribed_text = self.base_service._transcribe_audio(audio_data)
            logger.info(f"Audio transcribed: {len(transcribed_text)} characters")

            # Handle transcription failures gracefully
            transcription_successful = (
                transcribed_text and
                len(transcribed_text) > 10 and
                'error' not in transcribed_text.lower() and
                'could not' not in transcribed_text.lower()
            )

            # Step 2: Perform sentiment analysis
            sentiment_analysis = {}
            if transcription_successful:
                sentiment_analysis = self.base_service._analyze_speech_sentiment(transcribed_text)
            else:
                logger.warning("Transcription failed or unclear, using voice features only")
                sentiment_analysis = {
                    'polarity': 0.0,
                    'subjectivity': 0.5,
                    'sentiment_label': 'neutral',
                    'emotional_words': []
                }

            # Step 3: CRITICAL - Crisis detection with enhanced analyzer
            crisis_result = self._analyze_crisis_comprehensive(
                transcribed_text if transcription_successful else "",
                audio_features,
                sentiment_analysis
            )

            # Step 4: Emotion intensity analysis
            emotion_intensity = {}
            if transcription_successful and CRISIS_DETECTOR_AVAILABLE:
                emotion_intensity = analyze_emotion_intensity(
                    transcribed_text,
                    sentiment_analysis.get('polarity', 0.0)
                )
            else:
                emotion_intensity = self._get_fallback_emotion_intensity()

            # Step 5: Speech urgency detection
            urgency_analysis = {}
            if CRISIS_DETECTOR_AVAILABLE:
                urgency_analysis = detect_speech_urgency(
                    transcribed_text if transcription_successful else "",
                    audio_features.get('speaking_rate', 0.5),
                    audio_features.get('pause_frequency', 0.5)
                )
            else:
                urgency_analysis = self._get_fallback_urgency()

            # Step 6: Calculate enhanced depression score
            depression_analysis = self._calculate_enhanced_depression_score(
                audio_features,
                sentiment_analysis,
                crisis_result,
                emotion_intensity,
                urgency_analysis,
                transcribed_text if transcription_successful else ""
            )

            # Step 7: Calculate confidence score
            confidence_analysis = self._calculate_confidence_score(
                audio_features,
                sentiment_analysis,
                emotion_intensity
            )

            # Step 8: Generate safety-focused recommendations
            recommendations = self._generate_safety_recommendations(
                crisis_result,
                depression_analysis,
                confidence_analysis,
                urgency_analysis
            )

            # Step 9: Determine overall wellbeing and risk level
            overall_assessment = self._assess_overall_status(
                depression_analysis,
                confidence_analysis,
                crisis_result
            )

            # Compile comprehensive result
            result.update({
                'processing_successful': True,
                'transcription_successful': transcription_successful,
                'transcribed_text': transcribed_text[:500] + '...' if len(transcribed_text) > 500 else transcribed_text,
                'transcription_length': len(transcribed_text),

                # Core assessments
                'crisis_assessment': crisis_result,
                'depression_analysis': depression_analysis,
                'confidence_analysis': confidence_analysis,
                'emotion_intensity': emotion_intensity,
                'urgency_analysis': urgency_analysis,

                # Overall metrics (for frontend convenience)
                'depression_score': depression_analysis['score'],
                'depression_level': depression_analysis['level'],
                'confidence_score': confidence_analysis['score'],
                'confidence_level': confidence_analysis['level'],
                'overall_wellbeing': overall_assessment['wellbeing'],
                'risk_level': overall_assessment['risk_level'],

                # Safety and intervention
                'requires_immediate_intervention': crisis_result['requires_immediate_intervention'],
                'requires_professional_followup': crisis_result['requires_professional_followup'],
                'safety_recommendations': recommendations,

                # Technical details
                'audio_features': {
                    'pitch_variation': audio_features.get('pitch_std', 0.5),
                    'energy_level': audio_features.get('energy_mean', 0.5),
                    'speaking_rate': audio_features.get('speaking_rate', 0.5),
                    'pause_frequency': audio_features.get('pause_frequency', 0.5),
                    'is_silent': audio_features.get('is_silent', False)
                },
                'sentiment_analysis': sentiment_analysis,

                # Processing metadata
                'analysis_confidence': self._calculate_analysis_confidence(
                    transcription_successful,
                    audio_features,
                    crisis_result
                ),
                'processing_info': {
                    'base_service_available': BASE_SERVICE_AVAILABLE,
                    'crisis_detector_available': CRISIS_DETECTOR_AVAILABLE,
                    'ml_services_available': ML_SERVICES_AVAILABLE,
                    'transcription_method': 'Google/Whisper/Sphinx',
                    'crisis_detection_enabled': True
                }
            })

            logger.info(f"Comprehensive analysis complete: crisis={crisis_result['is_crisis']}, "
                       f"depression={depression_analysis['score']:.3f}, "
                       f"risk={overall_assessment['risk_level']}")

            return result

        except Exception as e:
            logger.error(f"Error in comprehensive audio analysis: {e}", exc_info=True)
            return self._get_error_response('analysis_exception', str(e))

    def _analyze_crisis_comprehensive(
        self,
        text: str,
        audio_features: Dict,
        sentiment: Dict
    ) -> Dict:
        """
        Comprehensive crisis analysis combining text, audio, and sentiment

        Args:
            text: Transcribed text
            audio_features: Voice feature analysis
            sentiment: Sentiment analysis results

        Returns:
            Dict with crisis assessment
        """
        # Initialize crisis result
        crisis_result = {
            'is_crisis': False,
            'severity_score': 0.0,
            'confidence': 0.0,
            'requires_immediate_intervention': False,
            'requires_professional_followup': False,
            'risk_level': 'LOW',
            'crisis_indicators': [],
            'detection_method': 'none'
        }

        # Method 1: Text-based crisis detection (most reliable)
        if text and len(text) > 10 and CRISIS_DETECTOR_AVAILABLE:
            is_crisis_text, crisis_confidence, crisis_details = detect_crisis_from_text(text)

            if is_crisis_text:
                crisis_result.update({
                    'is_crisis': True,
                    'severity_score': crisis_details['severity_score'],
                    'confidence': crisis_confidence,
                    'requires_immediate_intervention': crisis_details['requires_immediate_intervention'],
                    'requires_professional_followup': crisis_details['requires_professional_followup'],
                    'risk_level': crisis_details['risk_level'],
                    'crisis_indicators': crisis_details['indicators'],
                    'detection_method': 'text_analysis',
                    'base_severity': crisis_details['base_severity'],
                    'urgency_multiplier': crisis_details['urgency_multiplier'],
                    'text_sample': crisis_details.get('text_sample', '')
                })
                return crisis_result

        # Method 2: ML services crisis detection (fallback)
        if text and ML_SERVICES_AVAILABLE:
            try:
                is_crisis_ml, ml_confidence = ml_detect_crisis(text)
                if is_crisis_ml:
                    crisis_result.update({
                        'is_crisis': True,
                        'severity_score': ml_confidence,
                        'confidence': ml_confidence,
                        'requires_immediate_intervention': ml_confidence >= 0.85,
                        'requires_professional_followup': ml_confidence >= 0.65,
                        'risk_level': self._categorize_risk_from_score(ml_confidence),
                        'detection_method': 'ml_services'
                    })
                    return crisis_result
            except Exception as e:
                logger.warning(f"ML services crisis detection failed: {e}")

        # Method 3: Voice-based crisis indicators
        voice_crisis_score = self._detect_crisis_from_voice(audio_features, sentiment)
        if voice_crisis_score >= 0.65:
            crisis_result.update({
                'is_crisis': True,
                'severity_score': voice_crisis_score,
                'confidence': 0.7,
                'requires_immediate_intervention': voice_crisis_score >= 0.85,
                'requires_professional_followup': voice_crisis_score >= 0.65,
                'risk_level': self._categorize_risk_from_score(voice_crisis_score),
                'detection_method': 'voice_features',
                'voice_indicators': self._get_voice_crisis_indicators(audio_features, sentiment)
            })

        return crisis_result

    def _detect_crisis_from_voice(self, audio_features: Dict, sentiment: Dict) -> float:
        """
        Detect crisis indicators from voice features alone

        Voice patterns in crisis:
        - Very low energy (extreme fatigue/depression)
        - Monotone speech (flat affect)
        - Very slow speaking rate or very fast (agitation)
        - Excessive pauses (cognitive impairment/distress)
        - Highly negative sentiment

        Returns:
            Crisis score (0-1)
        """
        if audio_features.get('is_silent', False) or audio_features.get('extraction_failed', False):
            return 0.0

        crisis_score = 0.0

        # Extreme low energy (severe depression indicator)
        energy = audio_features.get('energy_mean', 0.5)
        if energy < 0.15:
            crisis_score += 0.25
        elif energy < 0.25:
            crisis_score += 0.15

        # Monotone speech (flat affect)
        pitch_std = audio_features.get('pitch_std', 0.5)
        if pitch_std < 0.15:
            crisis_score += 0.20
        elif pitch_std < 0.25:
            crisis_score += 0.10

        # Abnormal speaking rate
        speaking_rate = audio_features.get('speaking_rate', 0.5)
        if speaking_rate < 0.20 or speaking_rate > 0.90:
            crisis_score += 0.15

        # Excessive pauses (severe hesitation/distress)
        pause_freq = audio_features.get('pause_frequency', 0.5)
        if pause_freq > 0.75:
            crisis_score += 0.20
        elif pause_freq > 0.65:
            crisis_score += 0.10

        # Voice quality deterioration
        jitter = audio_features.get('jitter', 0.5)
        shimmer = audio_features.get('shimmer', 0.5)
        if jitter > 0.70 and shimmer > 0.70:
            crisis_score += 0.15

        # Highly negative sentiment
        polarity = sentiment.get('polarity', 0.0)
        if polarity < -0.5:
            crisis_score += 0.20
        elif polarity < -0.3:
            crisis_score += 0.10

        return min(crisis_score, 1.0)

    def _get_voice_crisis_indicators(self, audio_features: Dict, sentiment: Dict) -> List[str]:
        """Identify specific voice-based crisis indicators"""
        indicators = []

        if audio_features.get('energy_mean', 0.5) < 0.20:
            indicators.append("Extremely low vocal energy (severe fatigue/depression)")
        if audio_features.get('pitch_std', 0.5) < 0.20:
            indicators.append("Highly monotone speech (flat affect)")
        if audio_features.get('speaking_rate', 0.5) < 0.20:
            indicators.append("Very slow speech (psychomotor retardation)")
        if audio_features.get('pause_frequency', 0.5) > 0.75:
            indicators.append("Excessive pauses (severe distress/impairment)")
        if sentiment.get('polarity', 0.0) < -0.5:
            indicators.append("Extremely negative emotional tone")

        return indicators if indicators else ["Voice-based distress patterns detected"]

    def _calculate_enhanced_depression_score(
        self,
        audio_features: Dict,
        sentiment: Dict,
        crisis_result: Dict,
        emotion_intensity: Dict,
        urgency: Dict,
        text: str
    ) -> Dict:
        """
        Calculate enhanced depression score with crisis integration

        High-risk phrases significantly boost depression score
        """
        # Get base depression score from voice
        if self.base_service:
            base_score = self.base_service._calculate_voice_depression_score(
                audio_features,
                sentiment
            )
            # If base_score is None, audio is silent or invalid
            if base_score is None:
                return None  # Propagate the error
        else:
            base_score = 0.5

        # If extraction failed, use text-based analysis primarily
        if audio_features.get('extraction_failed', False):
            logger.warning("Audio extraction failed - relying on text analysis")

            # Analyze sentiment polarity
            polarity = sentiment.get('polarity', 0.0)

            # Base score from sentiment
            if polarity < -0.5:
                base_score = 0.70  # Strong negative sentiment
            elif polarity < -0.3:
                base_score = 0.60  # Moderate negative
            elif polarity < -0.1:
                base_score = 0.45  # Mild negative
            elif polarity < 0.1:
                base_score = 0.35  # Neutral
            else:
                base_score = 0.25  # Positive

            # Adjust for text length (very short text = less reliable)
            if len(text) < 20:
                base_score = 0.50  # Neutral/uncertain
                logger.warning("Transcription too short for reliable analysis")

        # CRITICAL: Crisis detection dramatically increases depression score
        if crisis_result['is_crisis']:
            crisis_boost = crisis_result['severity_score'] * 0.4
            base_score = min(base_score + crisis_boost, 1.0)
            logger.warning(f"Crisis detected - depression score boosted by {crisis_boost:.3f}")

        # High emotion intensity with negative sentiment increases score
        if emotion_intensity.get('intensity_score', 0.5) > 0.75 and sentiment.get('polarity', 0.0) < -0.2:
            intensity_boost = 0.15
            base_score = min(base_score + intensity_boost, 1.0)

        # High urgency with distress increases score
        if urgency.get('urgency_score', 0.5) > 0.70 and base_score > 0.50:
            urgency_boost = 0.10
            base_score = min(base_score + urgency_boost, 1.0)

        # Categorize depression level
        if base_score >= 0.75:
            level = 'severe'
        elif base_score >= 0.55:
            level = 'moderate'
        elif base_score >= 0.35:
            level = 'mild'
        else:
            level = 'minimal'

        # Identify contributing factors
        factors = []
        if crisis_result['is_crisis']:
            factors.extend(crisis_result.get('crisis_indicators', []))
        if self.base_service:
            factors.extend(self.base_service._identify_voice_depression_factors(audio_features))

        return {
            'score': round(base_score, 3),
            'level': level,
            'factors': factors[:5],  # Top 5 factors
            'crisis_influenced': crisis_result['is_crisis'],
            'clinical_severity': self._map_to_clinical_severity(base_score)
        }

    def _calculate_confidence_score(
        self,
        audio_features: Dict,
        sentiment: Dict,
        emotion_intensity: Dict
    ) -> Dict:
        """Calculate confidence score from audio features and sentiment"""
        if self.base_service:
            base_score = self.base_service._calculate_voice_confidence_score(
                audio_features,
                sentiment
            )
        else:
            base_score = 0.5

        # Low emotion intensity might indicate low confidence
        if emotion_intensity.get('intensity_score', 0.5) < 0.35:
            base_score = max(base_score - 0.10, 0.0)

        # Categorize confidence level
        if base_score >= 0.70:
            level = 'high'
        elif base_score >= 0.45:
            level = 'moderate'
        else:
            level = 'low'

        return {
            'score': round(base_score, 3),
            'level': level,
            'factors': self.base_service._identify_voice_confidence_factors(audio_features) if self.base_service else []
        }

    def _generate_safety_recommendations(
        self,
        crisis_result: Dict,
        depression: Dict,
        confidence: Dict,
        urgency: Dict
    ) -> Dict:
        """
        Generate safety-focused recommendations based on risk level

        CRITICAL: High-risk cases receive immediate intervention recommendations
        Moderate cases receive professional consultation recommendations
        Low risk cases receive wellness suggestions
        """
        recommendations = {
            'primary_actions': [],
            'secondary_actions': [],
            'self_care_suggestions': [],
            'professional_resources': []
        }

        # ===== CRITICAL RISK: IMMEDIATE INTERVENTION =====
        if crisis_result['requires_immediate_intervention']:
            recommendations['primary_actions'] = [
                {
                    'priority': 'CRITICAL',
                    'action': 'Contact Emergency Services Immediately',
                    'description': 'If you are in immediate danger or having thoughts of harming yourself, please contact emergency services or a crisis hotline RIGHT NOW.',
                    'urgent': True
                },
                {
                    'priority': 'CRITICAL',
                    'action': 'Crisis Helpline',
                    'description': 'Call NIMHANS 24/7 Helpline: 080-46110007 or iCall: 9152987821 (Mon-Sat, 8 AM - 10 PM) or Vandrevala Foundation: 1860-2662-345 (24/7)',
                    'urgent': True
                },
                {
                    'priority': 'CRITICAL',
                    'action': 'Do Not Stay Alone',
                    'description': 'Reach out to a trusted friend, family member, or go to the nearest hospital emergency room immediately.',
                    'urgent': True
                }
            ]

            recommendations['professional_resources'] = [
                {
                    'type': 'Crisis Helpline',
                    'name': 'NIMHANS Helpline',
                    'contact': '080-46110007',
                    'availability': '24/7',
                    'urgent': True
                },
                {
                    'type': 'Crisis Helpline',
                    'name': 'iCall - Suicide Prevention',
                    'contact': '9152987821',
                    'availability': 'Monday to Saturday, 8 AM - 10 PM',
                    'urgent': True
                },
                {
                    'type': 'Crisis Helpline',
                    'name': 'Vandrevala Foundation',
                    'contact': '1860-2662-345',
                    'availability': '24/7',
                    'urgent': True
                },
                {
                    'type': 'Emergency',
                    'name': 'Nearest Hospital Emergency Room',
                    'contact': 'Go in person immediately',
                    'urgent': True
                }
            ]

            # NO self-care suggestions for critical cases
            return recommendations

        # ===== HIGH RISK: URGENT PROFESSIONAL HELP =====
        if crisis_result['requires_professional_followup'] or depression['level'] == 'severe':
            recommendations['primary_actions'] = [
                {
                    'priority': 'HIGH',
                    'action': 'Schedule Immediate Mental Health Consultation',
                    'description': 'Contact a psychiatrist or psychologist within the next 24-48 hours. Your symptoms indicate you need professional evaluation.',
                    'urgent': True
                },
                {
                    'priority': 'HIGH',
                    'action': 'Inform Trusted Support Person',
                    'description': 'Tell a family member, close friend, or trusted person about how you\'re feeling. Do not face this alone.',
                    'urgent': False
                },
                {
                    'priority': 'HIGH',
                    'action': 'Monitor Your Symptoms',
                    'description': 'If your symptoms worsen or you develop thoughts of self-harm, contact a crisis helpline immediately.',
                    'urgent': False
                }
            ]

            recommendations['professional_resources'] = [
                {
                    'type': 'Mental Health Professional',
                    'name': 'Psychiatrist',
                    'description': 'Medical doctor specializing in mental health who can prescribe medication',
                    'urgency': 'high'
                },
                {
                    'type': 'Mental Health Professional',
                    'name': 'Clinical Psychologist',
                    'description': 'Licensed therapist specializing in mental health treatment',
                    'urgency': 'high'
                },
                {
                    'type': 'Support',
                    'name': 'Mental Health Helpline',
                    'contact': 'iCall: 9152987821',
                    'urgency': 'moderate'
                }
            ]

            # Limited self-care for high risk (not primary focus)
            recommendations['self_care_suggestions'] = [
                'Maintain regular sleep schedule',
                'Avoid isolation - stay connected with supportive people',
                'Avoid alcohol and substance use'
            ]

            return recommendations

        # ===== MODERATE RISK: PROFESSIONAL CONSULTATION RECOMMENDED =====
        if depression['level'] in ['moderate', 'mild']:
            recommendations['primary_actions'] = [
                {
                    'priority': 'MODERATE',
                    'action': 'Consider Professional Therapy',
                    'description': 'Schedule an appointment with a mental health professional to discuss your symptoms and treatment options.',
                    'urgent': False
                },
                {
                    'priority': 'MODERATE',
                    'action': 'Connect with Support Network',
                    'description': 'Reach out to friends, family, or support groups. Social connection is crucial for recovery.',
                    'urgent': False
                }
            ]

            recommendations['secondary_actions'] = [
                'Start daily journaling to track mood patterns',
                'Establish consistent sleep and wake times',
                'Engage in regular physical activity (even light walking)',
                'Practice stress-reduction techniques (meditation, deep breathing)'
            ]

            recommendations['professional_resources'] = [
                {
                    'type': 'Therapy',
                    'name': 'Cognitive Behavioral Therapy (CBT)',
                    'description': 'Evidence-based treatment for depression and anxiety'
                },
                {
                    'type': 'Support',
                    'name': 'Support Groups',
                    'description': 'Connect with others experiencing similar challenges'
                }
            ]

            recommendations['self_care_suggestions'] = [
                'Practice daily mindfulness or meditation (10-15 minutes)',
                'Maintain social connections - schedule regular contact with friends/family',
                'Get 7-9 hours of sleep per night',
                'Exercise for 30 minutes, 3-5 times per week',
                'Eat balanced, nutritious meals',
                'Limit alcohol and caffeine intake',
                'Engage in enjoyable hobbies and activities'
            ]

            return recommendations

        # ===== LOW RISK: WELLNESS AND PREVENTION =====
        recommendations['primary_actions'] = [
            {
                'priority': 'LOW',
                'action': 'Continue Positive Mental Health Practices',
                'description': 'Maintain your current healthy habits and continue monitoring your wellbeing.',
                'urgent': False
            }
        ]

        recommendations['self_care_suggestions'] = [
            'Continue regular self-assessment and monitoring',
            'Maintain healthy sleep, exercise, and nutrition habits',
            'Stay socially connected with friends and family',
            'Practice stress management techniques',
            'Engage in activities that bring you joy and fulfillment',
            'Build resilience through learning new skills or hobbies',
            'Help others in your community (volunteering can boost wellbeing)'
        ]

        recommendations['professional_resources'] = [
            {
                'type': 'Preventive',
                'name': 'Annual Mental Health Checkup',
                'description': 'Consider scheduling a preventive mental health consultation'
            }
        ]

        return recommendations

    def _assess_overall_status(
        self,
        depression: Dict,
        confidence: Dict,
        crisis: Dict
    ) -> Dict:
        """Assess overall mental health status and risk level"""

        # Determine risk level
        if crisis['requires_immediate_intervention']:
            risk_level = 'CRITICAL - Immediate intervention required'
            wellbeing = 'critical'
        elif crisis['requires_professional_followup'] or depression['level'] == 'severe':
            risk_level = 'HIGH - Urgent professional help needed'
            wellbeing = 'concerning'
        elif depression['level'] == 'moderate':
            risk_level = 'MODERATE - Professional consultation recommended'
            wellbeing = 'fair'
        elif depression['level'] == 'mild':
            risk_level = 'MILD - Monitor and provide support'
            wellbeing = 'fair'
        else:
            risk_level = 'LOW - General wellness support'
            wellbeing = 'good'

        return {
            'risk_level': risk_level,
            'wellbeing': wellbeing,
            'requires_monitoring': depression['score'] >= 0.35,
            'is_safe': not crisis['is_crisis'] and depression['level'] != 'severe'
        }

    def _map_to_clinical_severity(self, score: float) -> str:
        """Map score to clinical depression severity levels"""
        if score >= 0.85:
            return 'Severe (consider PHQ-9 score 20+)'
        elif score >= 0.70:
            return 'Moderately Severe (PHQ-9 15-19)'
        elif score >= 0.55:
            return 'Moderate (PHQ-9 10-14)'
        elif score >= 0.35:
            return 'Mild (PHQ-9 5-9)'
        else:
            return 'Minimal/None (PHQ-9 0-4)'

    def _categorize_risk_from_score(self, score: float) -> str:
        """Categorize risk level from score"""
        if score >= 0.85:
            return 'CRITICAL - Immediate intervention required'
        elif score >= 0.70:
            return 'HIGH - Urgent professional help needed'
        elif score >= 0.50:
            return 'MODERATE - Professional consultation recommended'
        elif score >= 0.30:
            return 'MILD - Monitor and provide support'
        else:
            return 'LOW - General support sufficient'

    def _calculate_analysis_confidence(
        self,
        transcription_successful: bool,
        audio_features: Dict,
        crisis_result: Dict
    ) -> float:
        """Calculate confidence in the analysis results"""
        confidence = 0.5

        if transcription_successful:
            confidence += 0.3
        if not audio_features.get('is_silent', False) and not audio_features.get('extraction_failed', False):
            confidence += 0.2
        if crisis_result['detection_method'] in ['text_analysis', 'ml_services']:
            confidence += 0.1

        return round(min(confidence, 1.0), 2)

    def _get_fallback_emotion_intensity(self) -> Dict:
        """Fallback emotion intensity when detection unavailable"""
        return {
            'intensity_score': 0.5,
            'intensity_level': 'neutral',
            'modifiers_detected': False,
            'exclamation_emphasis': False,
            'repeated_words': [],
            'emotional_amplification': False
        }

    def _get_fallback_urgency(self) -> Dict:
        """Fallback urgency analysis when detection unavailable"""
        return {
            'urgency_score': 0.4,
            'urgency_level': 'LOW',
            'text_urgency': 0.0,
            'prosody_urgency': 0.4,
            'prosody_type': 'normal',
            'urgent_phrases_detected': 0,
            'requires_immediate_attention': False
        }

    def _get_error_response(self, error_type: str, details: str = "") -> Dict:
        """Generate error response with fallback recommendations"""
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'enhanced_audio_comprehensive',
            'processing_successful': False,
            'error_type': error_type,
            'error_details': details,
            'depression_score': 0.5,
            'depression_level': 'unknown',
            'confidence_score': 0.5,
            'confidence_level': 'unknown',
            'overall_wellbeing': 'unknown',
            'risk_level': 'UNKNOWN',
            'requires_immediate_intervention': False,
            'requires_professional_followup': True,
            'safety_recommendations': {
                'primary_actions': [
                    {
                        'priority': 'HIGH',
                        'action': 'Technical Issue - Please Try Again',
                        'description': 'The audio analysis encountered an error. Please try recording again with clear audio and minimal background noise.',
                        'urgent': False
                    },
                    {
                        'priority': 'MODERATE',
                        'action': 'If You Need Help',
                        'description': 'If you are experiencing distress, please contact a mental health professional or crisis helpline regardless of technical issues.',
                        'urgent': False
                    }
                ],
                'professional_resources': [
                    {
                        'type': 'Support',
                        'name': 'Mental Health Helpline',
                        'contact': 'iCall: 9152987821 or NIMHANS: 080-46110007',
                        'urgency': 'available'
                    }
                ]
            },
            'analysis_confidence': 0.0
        }


# Global service instance
_enhanced_service = None

def get_enhanced_audio_service() -> EnhancedAudioAnalysisService:
    """Get or create the global enhanced audio service instance"""
    global _enhanced_service
    if _enhanced_service is None:
        _enhanced_service = EnhancedAudioAnalysisService()
    return _enhanced_service


# Convenience function
def analyze_audio_with_crisis_detection(audio_data: bytes) -> Dict:
    """
    Analyze audio with comprehensive crisis detection

    This is the main entry point for enhanced audio analysis
    """
    return get_enhanced_audio_service().analyze_audio_comprehensive(audio_data)
