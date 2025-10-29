"""
Text Sentiment Analysis for MindTrackAI
Analyzes transcribed speech for mental health indicators
"""

import re
from typing import Dict, List, Tuple


class TextSentimentAnalyzer:
    """
    Analyze transcribed text for mental health indicators
    """

    def __init__(self):
        """Initialize keyword dictionaries"""

        # DEPRESSION KEYWORDS - weighted by severity
        self.depression_keywords = {
            'severe': [
                'suicide', 'suicidal', 'kill myself', 'end it all', 'no point',
                'worthless', 'hopeless', 'can\'t go on', 'give up', 'want to die'
            ],
            'moderate': [
                'depressed', 'miserable', 'terrible', 'awful', 'horrible',
                'empty', 'numb', 'dark', 'hopeless', 'alone', 'isolated',
                'nobody cares', 'nothing matters', 'can\'t take it'
            ],
            'mild': [
                'sad', 'down', 'low', 'blue', 'unhappy', 'struggling',
                'hard', 'difficult', 'tired', 'exhausted', 'drained'
            ]
        }

        # ANXIETY KEYWORDS
        self.anxiety_keywords = {
            'severe': [
                'panic', 'terrified', 'can\'t breathe', 'heart racing',
                'dying', 'losing control', 'going crazy'
            ],
            'moderate': [
                'anxious', 'worried', 'nervous', 'scared', 'afraid',
                'stress', 'stressed', 'overwhelmed', 'on edge'
            ],
            'mild': [
                'concerned', 'uneasy', 'tense', 'restless', 'worried'
            ]
        }

        # POSITIVE INDICATORS
        self.positive_keywords = [
            'happy', 'great', 'wonderful', 'amazing', 'excellent',
            'good', 'better', 'improving', 'hopeful', 'excited',
            'grateful', 'blessed', 'love', 'enjoy', 'fun'
        ]

        # RECOVERY INDICATORS
        self.recovery_keywords = [
            'getting better', 'improving', 'feeling better', 'progress',
            'healing', 'recovering', 'hope', 'looking forward', 'positive'
        ]

        # SITUATIONAL STRESS INDICATORS
        self.situational_keywords = [
            'work', 'job', 'deadline', 'exam', 'test', 'interview',
            'temporary', 'this week', 'right now', 'today', 'currently'
        ]

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for mental health indicators

        Args:
            text: Transcribed speech

        Returns:
            Dictionary with sentiment scores and detected keywords
        """
        if not text:
            return {
                'depression_sentiment': 0.0,
                'anxiety_sentiment': 0.0,
                'positive_sentiment': 0.0,
                'crisis_detected': False,
                'situational_stress': False,
                'recovery_signals': False,
                'detected_keywords': []
            }

        text_lower = text.lower()

        # Detect keywords
        depression_score = self._calculate_depression_sentiment(text_lower)
        anxiety_score = self._calculate_anxiety_sentiment(text_lower)
        positive_score = self._calculate_positive_sentiment(text_lower)

        # Crisis detection
        crisis_detected = self._detect_crisis(text_lower)

        # Situational stress
        situational_stress = self._detect_situational_stress(text_lower)

        # Recovery signals
        recovery_signals = self._detect_recovery(text_lower)

        # Incongruence detection (positive words but overall negative tone)
        incongruence_detected = self._detect_incongruence(
            text_lower, positive_score, depression_score
        )

        return {
            'depression_sentiment': depression_score,
            'anxiety_sentiment': anxiety_score,
            'positive_sentiment': positive_score,
            'crisis_detected': crisis_detected,
            'situational_stress': situational_stress,
            'recovery_signals': recovery_signals,
            'incongruence_detected': incongruence_detected,
            'word_count': len(text_lower.split())
        }

    def _calculate_depression_sentiment(self, text: str) -> float:
        """Calculate depression sentiment from keywords"""
        score = 0.0
        word_count = len(text.split())

        # Severe keywords (higher weight, counts once)
        severe_count = sum(1 for keyword in self.depression_keywords['severe'] if keyword in text)
        if severe_count > 0:
            score += min(severe_count * 0.25, 0.50)  # Cap severe contribution

        # Moderate keywords
        moderate_count = sum(1 for keyword in self.depression_keywords['moderate'] if keyword in text)
        if moderate_count > 0:
            score += min(moderate_count * 0.12, 0.35)  # Cap moderate contribution

        # Mild keywords (lower weight)
        mild_count = sum(1 for keyword in self.depression_keywords['mild'] if keyword in text)
        if mild_count > 0:
            score += min(mild_count * 0.06, 0.20)  # Cap mild contribution

        # Normalize by text length (longer text = more keywords expected)
        if word_count > 20:
            score *= 0.85  # Reduce score for very long text

        # Cap at 1.0
        return min(score, 1.0)

    def _calculate_anxiety_sentiment(self, text: str) -> float:
        """Calculate anxiety sentiment from keywords"""
        score = 0.0

        # Severe keywords (0.35 each)
        for keyword in self.anxiety_keywords['severe']:
            if keyword in text:
                score += 0.35

        # Moderate keywords (0.18 each)
        for keyword in self.anxiety_keywords['moderate']:
            if keyword in text:
                score += 0.18

        # Mild keywords (0.10 each)
        for keyword in self.anxiety_keywords['mild']:
            if keyword in text:
                score += 0.10

        return min(score, 1.0)

    def _calculate_positive_sentiment(self, text: str) -> float:
        """Calculate positive sentiment"""
        score = 0.0

        for keyword in self.positive_keywords:
            if keyword in text:
                score += 0.10

        return min(score, 1.0)

    def _detect_crisis(self, text: str) -> bool:
        """Detect crisis keywords"""
        crisis_phrases = [
            'suicide', 'kill myself', 'end it all', 'want to die',
            'can\'t go on', 'no point living', 'better off dead'
        ]

        for phrase in crisis_phrases:
            if phrase in text:
                return True

        return False

    def _detect_situational_stress(self, text: str) -> bool:
        """Detect if stress is situational/temporary"""
        count = 0

        for keyword in self.situational_keywords:
            if keyword in text:
                count += 1

        # If 2+ situational keywords, likely temporary stress
        return count >= 2

    def _detect_recovery(self, text: str) -> bool:
        """Detect recovery/improvement signals"""
        for keyword in self.recovery_keywords:
            if keyword in text:
                return True

        return False

    def _detect_incongruence(self, text: str, positive_score: float,
                            depression_score: float) -> bool:
        """
        Detect incongruence between words and likely emotional state

        Example: "I'm fine! Everything's great!" (forced positivity)
        """
        # Check for forced positivity patterns
        forced_positivity_patterns = [
            r"i'm fine[!.]*",
            r"everything's? (great|fine|okay|perfect)[!.]*",
            r"never been better[!.]*",
            r"totally fine[!.]*",
            r"i'm (great|good|okay)[!.]+"  # Multiple exclamation marks
        ]

        forced_positivity = False
        for pattern in forced_positivity_patterns:
            if re.search(pattern, text):
                forced_positivity = True
                break

        # Incongruence: forced positivity + some depression indicators
        if forced_positivity and depression_score > 0.05:
            return True

        # Incongruence: high positive words but also high depression words
        if positive_score > 0.15 and depression_score > 0.20:
            return True

        return False

    def calculate_text_adjustment(self, text_analysis: Dict,
                                 emotion_depression: float,
                                 emotions: Dict[str, float] = None) -> float:
        """
        Calculate adjustment to depression score based on text analysis

        Args:
            text_analysis: Result from analyze_text()
            emotion_depression: Depression score from emotions
            emotions: Optional emotion scores for correlation analysis

        Returns:
            Adjustment value (-0.15 to +0.15)
        """
        adjustment = 0.0

        # EMOTION-TEXT CORRELATION (NEW!)
        # When text and emotions align, increase confidence in the signal
        # When they conflict, flag incongruence
        correlation_bonus = 0.0
        if emotions:
            sad_emotion = emotions.get('sad', 0)
            happy_emotion = emotions.get('happy', 0)

            # If both text and emotions indicate depression, strengthen signal
            if text_analysis['depression_sentiment'] > 0.2 and sad_emotion > 0.4:
                correlation_bonus = 0.03  # They agree - strengthen

            # If text is positive but emotions are sad (or vice versa), already handled by incongruence
            # But if both are positive, reduce depression score
            if text_analysis['positive_sentiment'] > 0.2 and happy_emotion > 0.5:
                correlation_bonus = -0.04  # Both positive - reduce depression

        # Depression sentiment adds to score (CALIBRATED)
        if text_analysis['depression_sentiment'] > 0:
            # Only add if it's significant
            if text_analysis['depression_sentiment'] > 0.15:
                adjustment += text_analysis['depression_sentiment'] * 0.08

        # Anxiety sentiment adds slightly (comorbidity)
        if text_analysis['anxiety_sentiment'] > 0.3:
            adjustment += text_analysis['anxiety_sentiment'] * 0.02

        # Positive sentiment reduces score (CALIBRATED)
        if text_analysis['positive_sentiment'] > 0.2:
            adjustment -= text_analysis['positive_sentiment'] * 0.05

        # Crisis detection adds significantly (BOOSTED for better crisis detection)
        if text_analysis['crisis_detected']:
            adjustment += 0.20  # Increased from 0.15 - crisis is critical!

        # Recovery signals reduce score (BOOSTED for better recovery detection)
        if text_analysis['recovery_signals']:
            adjustment -= 0.08  # Increased from 0.06

        # Situational stress reduces score (temporary, not clinical)
        if text_analysis['situational_stress']:
            adjustment -= 0.05

        # Incongruence adds (masking depression) - important signal
        if text_analysis['incongruence_detected']:
            adjustment += 0.10

        # Add correlation bonus
        adjustment += correlation_bonus

        # Cap adjustment to narrower range (more conservative)
        adjustment = max(-0.15, min(0.15, adjustment))

        return adjustment
