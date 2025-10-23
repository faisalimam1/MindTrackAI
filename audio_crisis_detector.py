#!/usr/bin/env python3
"""
Enhanced Audio Crisis Detection and Intent Analysis Service
Detects serious verbal cues including distress, hopelessness, and suicidal intent
"""

import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrisisIndicator:
    """Represents a detected crisis indicator"""
    keyword: str
    category: str
    severity: float  # 0.0-1.0
    context: str
    position: int


class AudioCrisisDetector:
    """
    Advanced crisis detection system for audio transcripts
    Detects suicidal ideation, self-harm intent, hopelessness, and severe distress
    """

    # ===== CRISIS KEYWORD DEFINITIONS =====

    # CRITICAL: Explicit suicidal ideation (Severity: 1.0)
    EXPLICIT_SUICIDAL_KEYWORDS = [
        'kill myself', 'end my life', 'take my life', 'want to die',
        'going to die', 'gonna die', 'wish i was dead', 'wish i were dead',
        'better off dead', 'world without me', 'no reason to live',
        'ready to die', 'time to die', 'commit suicide', 'end it all',
        'can\'t go on', 'cannot go on', 'don\'t want to be alive',
        'don\'t want to live', 'tired of living', 'done with life',
        'life is not worth', 'not worth living', 'nothing to live for'
    ]

    # HIGH: Self-harm intent (Severity: 0.95)
    SELF_HARM_KEYWORDS = [
        'hurt myself', 'harm myself', 'cut myself', 'cutting myself',
        'overdose', 'take pills', 'jump off', 'hang myself',
        'shoot myself', 'drown myself', 'poison myself',
        'end the pain', 'make it stop', 'stop the suffering'
    ]

    # HIGH: Hopelessness indicators (Severity: 0.85)
    HOPELESSNESS_KEYWORDS = [
        'no hope', 'hopeless', 'no point', 'pointless', 'meaningless',
        'nothing matters', 'no future', 'can\'t see tomorrow',
        'never get better', 'won\'t improve', 'always be this way',
        'give up', 'given up', 'giving up on', 'lost all hope',
        'no way out', 'trapped forever', 'can\'t escape', 'no escape'
    ]

    # MODERATE: Severe depression indicators (Severity: 0.75)
    SEVERE_DEPRESSION_KEYWORDS = [
        'worthless', 'i\'m worthless', 'feel worthless',
        'useless', 'i\'m useless', 'feel useless',
        'failure', 'total failure', 'complete failure',
        'hate myself', 'despise myself', 'loathe myself',
        'burden', 'i\'m a burden', 'burden to everyone',
        'empty inside', 'feel empty', 'numb', 'feel nothing',
        'can\'t feel', 'emotionally dead', 'walking corpse'
    ]

    # MODERATE: Isolation and withdrawal (Severity: 0.65)
    ISOLATION_KEYWORDS = [
        'nobody cares', 'no one cares', 'alone forever', 'completely alone',
        'everyone hates me', 'better off without me', 'disappear',
        'want to disappear', 'invisible', 'don\'t exist',
        'nobody would notice', 'wouldn\'t be missed'
    ]

    # MODERATE: Pain and suffering (Severity: 0.70)
    PAIN_KEYWORDS = [
        'can\'t take it anymore', 'unbearable pain', 'too much pain',
        'suffering too much', 'can\'t stand it', 'can\'t bear it',
        'hurts too much', 'pain won\'t stop', 'constant pain',
        'torture', 'torment', 'agony'
    ]

    # LOW-MODERATE: General distress (Severity: 0.50)
    DISTRESS_KEYWORDS = [
        'can\'t cope', 'falling apart', 'breaking down', 'losing it',
        'can\'t handle', 'too overwhelmed', 'drowning', 'suffocating',
        'exhausted', 'drained', 'defeated'
    ]

    # ===== PROTECTIVE FACTORS (reduce severity) =====
    PROTECTIVE_PHRASES = [
        'getting help', 'seeing therapist', 'taking medication',
        'support system', 'people care', 'reasons to live',
        'getting better', 'feeling better', 'improvement',
        'hope for', 'looking forward', 'future plans',
        'won\'t act on it', 'not going to', 'just thoughts'
    ]

    # ===== URGENCY INDICATORS =====
    URGENCY_INDICATORS = [
        'right now', 'today', 'tonight', 'this moment',
        'going to', 'about to', 'planning to', 'decided to',
        'ready to', 'can\'t wait', 'soon', 'very soon'
    ]

    # ===== PAST TENSE INDICATORS (lower immediate risk) =====
    PAST_TENSE_INDICATORS = [
        'used to', 'in the past', 'before', 'previously',
        'long ago', 'years ago', 'months ago', 'back then'
    ]

    def __init__(self):
        """Initialize the crisis detector"""
        self.crisis_categories = {
            'explicit_suicidal': (self.EXPLICIT_SUICIDAL_KEYWORDS, 1.0),
            'self_harm': (self.SELF_HARM_KEYWORDS, 0.95),
            'hopelessness': (self.HOPELESSNESS_KEYWORDS, 0.85),
            'severe_depression': (self.SEVERE_DEPRESSION_KEYWORDS, 0.75),
            'pain_suffering': (self.PAIN_KEYWORDS, 0.70),
            'isolation': (self.ISOLATION_KEYWORDS, 0.65),
            'distress': (self.DISTRESS_KEYWORDS, 0.50)
        }

    def detect_crisis(self, text: str) -> Tuple[bool, float, Dict]:
        """
        Comprehensive crisis detection with detailed analysis

        Args:
            text: Transcribed speech text

        Returns:
            Tuple of (is_crisis, confidence, details_dict)
        """
        if not text or len(text.strip()) < 3:
            return False, 0.0, {'reason': 'Empty or invalid text'}

        text_lower = text.lower().strip()

        # Detect all indicators
        indicators = self._detect_all_indicators(text_lower)

        if not indicators:
            return False, 0.0, {'reason': 'No crisis indicators detected', 'text_analyzed': True}

        # Calculate base severity score
        base_severity = self._calculate_base_severity(indicators)

        # Apply contextual modifiers
        urgency_multiplier = self._detect_urgency(text_lower)
        past_tense_reduction = self._detect_past_tense(text_lower)
        protective_reduction = self._detect_protective_factors(text_lower)

        # Calculate final crisis score
        final_score = base_severity * urgency_multiplier * (1.0 - past_tense_reduction) * (1.0 - protective_reduction)
        final_score = min(max(final_score, 0.0), 1.0)

        # Determine if this is a crisis
        is_crisis = final_score >= 0.65  # Threshold for crisis intervention

        # Generate detailed response
        details = {
            'crisis_detected': is_crisis,
            'severity_score': round(final_score, 3),
            'base_severity': round(base_severity, 3),
            'urgency_multiplier': round(urgency_multiplier, 2),
            'past_tense_reduction': round(past_tense_reduction, 2),
            'protective_reduction': round(protective_reduction, 2),
            'indicators': [
                {
                    'keyword': ind.keyword,
                    'category': ind.category,
                    'severity': ind.severity,
                    'context': ind.context
                }
                for ind in indicators
            ],
            'risk_level': self._categorize_risk(final_score),
            'requires_immediate_intervention': final_score >= 0.85,
            'requires_professional_followup': final_score >= 0.65,
            'text_sample': text[:200] + '...' if len(text) > 200 else text
        }

        logger.warning(f"Crisis detection: is_crisis={is_crisis}, score={final_score:.3f}, "
                      f"indicators={len(indicators)}, risk={details['risk_level']}")

        return is_crisis, final_score, details

    def _detect_all_indicators(self, text: str) -> List[CrisisIndicator]:
        """Detect all crisis indicators in text"""
        indicators = []

        for category, (keywords, base_severity) in self.crisis_categories.items():
            for keyword in keywords:
                # Find all occurrences of the keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()

                    indicators.append(CrisisIndicator(
                        keyword=keyword,
                        category=category,
                        severity=base_severity,
                        context=context,
                        position=match.start()
                    ))

        return indicators

    def _calculate_base_severity(self, indicators: List[CrisisIndicator]) -> float:
        """
        Calculate base severity from detected indicators

        Uses a weighted combination approach:
        - Highest severity indicator (50% weight)
        - Average of all indicators (30% weight)
        - Count of unique categories (20% weight)
        """
        if not indicators:
            return 0.0

        # Get highest severity
        max_severity = max(ind.severity for ind in indicators)

        # Calculate average severity
        avg_severity = sum(ind.severity for ind in indicators) / len(indicators)

        # Count unique categories (more categories = higher concern)
        unique_categories = len(set(ind.category for ind in indicators))
        category_factor = min(unique_categories / 4.0, 1.0)  # Max 4 categories for full weight

        # Weighted combination
        base_score = (
            max_severity * 0.50 +
            avg_severity * 0.30 +
            category_factor * 0.20
        )

        # Multiple indicators of same category increase severity
        if len(indicators) > 1:
            repetition_boost = min((len(indicators) - 1) * 0.05, 0.15)
            base_score = min(base_score + repetition_boost, 1.0)

        return base_score

    def _detect_urgency(self, text: str) -> float:
        """
        Detect urgency indicators that increase immediate risk

        Returns:
            Multiplier between 1.0 (no urgency) and 1.5 (high urgency)
        """
        urgency_count = sum(1 for indicator in self.URGENCY_INDICATORS if indicator in text)

        if urgency_count == 0:
            return 1.0
        elif urgency_count == 1:
            return 1.2
        elif urgency_count == 2:
            return 1.35
        else:
            return 1.5

    def _detect_past_tense(self, text: str) -> float:
        """
        Detect past tense indicators that reduce immediate risk

        Returns:
            Reduction factor between 0.0 (no reduction) and 0.3 (significant reduction)
        """
        past_tense_count = sum(1 for indicator in self.PAST_TENSE_INDICATORS if indicator in text)

        if past_tense_count == 0:
            return 0.0
        elif past_tense_count == 1:
            return 0.15
        else:
            return 0.30

    def _detect_protective_factors(self, text: str) -> float:
        """
        Detect protective factors that reduce crisis severity

        Returns:
            Reduction factor between 0.0 (no protection) and 0.25 (strong protection)
        """
        protective_count = sum(1 for phrase in self.PROTECTIVE_PHRASES if phrase in text)

        if protective_count == 0:
            return 0.0
        elif protective_count == 1:
            return 0.10
        elif protective_count == 2:
            return 0.18
        else:
            return 0.25

    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level from severity score"""
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

    def analyze_emotion_intensity(self, text: str, sentiment_polarity: float) -> Dict:
        """
        Analyze emotional intensity from text and sentiment

        Args:
            text: Transcribed speech text
            sentiment_polarity: Sentiment polarity score (-1 to 1)

        Returns:
            Dict with emotion intensity analysis
        """
        text_lower = text.lower()

        # Detect intensity modifiers
        intensity_modifiers = {
            'extreme': ['extremely', 'incredibly', 'completely', 'totally', 'absolutely'],
            'high': ['very', 'really', 'so', 'too', 'quite'],
            'moderate': ['somewhat', 'fairly', 'pretty', 'rather'],
            'low': ['a little', 'slightly', 'a bit', 'kind of', 'sort of']
        }

        intensity_level = 'neutral'
        intensity_score = 0.5

        for level, modifiers in intensity_modifiers.items():
            if any(mod in text_lower for mod in modifiers):
                intensity_level = level
                intensity_score = {
                    'extreme': 0.95,
                    'high': 0.80,
                    'moderate': 0.60,
                    'low': 0.35,
                    'neutral': 0.50
                }.get(level, 0.50)
                break

        # Check for emotional amplification (exclamation marks, repeated words)
        exclamation_count = text.count('!')
        if exclamation_count >= 3:
            intensity_score = min(intensity_score * 1.3, 1.0)
        elif exclamation_count >= 1:
            intensity_score = min(intensity_score * 1.15, 1.0)

        # Check for repeated words (emotional emphasis)
        words = text_lower.split()
        repeated_words = []
        for i in range(len(words) - 1):
            if words[i] == words[i+1] and len(words[i]) > 3:
                repeated_words.append(words[i])
                intensity_score = min(intensity_score * 1.2, 1.0)

        # Adjust based on sentiment polarity
        if abs(sentiment_polarity) > 0.5:
            intensity_score = min(intensity_score * 1.2, 1.0)

        return {
            'intensity_score': round(intensity_score, 3),
            'intensity_level': intensity_level,
            'modifiers_detected': intensity_level != 'neutral',
            'exclamation_emphasis': exclamation_count > 0,
            'repeated_words': repeated_words,
            'emotional_amplification': len(repeated_words) > 0 or exclamation_count > 0
        }

    def detect_speech_urgency(self, text: str, speaking_rate: float, pause_frequency: float) -> Dict:
        """
        Detect urgency in speech combining text and prosodic features

        Args:
            text: Transcribed speech text
            speaking_rate: Speaking rate (0-1, higher = faster)
            pause_frequency: Pause frequency (0-1, higher = more pauses)

        Returns:
            Dict with urgency analysis
        """
        text_lower = text.lower()

        # Text-based urgency indicators
        urgent_phrases = [
            'right now', 'immediately', 'urgent', 'emergency', 'asap',
            'hurry', 'quick', 'fast', 'now', 'today', 'tonight',
            'can\'t wait', 'need to', 'have to', 'must'
        ]

        urgent_phrase_count = sum(1 for phrase in urgent_phrases if phrase in text_lower)
        text_urgency_score = min(urgent_phrase_count / 3.0, 1.0)

        # Prosodic urgency indicators
        # High urgency: fast speech, few pauses OR very slow with many pauses (distress)
        if speaking_rate > 0.7 and pause_frequency < 0.3:
            # Fast, fluent speech = high urgency
            prosody_urgency_score = 0.8
            prosody_type = 'high_urgency_fluent'
        elif speaking_rate < 0.3 and pause_frequency > 0.7:
            # Slow, halting speech = distressed urgency
            prosody_urgency_score = 0.75
            prosody_type = 'distressed_urgency'
        elif speaking_rate > 0.6:
            # Moderately fast = moderate urgency
            prosody_urgency_score = 0.6
            prosody_type = 'moderate_urgency'
        else:
            # Normal speech rate
            prosody_urgency_score = 0.4
            prosody_type = 'low_urgency'

        # Combined urgency score
        combined_urgency = (text_urgency_score * 0.6 + prosody_urgency_score * 0.4)

        # Categorize urgency level
        if combined_urgency >= 0.75:
            urgency_level = 'HIGH'
        elif combined_urgency >= 0.50:
            urgency_level = 'MODERATE'
        else:
            urgency_level = 'LOW'

        return {
            'urgency_score': round(combined_urgency, 3),
            'urgency_level': urgency_level,
            'text_urgency': round(text_urgency_score, 3),
            'prosody_urgency': round(prosody_urgency_score, 3),
            'prosody_type': prosody_type,
            'urgent_phrases_detected': urgent_phrase_count,
            'requires_immediate_attention': combined_urgency >= 0.75
        }


# Global detector instance
_crisis_detector = None

def get_crisis_detector() -> AudioCrisisDetector:
    """Get or create the global crisis detector instance"""
    global _crisis_detector
    if _crisis_detector is None:
        _crisis_detector = AudioCrisisDetector()
    return _crisis_detector


# Convenience functions
def detect_crisis_from_text(text: str) -> Tuple[bool, float, Dict]:
    """Detect crisis indicators in text"""
    return get_crisis_detector().detect_crisis(text)


def analyze_emotion_intensity(text: str, sentiment_polarity: float = 0.0) -> Dict:
    """Analyze emotion intensity in text"""
    return get_crisis_detector().analyze_emotion_intensity(text, sentiment_polarity)


def detect_speech_urgency(text: str, speaking_rate: float = 0.5, pause_frequency: float = 0.5) -> Dict:
    """Detect urgency in speech"""
    return get_crisis_detector().detect_speech_urgency(text, speaking_rate, pause_frequency)
