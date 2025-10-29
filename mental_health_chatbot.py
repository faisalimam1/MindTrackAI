"""
Advanced Mental Health Chatbot Engine
Uses pattern matching, context awareness, and severity detection
"""

import re
from datetime import datetime, timedelta

class MentalHealthChatbot:
    def __init__(self):
        # Define comprehensive patterns for mental health issues
        self.patterns = {
            'depression': {
                'severe': [
                    r'want to (die|kill myself|end it)',
                    r'(no|dont|don\'t) want to (live|be here)',
                    r'(life|living) is(n\'t| not) worth',
                    r'everyone.*(better off without|better without)',
                    r'(cant|can\'t|cannot) go on',
                ],
                'moderate': [
                    r'feel(ing)? (so|very|extremely|really)? (depressed|hopeless|worthless|empty)',
                    r'(no|zero|lost).*(motivation|interest|energy)',
                    r'(nothing|everything) (matters|pointless)',
                    r'hate myself',
                    r'(cant|can\'t|cannot) (feel|enjoy) anything',
                    r'crying (all|every|constantly)',
                ],
                'mild': [
                    r'feel(ing)? (sad|down|low|blue)',
                    r'not (happy|myself)',
                    r'feel(ing)? (numb|disconnected)',
                ]
            },
            'anxiety': {
                'severe': [
                    r'(panic attack|having a panic attack)',
                    r'(cant|can\'t|cannot) (breathe|breath)',
                    r'(heart|chest).*(pounding|racing|tight)',
                    r'feel(ing)? like (dying|death|losing control)',
                    r'constant(ly)? (panic|anxious|worried|fear)',
                ],
                'moderate': [
                    r'(very|so|extremely|really) (anxious|worried|stressed|nervous)',
                    r'racing thoughts',
                    r'(cant|can\'t|cannot) (stop|control) (worry|thinking)',
                    r'on edge',
                    r'(overthinking|overanalyzing)',
                ],
                'mild': [
                    r'feel(ing)? (anxious|worried|nervous|stressed)',
                    r'(a bit|little|somewhat) (anxious|worried)',
                ]
            },
            'sleep': {
                'severe': [
                    r'(haven\'t|havent|not) (slept|sleeping) (in|for) (days|week)',
                    r'sleep.*(hours|hour) (a|per) (night|day)',
                    r'(no|zero) sleep',
                ],
                'moderate': [
                    r'(cant|can\'t|cannot) (sleep|fall asleep)',
                    r'(insomnia|sleep problems)',
                    r'wake up.*(constantly|frequently|all night)',
                    r'(nightmares|bad dreams)',
                ],
                'mild': [
                    r'(trouble|difficulty) sleeping',
                    r'(tired|exhausted|fatigued)',
                ]
            },
            'loneliness': {
                'severe': [
                    r'(no one|nobody).*(cares|loves|understands)',
                    r'(completely|totally|absolutely) (alone|isolated)',
                    r'(no|zero|dont have|don\'t have) (friends|anyone)',
                ],
                'moderate': [
                    r'feel(ing)? (so|very|really)? (alone|lonely|isolated)',
                    r'(no one|nobody) to talk to',
                ],
                'mild': [
                    r'feel(ing)? (a bit|somewhat) lonely',
                    r'wish (i|I) had (more|someone)',
                ]
            },
            'relationship': [
                r'(broke up|break up|divorce)',
                r'(fight|fighting|argue|arguing) with (partner|spouse|boyfriend|girlfriend)',
                r'(toxic|abusive) relationship',
                r'(trust issues|betrayed|cheated)',
                r'family (problems|issues|conflict)',
            ],
            'work_stress': [
                r'(work|job).*(stress|pressure|overwhelm)',
                r'(burn|burnt|burned) out',
                r'(hate|cant stand|can\'t stand) my (work|job)',
                r'(lost|losing) my job',
            ]
        }

    def analyze_message(self, message, conversation_history=None):
        """
        Analyze user message and return category, severity, and appropriate response
        """
        message_lower = message.lower()

        # Detect emotional intensity
        intensity_markers = ['very', 'extremely', 'really', 'so', 'always', 'never',
                            'completely', 'totally', 'constantly', 'all the time']
        has_intensity = any(marker in message_lower for marker in intensity_markers)

        # Check duration markers
        duration_markers = ['weeks', 'months', 'years', 'every day', 'everyday', 'daily']
        has_duration = any(marker in message_lower for marker in duration_markers)

        # Analyze for each category
        results = {
            'category': None,
            'severity': 'mild',
            'has_intensity': has_intensity,
            'has_duration': has_duration,
            'matches': []
        }

        # Check depression patterns
        for severity in ['severe', 'moderate', 'mild']:
            for pattern in self.patterns['depression'][severity]:
                if re.search(pattern, message_lower):
                    results['category'] = 'depression'
                    results['severity'] = severity
                    results['matches'].append(pattern)
                    return results

        # Check anxiety patterns
        for severity in ['severe', 'moderate', 'mild']:
            for pattern in self.patterns['anxiety'][severity]:
                if re.search(pattern, message_lower):
                    results['category'] = 'anxiety'
                    results['severity'] = severity
                    results['matches'].append(pattern)
                    return results

        # Check sleep patterns
        for severity in ['severe', 'moderate', 'mild']:
            for pattern in self.patterns['sleep'][severity]:
                if re.search(pattern, message_lower):
                    results['category'] = 'sleep'
                    results['severity'] = severity
                    results['matches'].append(pattern)
                    return results

        # Check loneliness patterns
        for severity in ['severe', 'moderate', 'mild']:
            for pattern in self.patterns['loneliness'][severity]:
                if re.search(pattern, message_lower):
                    results['category'] = 'loneliness'
                    results['severity'] = severity
                    results['matches'].append(pattern)
                    return results

        # Check other categories
        for pattern in self.patterns['relationship']:
            if re.search(pattern, message_lower):
                results['category'] = 'relationship'
                results['severity'] = 'moderate' if has_intensity else 'mild'
                return results

        for pattern in self.patterns['work_stress']:
            if re.search(pattern, message_lower):
                results['category'] = 'work_stress'
                results['severity'] = 'moderate' if has_intensity else 'mild'
                return results

        # Check for positive sentiment
        positive_patterns = [r'feel(ing)? (better|good|happy|hopeful)',
                            r'(doing|going) (well|better|good)',
                            r'(improved|improving)',
                            r'(thank|thanks|grateful)']

        for pattern in positive_patterns:
            if re.search(pattern, message_lower):
                results['category'] = 'positive'
                return results

        # Check for help-seeking
        help_patterns = [r'(need|want|looking for) (help|support|therapy)',
                        r'(therapist|counselor|psychologist)',
                        r'how (do|can) (i|I) get help']

        for pattern in help_patterns:
            if re.search(pattern, message_lower):
                results['category'] = 'help_seeking'
                return results

        return results

    def generate_response(self, analysis):
        """Generate contextual response based on analysis"""

        category = analysis['category']
        severity = analysis['severity']
        has_duration = analysis['has_duration']

        if category == 'depression':
            if severity == 'severe':
                return self._severe_depression_response()
            elif severity == 'moderate':
                return self._moderate_depression_response(has_duration)
            else:
                return self._mild_depression_response()

        elif category == 'anxiety':
            if severity == 'severe':
                return self._severe_anxiety_response()
            elif severity == 'moderate':
                return self._moderate_anxiety_response()
            else:
                return self._mild_anxiety_response()

        elif category == 'sleep':
            return self._sleep_response(severity)

        elif category == 'loneliness':
            return self._loneliness_response(severity)

        elif category == 'relationship':
            return self._relationship_response()

        elif category == 'work_stress':
            return self._work_stress_response()

        elif category == 'positive':
            return self._positive_response()

        elif category == 'help_seeking':
            return self._help_seeking_response()

        else:
            return self._default_response()

    # Response methods
    def _severe_depression_response(self):
        return """I'm deeply concerned about what you're experiencing. These feelings sound overwhelming and serious.

🚨 **Immediate Action Needed:**
• Please reach out to a mental health professional TODAY
• If you're having thoughts of harming yourself, call 080-46110007 (NIMHANS 24/7) or 9152987821 (iCall)
• Tell someone you trust what you're going through RIGHT NOW

**Why this matters:** When depression reaches this level, professional intervention is crucial. You don't have to handle this alone, and treatment can make a significant difference.

📍 Use our "Find Doctors" feature to locate mental health professionals near you.

Please tell me: Do you have someone you can call right now? Are you in a safe place?"""

    def _moderate_depression_response(self, has_duration):
        if has_duration:
            return """What you're describing sounds like clinical depression, especially since it's been ongoing. I want you to know this is treatable.

**Recommended Next Steps:**
1. **See a mental health professional within the next week** - This level of depression typically benefits from professional treatment (therapy and/or medication)
2. **Take our PHQ-9 assessment** - It helps measure depression severity objectively
3. **Reach out to your support system** - Let someone know what you're experiencing

**In the meantime:**
• Break tasks into very small steps
• Be compassionate with yourself - depression is an illness
• Maintain basic routines (sleep, eating) even if you don't feel like it
• Avoid major life decisions while depressed

How long has this been going on? Are you able to work/function in daily activities?"""
        else:
            return """I hear that you're going through a difficult time emotionally. These feelings deserve attention and care.

**Let's explore this:**
• What triggered these feelings, if you can identify?
• Is this a pattern you've experienced before?
• Are you still able to function (work, relationships, self-care)?

**Helpful actions:**
• Talk to someone you trust today
• Gentle physical activity (even a 10-minute walk helps)
• Avoid isolating - stay connected
• Journal your thoughts
• Challenge negative thoughts - are they facts or feelings?

If this persists beyond 2 weeks or gets worse, please see a mental health professional. Would you like help finding resources?"""

    def _mild_depression_response(self):
        return """It sounds like you're feeling down. That's a valid feeling, and it's good that you're recognizing and expressing it.

**Some questions to explore:**
• What's been happening in your life recently?
• Have you noticed any patterns (time of day, situations, people)?
• How is your sleep, exercise, and social connection?

**Quick mood boosters:**
• Connect with a friend or loved one
• Get outside in natural light
• Do something small you enjoy
• Move your body (walk, stretch, dance)
• Help someone else (volunteering boosts mood)

Many people feel down sometimes. If it lasts more than a few days or starts affecting your life, consider reaching out for support.

What do you think might help you feel better right now?"""

    def _severe_anxiety_response(self):
        return """It sounds like you're experiencing severe anxiety or a panic attack. Let's focus on getting you through this moment.

**IMMEDIATE RELIEF - Do this NOW:**

**5-4-3-2-1 Grounding:**
• Name 5 things you SEE
• Name 4 things you can TOUCH
• Name 3 things you can HEAR
• Name 2 things you can SMELL
• Name 1 thing you can TASTE

**Box Breathing:**
• Breathe IN for 4 counts
• HOLD for 4 counts
• Breathe OUT for 4 counts
• HOLD for 4 counts
• Repeat 5 times

**Remember:** Panic attacks can't harm you. They peak around 10 minutes and will pass.

If panic attacks are frequent, please see a mental health professional - they're highly treatable. Have you experienced this before? Are you somewhere safe right now?"""

    def _moderate_anxiety_response(self):
        return """Anxiety can feel overwhelming. Let's work on managing it together.

**Understanding your anxiety:**
• What situation or thought is triggering it?
• Is this a specific worry or general unease?
• Are you experiencing physical symptoms (racing heart, tension, etc.)?

**Coping strategies:**
• **Ground yourself** - Focus on your breath and present moment
• **Challenge anxious thoughts** - What's the evidence? What's more likely?
• **Move** - Exercise reduces anxiety significantly
• **Limit caffeine/alcohol** - They can worsen anxiety
• **Set worry time** - Give yourself 15 minutes to worry, then move on

**Long-term help:**
If anxiety is frequent or interfering with your life, consider:
• Cognitive Behavioral Therapy (CBT) - very effective for anxiety
• Mindfulness meditation
• Regular exercise routine
• Professional evaluation

What usually helps you feel calmer? How often are you experiencing this?"""

    def _mild_anxiety_response(self):
        return """It's normal to feel anxious sometimes, especially during stressful situations.

**Quick anxiety relief:**
• Take 5 deep, slow breaths
• Name what's making you anxious
• Ask yourself: "Is this thought helpful?" "What would I tell a friend?"
• Do something that requires focus (puzzle, organizing, cooking)
• Talk it out with someone

**Is this anxiety:**
• About something specific? (That's normal situational anxiety)
• Vague and free-floating? (Might need more attention)
• Affecting your sleep or daily activities? (Consider professional help)

Sometimes anxiety is our mind's way of preparing us. What's worrying you right now?"""

    def _sleep_response(self, severity):
        if severity == 'severe':
            return """Serious sleep deprivation can significantly impact your mental and physical health. This needs professional attention.

**Immediate actions:**
• See a doctor this week - Sleep disorders need medical evaluation
• Avoid caffeine entirely for now
• Keep bedroom dark and cool
• No screens 2 hours before bed

Sleep deprivation can cause or worsen depression, anxiety, and other mental health issues. This is a priority to address.

How many hours are you sleeping per night? How long has this been going on?"""
        else:
            return """Sleep issues can really affect how we feel mentally. Let's improve your sleep hygiene:

**Sleep optimization:**
• Same sleep/wake time every day (yes, weekends too)
• Bedroom: cool (65-68°F), dark, quiet
• No screens 1 hour before bed
• Avoid caffeine after 2 PM
• Light exercise daily (but not near bedtime)
• Relaxation routine before bed

**If your mind is racing:**
• Keep a journal by your bed - write worries down
• Try progressive muscle relaxation
• Listen to sleep meditation or white noise

If sleep doesn't improve in 2-3 weeks, see a doctor. How long have you been having sleep issues?"""

    def _loneliness_response(self, severity):
        if severity == 'severe':
            return """Feeling deeply alone can be incredibly painful, and I want you to know that you deserve connection and support.

**Breaking isolation:**
• Reach out to ONE person today - text, call, or message someone
• If you have no one: Call a helpline to talk (080-46110007)
• Join online communities around your interests (Reddit, Discord, Facebook groups)
• Consider therapy - therapist can help with both loneliness and connection skills

**Building connections:**
• Volunteer - Shared purpose creates bonds
• Take a class or join a meetup group
• Regular places - Coffee shop, library (familiar faces become friends)
• Reconnect with someone from your past

Quality matters more than quantity. Even one good connection can transform loneliness.

What's preventing you from connecting with others? Let's problem-solve this together."""
        else:
            return """Loneliness is a common human experience, but that doesn't make it easier. Let's work on building meaningful connections.

**Connection strategies:**
• Reach out first - Most people appreciate contact
• Show up consistently - Same class, same group, same place
• Be vulnerable - Share something real about yourself
• Listen actively - People connect when feeling heard
• Join communities aligned with your values/interests

**Meanwhile:**
• Self-connection matters - Journal, create, reflect
• Pets provide companionship
• Helping others combats loneliness
• Online friendships are real friendships

Who in your life could you reach out to today? What interests could connect you with others?"""

    def _relationship_response(self):
        return """Relationship struggles can be deeply painful and affect our entire wellbeing.

**First, take care of yourself:**
• Your feelings are valid, whatever they are
• Seek support from friends, family, or therapist
• Maintain self-care routines
• Avoid major decisions when highly emotional

**If the relationship is abusive:**
• Your safety comes first
• Document concerning behaviors
• Have a safety plan
• Reach out to domestic violence resources

**For relationship conflicts:**
• Communication is key - "I feel X when Y happens"
• Listen to understand, not to respond
• Consider couples therapy if both willing
• Some relationships are worth fighting for; some aren't

What's happening in your relationship? How are you coping with this?"""

    def _work_stress_response(self):
        return """Work stress is one of the most common mental health stressors. Let's address this.

**Immediate stress management:**
• Set boundaries - Work hours are work hours
• Take real breaks - Step away from desk
• Prioritize tasks - Not everything is urgent
• Talk to someone who gets it
• Physical activity after work to decompress

**Longer-term:**
• Is this temporary (project deadline) or ongoing?
• Can you discuss workload with manager?
• Is the job aligned with your values?
• What would need to change for this to be sustainable?

**If burnout:**
• Time off may be needed
• Professional help can provide perspective
• Sometimes a job change is the healthiest choice

What specifically about work is stressing you most? Are you able to disconnect outside work hours?"""

    def _positive_response(self):
        return """That's wonderful! It's so important to acknowledge and celebrate when things are going better.

**Maintaining positive momentum:**
• What's contributing to feeling better? (Keep doing that!)
• Notice and appreciate small positive moments
• Stay connected with supportive people
• Continue healthy habits (sleep, exercise, connection)
• Remember this feeling during tough times

**Building resilience:**
• Keep a gratitude journal
• Maintain mental health practices even when feeling good
• Have a plan for when you feel down again (what helps?)

I'm genuinely glad you're feeling better. What's been the most helpful in getting here?"""

    def _help_seeking_response(self):
        return """Seeking help is a sign of strength and self-awareness. I'm glad you're considering professional support.

**Finding the right help:**

🏥 **Use our "Find Doctors" feature** - Locate mental health professionals near you

**Types of professionals:**
• Psychiatrist - Medical doctor, can prescribe medication
• Psychologist - PhD/PsyD, specializes in therapy
• Counselor/Therapist - Licensed for therapy (LCSW, LPC)
• Psychiatric Nurse Practitioner - Can prescribe and do therapy

**Getting started:**
• Check your insurance coverage
• Many therapists offer sliding scale fees
• Community mental health centers are affordable
• Employee Assistance Programs (EAP) often provide free sessions
• Online therapy (BetterHelp, Talkspace) if in-person isn't accessible

**First appointment tips:**
• Be honest about what you're experiencing
• It's okay if first therapist isn't the right fit
• Therapy takes time - give it 4-6 sessions

What specific support are you looking for? How urgent is your need?"""

    def _default_response(self):
        return """I'm here to support you with your mental health. I can help you talk through:

• **Emotional struggles** - Depression, anxiety, sadness, anger
• **Life stressors** - Relationships, work, loneliness, sleep
• **Coping strategies** - Evidence-based techniques for managing difficult feelings
• **Finding help** - Connecting you with professional resources

**How to get the most from our conversation:**
• Be specific about what you're experiencing
• Share how these feelings affect your daily life
• Let me know if this is new or ongoing
• Tell me what you've already tried

What's on your mind today? I'm here to listen without judgment."""

# Initialize chatbot instance
chatbot = MentalHealthChatbot()
