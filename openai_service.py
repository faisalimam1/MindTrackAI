import os
import random

def init_openai():
    """Initialize OpenAI - fallback version that doesn't require the package."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not set, using fallback responses")
    return True

class OpenAIService:
    """Fallback service for mental health support with predefined responses."""
    
    # Clinical mental health response templates
    RESPONSES = {
        'anxiety': [
            "I can see you're feeling anxious. Let's try one simple technique right now: Take a slow breath in for 4 counts, hold for 4, then out for 4. Try this 3 times. How does that feel?",
            "Anxiety can feel overwhelming. Here's what I want you to do: Look around and name 5 things you can see. This helps ground you in the present moment. What do you notice?",
            "Those anxious thoughts are racing, aren't they? Try this: Write down your biggest worry right now, then ask yourself 'What's one small thing I can actually control?' What comes to mind?",
            "Your body is holding tension from anxiety. Let's release it: Tense your shoulders for 5 seconds, then let them drop completely. Feel that relief? If anxiety persists daily, consider seeing a counselor.",
            "Anxiety is your nervous system trying to protect you, but it's overactive. Try placing your feet flat on the ground and pressing down. Feel that stability? How are you doing with this?"
        ],
        'depression': [
            "I hear that you're struggling with low mood. Let's start small: Can you get some sunlight today, even just 10 minutes by a window? Sunlight helps your brain make mood-lifting chemicals. How does that sound?",
            "Those negative thoughts are really loud right now. Try this: Write down one harsh thing you're telling yourself, then ask 'Would I say this to a good friend?' What would you say to them instead?",
            "It's hard to enjoy things when you're depressed - that's normal. Pick one tiny thing you used to like, maybe listening to a song or having tea. Can you try that today, even if you don't feel like it?",
            "Depression makes everything feel heavy. Here's one thing that helps: Take a 10-minute walk, even just around your room. Movement creates natural antidepressants. If this continues for weeks, please see a doctor.",
            "I'm concerned about how you're feeling. Are you having thoughts of hurting yourself? If so, please call NIMHANS (080-46110007) right now. You don't have to go through this alone."
        ],
        'stress': [
            "Stress is really getting to you right now. Let's bring it down a notch: Splash some cold water on your face or wrists. The cold helps reset your nervous system. Try it and tell me how you feel.",
            "Your stress levels are high. Here's a quick fix: Roll your shoulders back 5 times, then shake out your hands. Physical tension holds stress. Did that help release some pressure?",
            "Sounds like you're burning out. Today, can you say 'no' to just one thing that isn't absolutely necessary? Protecting your energy is crucial. What could you skip?",
            "You're using some unhealthy ways to cope with stress. Let's try one healthier swap: Instead of [your current coping method], try a 5-minute walk or deep breathing. If stress is affecting your daily life, consider seeing a counselor.",
            "Your stress is overwhelming your ability to cope. Right now, focus on just the next hour. What's one small thing you can do to feel slightly better? If this continues, professional support can really help."
        ],
        'sleep': [
            "Sleep troubles are exhausting. Tonight, try this one change: Put your phone away 1 hour before bed and keep your room cool. How's your bedtime routine currently?",
            "Can't fall asleep? When you're in bed, try the 4-7-8 breathing: breathe in for 4, hold for 7, out for 8. It naturally makes you drowsy. What usually keeps you awake?",
            "Poor sleep affects everything - mood, energy, thinking. Try writing your worries in a notebook 2 hours before bed, then close it. This helps your brain 'file away' concerns. If sleep problems continue, see a doctor."
        ],
        'screen_time': [
            "Too much screen time can really drain your energy and mood. Let's start with one small change: Can you put your phone in another room for just 30 minutes today? What do you think you'd do instead?",
            "Excessive screen time affects your brain's reward system. Try this: Set a timer for 20 minutes of screen-free time right now. Maybe go outside or stretch. How does that sound?",
            "Screen addiction is real and impacts mental health. Here's a gentle first step: Turn off notifications for social media apps. This reduces the urge to check constantly. What's your biggest screen time trigger?",
            "Digital overwhelm can cause anxiety and depression. Let's try the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds. This helps your eyes and mind reset. When do you use screens most?",
            "Constant scrolling can worsen mood and sleep. Tonight, try charging your phone outside your bedroom. Replace that bedtime scrolling with reading or gentle stretching. If screen time is affecting your daily life, consider talking to someone about digital wellness strategies."
        ],
        'eating_disorders': [
            "Eating disorders are serious but treatable. Right now, focus on one meal: Can you eat something nourishing without judgment? Try the 3-3-3 rule: Name 3 foods you enjoy, 3 reasons your body needs fuel, 3 people who care about you. If you're restricting, binging, or obsessing about food/weight, please reach out to an eating disorder specialist.",
            "Your relationship with food and your body is complex. Try this gentle approach: Before eating, take 3 breaths and ask 'What does my body need right now?' Honor hunger and fullness cues. If eating feels scary or out of control, specialized treatment can help you heal this relationship.",
            "Body image struggles are painful. Right now, try this: Look in the mirror and say one kind thing about what your body does for you (not how it looks). If you're engaging in harmful behaviors around food, exercise, or weight, please consider reaching out to an eating disorder treatment center."
        ],
        'addiction': [
            "Addiction is a medical condition, not a moral failing. If you're struggling with substances or behaviors, try the HALT check: Are you Hungry, Angry, Lonely, or Tired? Address these basic needs first. Consider calling a substance abuse helpline or attending a support group meeting today.",
            "Recovery is possible, and you don't have to do it alone. Right now, can you reach out to one supportive person? If you're thinking about using, try the 10-minute rule: wait 10 minutes and do something else. If you're ready for help, treatment programs and support groups are available.",
            "Cravings and urges are temporary - they will pass. Try the URGE surfing technique: Notice the craving, breathe through it, and wait for it to peak and decline like a wave. If you're in active addiction, medical detox and professional treatment can provide safe, effective support."
        ],
        'trauma_ptsd': [
            "Trauma responses are your nervous system trying to protect you. Right now, try grounding: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. This brings you back to the present moment. Trauma therapy with EMDR or other specialized approaches can help process these experiences safely.",
            "Flashbacks and triggers are real and valid. When triggered, try the STOP technique: Stop what you're doing, Take a breath, Observe your surroundings, Proceed mindfully. Your nervous system can heal with proper support - consider trauma-informed therapy.",
            "PTSD affects your whole system. Practice the 4-7-8 breathing: inhale 4, hold 7, exhale 8. This activates your calm response. If trauma is impacting your daily life, specialized trauma therapists can help you process and integrate these experiences."
        ],
        'ocd': [
            "OCD thoughts are not your fault and don't reflect who you are. When obsessive thoughts arise, try labeling: 'This is my OCD talking, not me.' Resist compulsions when possible - the anxiety will decrease naturally. ERP (Exposure Response Prevention) therapy is highly effective for OCD.",
            "Intrusive thoughts are common and don't mean anything about you. Try the 'leaves on a stream' technique: imagine your thoughts floating by on leaves in a stream. Don't engage or fight them, just observe. If OCD is interfering with your life, specialized OCD treatment can help.",
            "Compulsions provide temporary relief but strengthen OCD long-term. When you feel the urge to perform a ritual, try delaying it by 5 minutes. Use that time for deep breathing or a brief walk. OCD responds well to specialized therapy - you don't have to struggle alone."
        ],
        'adhd': [
            "ADHD brains work differently, not defectively. For focus issues, try the Pomodoro technique: 25 minutes focused work, 5-minute break. Break large tasks into smaller steps. If ADHD symptoms are impacting your life, consider evaluation by a specialist who understands neurodivergent brains.",
            "Executive function challenges are real. Try body doubling: work alongside someone else (in person or virtually). Use timers, lists, and external structure. If you're struggling with attention, hyperactivity, or organization, ADHD coaching or therapy can provide practical strategies.",
            "Time management with ADHD requires different strategies. Try time-blocking: assign specific times for tasks. Use visual reminders and break overwhelming projects into tiny steps. Medication and behavioral strategies together often provide the best support for ADHD."
        ],
        'autism': [
            "Sensory overload is overwhelming. Right now, find a quiet space and try noise-canceling headphones or soft music. Stimming is natural and helpful - don't suppress it. If you're struggling with social communication or sensory issues, autism-affirming therapists can provide support without trying to change who you are.",
            "Masking is exhausting. It's okay to take breaks from social demands. Practice unmasking in safe spaces - let your natural behaviors emerge. If you're newly discovering you might be autistic, seek out autistic communities and autism-informed professionals.",
            "Meltdowns happen when your nervous system is overwhelmed. After a meltdown, be gentle with yourself - rest, hydrate, and engage in preferred activities. If you need support navigating an autistic life, look for therapists who understand neurodivergence."
        ],
        'bipolar': [
            "Mood episodes can feel overwhelming. Track your moods daily to identify patterns and triggers. During depression, focus on basic self-care. During mania/hypomania, try to maintain routine and avoid major decisions. Bipolar disorder requires professional treatment - medication and therapy together are most effective.",
            "Rapid cycling moods are exhausting. Create a mood management plan: identify early warning signs, have coping strategies ready, and maintain consistent sleep. If you're experiencing extreme mood swings, a psychiatrist can help stabilize your mood with appropriate medication.",
            "Mixed episodes are particularly challenging. When you feel both depressed and agitated, focus on safety and grounding techniques. Reach out to your support system. Bipolar disorder is highly treatable with the right combination of medication, therapy, and lifestyle management."
        ],
        'social_issues': [
            "Social anxiety is treatable. Before social situations, try power posing for 2 minutes - it reduces stress hormones. Start with small social interactions and gradually build confidence. If social fears are limiting your life, CBT and exposure therapy can help you develop social confidence.",
            "Loneliness is painful but temporary. Try one small social connection today - text someone, comment positively online, or smile at a stranger. Join groups based on interests rather than just socializing. If isolation persists, therapy can help address underlying social fears.",
            "Making friends as an adult is challenging but possible. Focus on shared activities rather than forced conversation. Be genuinely interested in others and practice active listening. If social skills feel difficult, social skills training or therapy can provide practical tools."
        ],
        'work_burnout': [
            "Burnout is your body's way of saying 'enough.' Right now, take a 10-minute break from work demands. Practice saying 'no' to non-essential tasks. If you're experiencing chronic exhaustion and cynicism about work, consider speaking with HR about workload or seeking career counseling.",
            "Imposter syndrome affects high achievers. Keep a 'success journal' - write down daily accomplishments, no matter how small. Remember: you were hired for a reason. If work anxiety is overwhelming, therapy can help you develop confidence and boundary-setting skills.",
            "Work-life balance requires intentional boundaries. Try a 'shutdown ritual' - at day's end, write tomorrow's priorities and mentally 'close' work. If toxic workplace dynamics are affecting your mental health, consider whether this environment aligns with your values."
        ],
        'financial_stress': [
            "Financial stress affects mental health significantly. Right now, focus on what you can control today - maybe one small money-saving action. Break financial problems into smaller, manageable steps. If money worries are consuming your thoughts, financial counseling and therapy can help reduce anxiety.",
            "Money anxiety is common and valid. Try the 50/30/20 budgeting rule as a starting framework. Focus on needs vs. wants for this week only. If debt or financial insecurity is causing panic, nonprofit credit counseling services offer free guidance.",
            "Financial trauma is real. Your worth isn't determined by your bank account. Practice one small act of financial self-care today. If money issues are triggering shame or despair, therapy can help you develop a healthier relationship with finances."
        ],
        'family_issues': [
            "Family dysfunction affects your nervous system. You can't change family members, but you can change your responses. Try the 'gray rock' method with toxic family - be boring and unresponsive to drama. If family trauma is impacting your life, family therapy or individual therapy can help.",
            "Setting boundaries with family is challenging but necessary. Start small - maybe limiting one type of interaction. Remember: you can love someone and still protect your mental health from them. If family relationships are causing distress, therapy can help you navigate these complex dynamics.",
            "Generational trauma passes down through families unconsciously. You can be the one to break unhealthy patterns. Practice self-compassion and recognize that healing yourself helps heal the family system. Consider therapy to process family-of-origin issues."
        ],
        'grief_loss': [
            "Grief has no timeline and no 'right' way to process. Right now, be gentle with yourself. Do one small thing that honors your loss - light a candle, look at photos, or write a letter. If grief is overwhelming daily functioning, grief counseling can provide support through this difficult process.",
            "Loss changes you permanently, and that's normal. Allow yourself to feel whatever comes up without judgment. Create small rituals to remember what/who you've lost. If grief feels stuck or complicated, specialized grief therapy can help you process and integrate the loss.",
            "Complicated grief can feel like you're stuck in the pain. It's okay to seek help - grief counseling isn't about 'getting over it' but learning to carry the loss in a way that allows you to live fully again."
        ],
        'body_image': [
            "Body image struggles are painful in our appearance-focused culture. Right now, try body neutrality: instead of 'I love my body,' try 'My body is doing its best.' Focus on what your body can do rather than how it looks. If body image is severely impacting your life, therapy can help heal this relationship.",
            "Self-esteem isn't about appearance - it's about your inherent worth as a human. Practice speaking to yourself like you would a good friend. Challenge appearance-focused thoughts with evidence-based responses. If body shame is overwhelming, consider body-positive therapy approaches.",
            "Beauty standards are artificial and constantly changing. Your worth isn't determined by how closely you match current trends. Try a social media detox if comparison is triggering. If body dysmorphia is present, specialized treatment can help you see yourself more accurately."
        ],
        'academic_stress': [
            "Academic pressure can be overwhelming. Right now, break your biggest assignment into 15-minute chunks. You don't have to be perfect - 'good enough' is actually good enough. If school anxiety is impacting your performance, academic counseling and therapy can help.",
            "Test anxiety is treatable. Before exams, try progressive muscle relaxation: tense and release each muscle group. Prepare thoroughly but also practice self-compassion. If academic stress is causing panic or avoidance, consider speaking with school counselors about accommodations.",
            "Perfectionism in academics often backfires. Try the 80/20 rule - 80% effort often yields similar results to 100% effort but with much less stress. If academic burnout is affecting your mental health, it's okay to consider taking a break or reducing course load."
        ],
        'identity_crisis': [
            "Identity questions are normal, especially during life transitions. Right now, try journaling: 'What matters most to me?' and 'What brings me alive?' Identity develops over time - you don't need all the answers today. If existential concerns are causing distress, existential therapy can help explore meaning and purpose.",
            "Quarter-life and midlife crises are common developmental phases. Instead of fighting the uncertainty, try embracing it as growth. What would you do if you knew you couldn't fail? If identity confusion is causing anxiety or depression, therapy can help you explore and integrate different aspects of yourself.",
            "Finding your purpose doesn't have to be dramatic - it can be small daily choices aligned with your values. Try volunteering or helping someone today. If you're feeling lost or disconnected from meaning, spiritual counseling or existential therapy can provide guidance."
        ],
        'chronic_illness': [
            "Living with chronic illness requires grieving the life you expected while building the life you have. Right now, practice radical acceptance of your current limitations while advocating for your needs. If illness is affecting your mental health, therapy can help you adapt and thrive within your constraints.",
            "Chronic pain affects your whole system, including mood and relationships. Pace yourself and practice saying no to preserve energy for what matters most. If pain is causing depression or anxiety, integrated treatment addressing both physical and mental health is most effective.",
            "Health anxiety is common with chronic conditions. Try distinguishing between 'possible' and 'probable' when worry spirals start. Focus on what you can control today. If medical trauma or health anxiety is overwhelming, therapy can help you develop coping strategies."
        ],
        'relationship': [
            "Relationship issues can be really challenging. Let's start with one thing: What's the main pattern you notice in your conflicts? Is it communication, trust, or something else? Understanding the core issue helps us find the right approach.",
            "It sounds like you're going through a tough time in your relationship. Here's something to try: Before your next difficult conversation, take 5 minutes to write down what you really need from your partner. What would that look like?",
            "Relationship problems often stem from unmet needs or poor communication. Try this: Instead of saying 'You always...' or 'You never...', try 'I feel... when... I need...' What's one thing you'd like your partner to understand?",
            "Every relationship has challenges. The key is how you handle them together. Consider this: What drew you to your partner initially? Sometimes remembering the good can help navigate the difficult. If patterns keep repeating, couples therapy can provide tools you both need.",
            "Relationship stress affects your whole well-being. Right now, focus on what you can control - your own responses and communication. What's one small change you could make in how you interact? If the issues feel overwhelming, a relationship counselor can help you both learn healthier patterns."
        ],
        'professional_help': [
            "Based on what you're experiencing, here's who can help: **Anxiety/Panic**: Clinical Psychologist for CBT therapy. **Depression/Mood**: Psychiatrist for medication + Psychologist for therapy. **Stress/Burnout**: Licensed Counselor for coping strategies. **Trauma**: EMDR-trained therapist. **Relationship Issues**: Marriage/Family Therapist or Couples Counselor. What symptoms are you dealing with most?",
            "Different professionals help with different issues: **Therapist/Counselor**: Talk therapy, coping skills, relationship issues. **Psychologist**: Specialized therapy (CBT, DBT), psychological testing. **Psychiatrist**: Medication management, severe mental illness. **Marriage/Family Therapist**: Relationship and couples counseling. What's your main concern?",
            "Let me recommend the right professional: **Mild-moderate issues**: Licensed counselor or therapist. **Severe depression/anxiety**: Psychiatrist for medication evaluation. **Specific phobias/trauma**: Specialized psychologist. **Addiction**: Addiction counselor. **Relationship Problems**: Marriage/Family Therapist or Couples Counselor. What are you struggling with?",
            "Professional recommendations: **Crisis/suicidal thoughts**: Psychiatrist immediately. **Relationship problems**: Marriage/family therapist or couples counselor. **Work stress**: Licensed professional counselor. **Eating disorders**: Specialized therapist + psychiatrist. **Communication issues**: Relationship counselor. Tell me your specific challenges.",
            "Here's who to see: **General mental health**: Start with licensed counselor. **Need medication**: Psychiatrist. **Specialized therapy**: Clinical psychologist. **Crisis support**: Emergency psychiatric services. **Relationship Issues**: Marriage/Family Therapist for couples work or individual relationship counseling. What type of help are you looking for?"
        ],
        'crisis': [
            "IMMEDIATE SUICIDE RISK ASSESSMENT: Contact 911 or 988 Suicide & Crisis Lifeline NOW. Do you have a specific plan or means? Mandatory safety evaluation required.",
            "ACUTE SUICIDAL IDEATION: Emergency psychiatric evaluation needed. Go to nearest ER or call 988. Do you have someone for safety support?",
            "PSYCHIATRIC EMERGENCY: Contact 988 or emergency services immediately. Remove any means of self-harm from environment. Safety planning required."
        ],
        'general': [
            "Hi there. I'm here to help with whatever you're going through. What's been on your mind lately? Sometimes just talking about it can help.",
            "How are you feeling today? Whether it's stress, sadness, worry, or something else - I'm here to listen and help you find what works for you.",
            "Everyone goes through tough times. What's bringing you here today? We can work together to find some strategies that fit your situation.",
            "Mental health is just as important as physical health. What's one thing that's been challenging for you recently? Let's start there.",
            "I'm glad you're reaching out. Taking care of your mental health takes courage. What would you like to focus on today?"
        ]
    }
    
    CRISIS_KEYWORDS = [
        'suicide', 'kill myself', 'end my life', 'want to die', 'self-harm', 
        'hurt myself', 'no point living', 'better off dead', 'end it all',
        'not worth living', 'everyone would be better', 'can\'t go on', 'give up on life'
    ]
    
    TOPIC_KEYWORDS = {
        'anxiety': ['anxious', 'worried', 'panic', 'nervous', 'fear', 'stress', 'overwhelmed', 'panic attack', 'social anxiety', 'generalized anxiety', 'health anxiety', 'performance anxiety'],
        'depression': ['sad', 'depressed', 'hopeless', 'empty', 'worthless', 'tired', 'lonely', 'major depression', 'seasonal depression', 'postpartum depression', 'bipolar', 'mood swings', 'feeling numb'],
        'stress': ['stressed', 'pressure', 'overwhelmed', 'burnout', 'exhausted', 'tension', 'work stress', 'financial stress', 'chronic stress', 'acute stress'],
        'sleep': ['insomnia', 'sleep', 'tired', 'exhausted', 'can\'t sleep', 'nightmares', 'sleep disorder', 'sleep apnea', 'restless sleep', 'sleep anxiety'],
        'screen_time': ['screen time', 'too much screen', 'phone addiction', 'social media', 'scrolling', 'digital detox', 'phone all day', 'computer all day', 'online too much', 'internet addiction', 'gaming addiction', 'youtube addiction', 'tiktok addiction'],
        'eating_disorders': ['eating disorder', 'anorexia', 'bulimia', 'binge eating', 'body dysmorphia', 'food anxiety', 'restrictive eating', 'overeating', 'body image issues', 'weight obsession'],
        'addiction': ['addiction', 'substance abuse', 'alcohol problem', 'drug addiction', 'gambling addiction', 'shopping addiction', 'sex addiction', 'behavioral addiction', 'withdrawal', 'relapse'],
        'trauma_ptsd': ['trauma', 'ptsd', 'flashbacks', 'nightmares', 'triggered', 'childhood trauma', 'abuse', 'assault', 'accident trauma', 'war trauma', 'medical trauma', 'complex ptsd'],
        'ocd': ['ocd', 'obsessive', 'compulsive', 'intrusive thoughts', 'repetitive behaviors', 'checking', 'counting', 'contamination fears', 'symmetry obsessions'],
        'adhd': ['adhd', 'attention deficit', 'hyperactive', 'focus problems', 'concentration issues', 'executive function', 'procrastination', 'time management', 'organization problems'],
        'autism': ['autism', 'aspergers', 'sensory overload', 'social communication', 'stimming', 'meltdown', 'masking', 'neurodivergent'],
        'bipolar': ['bipolar', 'manic', 'hypomanic', 'mood episodes', 'cycling moods', 'mania', 'mixed episodes'],
        'personality_disorders': ['borderline', 'narcissistic', 'antisocial', 'avoidant personality', 'dependent personality', 'paranoid personality', 'identity issues', 'unstable relationships'],
        'social_issues': ['social anxiety', 'social phobia', 'loneliness', 'isolation', 'social skills', 'making friends', 'social rejection', 'peer pressure'],
        'work_burnout': ['burnout', 'work stress', 'job dissatisfaction', 'workplace anxiety', 'imposter syndrome', 'career crisis', 'work-life balance', 'toxic workplace'],
        'financial_stress': ['financial stress', 'money problems', 'debt anxiety', 'unemployment', 'financial insecurity', 'poverty stress', 'economic anxiety'],
        'family_issues': ['family problems', 'toxic family', 'family conflict', 'parenting stress', 'sibling rivalry', 'family dysfunction', 'generational trauma'],
        'grief_loss': ['grief', 'loss', 'bereavement', 'mourning', 'death of loved one', 'pet loss', 'job loss', 'relationship loss', 'complicated grief'],
        'body_image': ['body image', 'self-esteem', 'appearance anxiety', 'weight issues', 'beauty standards', 'body shame', 'dysmorphia'],
        'academic_stress': ['academic stress', 'school anxiety', 'exam stress', 'study pressure', 'college stress', 'academic burnout', 'performance pressure'],
        'identity_crisis': ['identity crisis', 'who am i', 'life purpose', 'existential crisis', 'quarter life crisis', 'midlife crisis', 'meaning of life'],
        'chronic_illness': ['chronic illness', 'chronic pain', 'disability', 'health anxiety', 'medical trauma', 'illness anxiety', 'chronic fatigue'],
        'relationship': ['relationship issues', 'relationship problems', 'relationship trouble', 'partner problems', 'marriage issues', 'dating problems', 'breakup', 'divorce', 'couples therapy', 'relationship advice', 'communication issues', 'trust issues', 'intimacy problems', 'conflict with partner', 'relationship stress', 'love problems', 'romantic problems', 'relationship counseling', 'couples counseling', 'marriage counseling'],
        'professional_help': ['which doctor', 'what professional', 'therapist or psychiatrist', 'who should i see', 'what kind of doctor', 'professional recommendation', 'specialist recommendation', 'mental health professional'],
        'coping': ['coping strategies', 'help me cope', 'what can i do', 'strategies', 'techniques', 'exercises', 'tools'],
        'books_reading': ['book', 'books', 'reading', 'suggest a book', 'recommend book', 'book recommendation', 'what to read', 'good books', 'calming books', 'self help books', 'mindfulness books', 'mental health books', 'therapeutic books', 'healing books', 'inspirational books', 'motivational books', 'psychology books', 'wellness books', 'meditation books', 'anxiety books', 'depression books', 'stress relief books'],
        'crisis': CRISIS_KEYWORDS
    }
    
    COPING_RESPONSES = [
        "Let's find what works for you. Right now, try this: Take 3 deep breaths (4 counts in, 8 counts out). Feel a bit calmer? What specific situation is stressing you most today?",
        "Coping strategies work best when they fit your life. What usually helps you feel better - talking to someone, moving your body, or quiet time alone? Let's build on what already works for you.",
        "When everything feels overwhelming, start with your body. Try this: Tense all your muscles for 5 seconds, then completely relax. Notice the difference? What's your biggest challenge right now?",
        "Quick relief technique: Put your hand on your chest and feel your heartbeat. This grounds you in your body. Now, what's one small step you can take today to feel a bit better?",
        "Everyone needs different coping tools. Some people need movement, others need stillness. What feels right for you today - a walk, some stretching, or just sitting quietly? If you're struggling daily, talking to a counselor can really help."
    ]
    
    @staticmethod
    def detect_topic(text):
        """Detect the main topic from user input."""
        text_lower = text.lower()
        print(f"DEBUG: Input text: '{text_lower}'")  # Debug line
        
        # Check for coping strategy requests first
        if any(keyword in text_lower for keyword in OpenAIService.TOPIC_KEYWORDS['coping']):
            print("DEBUG: Detected coping topic")
            return 'coping'
        
        for topic, keywords in OpenAIService.TOPIC_KEYWORDS.items():
            if topic not in ['coping']:
                for keyword in keywords:
                    if keyword in text_lower:
                        print(f"DEBUG: Detected topic '{topic}' with keyword '{keyword}'")
                        return topic
        
        print("DEBUG: No specific topic detected, returning 'general'")
        return 'general'
    
    @staticmethod
    def detect_crisis(text):
        """Detect crisis situations."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in OpenAIService.CRISIS_KEYWORDS)
    
    @staticmethod
    def analyze_safety(text):
        """Analyze message for safety concerns and risk factors."""
        text_lower = text.lower()
        
        # Check for crisis indicators
        crisis_detected = OpenAIService.detect_crisis(text)
        
        if crisis_detected:
            return {
                "risk_level": "high",
                "concerns": ["suicidal_ideation", "self_harm"],
                "requires_escalation": True,
                "suggested_response": "I'm deeply concerned about your safety. Please reach out for immediate professional help."
            }
        
        # Check for moderate risk indicators
        moderate_risk_phrases = [
            'hopeless', 'worthless', 'can\'t go on', 'nothing matters', 'give up',
            'no point', 'can\'t take it', 'too much pain', 'want it to stop',
            'can\'t handle', 'breaking point', 'lost all hope'
        ]
        
        moderate_risk = any(phrase in text_lower for phrase in moderate_risk_phrases)
        
        if moderate_risk:
            return {
                "risk_level": "moderate",
                "concerns": ["severe_distress", "hopelessness"],
                "requires_escalation": True,
                "suggested_response": "I can hear how much pain you're in. Professional support could provide you with additional tools and resources."
            }
        
        return {
            "risk_level": "low",
            "concerns": [],
            "requires_escalation": False,
            "suggested_response": "I'm here to support you through whatever you're experiencing."
        }
    
    @staticmethod
    def extract_user_situation(text):
        """Extract key elements from user's message for contextual follow-up."""
        text_lower = text.lower()
        situation = {
            'problems_mentioned': [],
            'emotions_expressed': [],
            'impacts_described': [],
            'relationships_involved': [],
            'time_context': None,
            'help_seeking': False
        }
        
        # Extract problems/challenges
        problem_patterns = {
            'family': ['family problems', 'family issues', 'family conflict', 'family drama', 'toxic family', 'family stress'],
            'work': ['work problems', 'job stress', 'workplace', 'career', 'boss', 'colleagues', 'deadlines'],
            'relationship': ['relationship problems', 'partner', 'boyfriend', 'girlfriend', 'marriage', 'dating'],
            'financial': ['money problems', 'financial', 'debt', 'bills', 'broke', 'unemployment'],
            'health': ['health problems', 'sick', 'illness', 'pain', 'medical', 'doctor'],
            'academic': ['school', 'college', 'studies', 'exams', 'grades', 'university'],
            'social': ['friends', 'social', 'lonely', 'isolated', 'rejected'],
            'books_reading': ['book', 'books', 'reading', 'recommend', 'suggest', 'what to read']
        }
        
        for category, patterns in problem_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                situation['problems_mentioned'].append(category)
        
        # Extract emotional states
        emotion_patterns = {
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear'],
            'depression': ['depressed', 'sad', 'hopeless', 'empty', 'worthless'],
            'anger': ['angry', 'frustrated', 'mad', 'furious', 'irritated'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'tension'],
            'confusion': ['confused', 'lost', 'don\'t know', 'uncertain']
        }
        
        for emotion, patterns in emotion_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                situation['emotions_expressed'].append(emotion)
        
        # Extract impacts/consequences
        impact_patterns = {
            'concentration': ['can\'t concentrate', 'focus', 'distracted', 'attention'],
            'sleep': ['can\'t sleep', 'insomnia', 'tired', 'exhausted'],
            'performance': ['performance', 'productivity', 'work suffering', 'grades'],
            'relationships': ['affecting relationships', 'pushing people away', 'isolating'],
            'physical': ['headaches', 'stomach', 'physical symptoms']
        }
        
        for impact, patterns in impact_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                situation['impacts_described'].append(impact)
        
        # Detect help-seeking language
        help_patterns = ['what should i do', 'help me', 'advice', 'suggestions', 'what can i try']
        situation['help_seeking'] = any(pattern in text_lower for pattern in help_patterns)
        
        return situation
    
    @staticmethod
    def analyze_conversation_context(conversation_history):
        """Analyze conversation history for context-aware responses."""
        context = {
            'last_bot_message': None,
            'last_user_message': None,
            'suggested_technique': None,
            'awaiting_followup': False,
            'conversation_topic': None,
            'user_situation': None
        }
        
        if not conversation_history:
            return context
            
        # Get last few messages for context
        recent_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
        
        for msg in reversed(recent_messages):
            if msg.get('role') == 'assistant' and not context['last_bot_message']:
                context['last_bot_message'] = msg.get('content', '')
            elif msg.get('role') == 'user' and not context['last_user_message']:
                context['last_user_message'] = msg.get('content', '')
        
        # Check if bot suggested a technique that needs follow-up
        if context['last_bot_message']:
            bot_msg = context['last_bot_message'].lower()
            
            # Detect if bot suggested specific techniques or activities
            suggestion_patterns = {
                'grounding_5_things': ['name 5 things', 'look around and name', '5 things you can see'],
                'breathing_exercise': ['take 3 deep breaths', 'breathe in for 4', '4-7-8 breathing', 'try breathing', 'breathing technique'],
                'muscle_relaxation': ['tense your muscles', 'progressive muscle relaxation', 'tense and release'],
                'heartbeat_grounding': ['put your hand on your chest', 'feel your heartbeat', 'heartbeat'],
                'journaling': ['write down', 'journal', 'try writing'],
                'music_therapy': ['listen to music', 'try music', 'songs', 'listen to some songs', 'music can help'],
                'physical_activity': ['take a walk', 'go for a walk', 'try walking', 'exercise', 'physical activity', 'movement'],
                'self_care': ['self-care', 'take care of yourself', 'do something nice', 'treat yourself'],
                'social_connection': ['reach out', 'talk to someone', 'call a friend', 'social support'],
                'mindfulness': ['mindfulness', 'meditation', 'be present', 'focus on now'],
                'sleep_hygiene': ['sleep routine', 'bedtime', 'sleep schedule', 'rest'],
                'nutrition': ['eat something', 'healthy meal', 'nutrition', 'food']
            }
            
            for technique, patterns in suggestion_patterns.items():
                if any(pattern in bot_msg for pattern in patterns):
                    context['awaiting_followup'] = True
                    context['suggested_technique'] = technique
                    break
        
        return context
    
    @staticmethod
    def generate_situational_response(user_text, situation):
        """Generate responses based on user's specific situation."""
        responses = []
        
        # Handle complex multi-problem scenarios
        if len(situation['problems_mentioned']) > 1:
            problems = ' and '.join(situation['problems_mentioned'])
            if 'concentration' in situation['impacts_described']:
                responses.extend([
                    f"It sounds like you're dealing with {problems}, and this is making it hard to concentrate. That's a really common experience - when we're stressed about multiple areas of life, our brain has trouble focusing on any one thing. Which of these feels most urgent or overwhelming right now?",
                    f"I can hear that {problems} are creating a cycle where it's hard to focus. When our mind is pulled in different directions, concentration suffers. Let's start with one area - which problem feels like it's taking up the most mental space for you?",
                    f"You're managing {problems} while trying to stay focused - that's incredibly challenging. Our brains aren't designed to compartmentalize stress easily. What would it look like if you could address just one of these issues first?"
                ])
        
        # Handle family problems affecting work
        if 'family' in situation['problems_mentioned'] and 'work' in situation['problems_mentioned']:
            responses.extend([
                "Family stress bleeding into work life is exhausting. It's hard to leave family problems at home when they're weighing on your mind. Are these family issues something that needs immediate attention, or are they ongoing situations you're trying to manage?",
                "When family problems affect our work concentration, it often means the family stress feels unresolved or urgent. What's the most challenging part of the family situation right now? Sometimes naming the specific issue helps us figure out next steps.",
                "It's really difficult to focus on work when family dynamics are chaotic or stressful. Are you finding yourself thinking about the family situation while you're trying to work, or is it more that the stress is affecting your overall energy and focus?"
            ])
        
        # Handle specific emotional + impact combinations
        if 'stress' in situation['emotions_expressed'] and 'concentration' in situation['impacts_described']:
            responses.extend([
                "Stress and concentration problems often go hand in hand - when we're stressed, our brain prioritizes scanning for threats over focusing on tasks. What's contributing most to your stress levels right now?",
                "That combination of feeling stressed and unable to concentrate is really frustrating. Stress literally changes how our brain processes information. What does your stress feel like in your body - tension, racing thoughts, restlessness?"
            ])
        
        # Handle book/reading requests with context
        if 'books_reading' in situation['problems_mentioned']:
            if situation['emotions_expressed']:
                main_emotion = situation['emotions_expressed'][0]
                responses.extend([
                    f"I can suggest some excellent books for when you're feeling {main_emotion}. What type of reading usually appeals to you - something practical with exercises, inspiring stories, or gentle philosophy? Also, are you dealing with any specific challenges I should consider?",
                    f"Books can be wonderful for managing {main_emotion}. Are you looking for something to help with anxiety, depression, stress, or just general peace of mind? I can recommend books based on what you're going through.",
                    "Reading for mental health is so beneficial. What's your current situation - are you feeling overwhelmed, anxious, sad, or just need some mental escape? This will help me suggest the most relevant books for you.",
                    "There are some amazing books that can really help calm and redirect your thoughts. Tell me a bit about what you're experiencing right now - stress, worry, sadness, or just need a mental break? I want to recommend something that truly fits your needs.",
                    "Books can be incredibly therapeutic. What's drawing you to reading right now - are you looking for coping strategies, inspiration, distraction, or understanding? And what's your preferred style - practical guides, memoirs, or gentle fiction?"
                ])
            elif other_problems := [p for p in situation['problems_mentioned'] if p != 'books_reading']:
                main_problem = other_problems[0]
                responses.extend([
                    f"I'd love to recommend books that specifically address {main_problem} situations. What's your reading preference - practical guides with actionable steps, personal stories from others who've been through similar experiences, or something more philosophical and reflective?",
                    f"There are some really helpful books for navigating {main_problem}. To give you the best recommendations, what aspect of this situation feels most challenging right now? This will help me suggest the most relevant reading."
                ])
            else:
                responses.extend([
                    "I'm happy to suggest some calming, mind-diverting books. What's prompting you to look for reading recommendations right now? Are you seeking comfort, distraction, personal growth, or just something peaceful to focus on?",
                    "Books can be such great companions for mental wellness. What kind of mental space are you hoping to create through reading - something soothing and escapist, or more focused on personal insight and growth?"
                ])
        
        # Handle help-seeking with specific problems
        elif situation['help_seeking'] and situation['problems_mentioned']:
            main_problem = situation['problems_mentioned'][0]
            responses.extend([
                f"I can hear that you're looking for guidance with your {main_problem} situation. Let's break this down - what's one specific aspect of this {main_problem} issue that feels most manageable to address first?",
                f"You're reaching out for help with {main_problem}, which takes courage. To give you the most relevant support, can you tell me what you've already tried to address this situation?"
            ])
        
        return responses
    
    @staticmethod
    def generate_followup_response(user_text, context):
        """Generate contextual follow-up responses based on previous suggestions and current situation."""
        user_lower = user_text.lower()
        technique = context.get('suggested_technique')
        
        # First check for technique-specific follow-ups (existing code)
        if technique == 'grounding_5_things':
            # User engaged with grounding exercise
            if any(word in user_lower for word in ['see', 'around', 'there is', 'i can see', 'chair', 'table', 'wall', 'window', 'door', 'book', 'phone', 'computer', 'light']):
                return [
                    "Great job engaging with the grounding technique! I can hear that you're noticing your environment. How do you feel right now compared to a few minutes ago? Sometimes just shifting our attention to the present moment can provide some relief.",
                    "Thank you for trying that grounding exercise. Now that you've connected with your immediate surroundings, let's build on this. Can you also notice 3 sounds you hear right now? This helps deepen the grounding effect.",
                    "Excellent - you're actively grounding yourself in the present moment. Notice how focusing on concrete details around you can interrupt anxious thoughts. What's one thing you noticed that you hadn't paid attention to before?"
                ]
        
        elif technique == 'music_therapy':
            # User tried music but it didn't help or had mixed results
            if any(phrase in user_lower for phrase in ['listened', 'heard', 'song', 'music', 'tried it', 'didn\'t help', 'didn\'t work', 'not satisfied', 'didn\'t satisfy', 'still feel', 'alternative']):
                return [
                    "I hear that music didn't provide the relief you were hoping for. That's completely valid - different coping strategies work for different people and situations. Let's try something else: What about gentle movement? Even stretching your arms above your head or rolling your shoulders can help shift your mood.",
                    "Thank you for trying the music suggestion. Since that didn't resonate with you right now, let's explore other options. Sometimes when we're feeling low, our body needs attention too. Could you try taking 5 slow, deep breaths while gently massaging your temples? How does that feel?",
                    "I appreciate you giving music a try, even though it didn't lift your spirits this time. Let's pivot to something different. What about connecting with nature? If possible, could you step outside for just 2 minutes, or even look out a window and describe what you see? Sometimes a change of environment helps.",
                    "It sounds like music wasn't the right fit for how you're feeling today, and that's okay. Let's try a different approach: What's one very small thing that usually brings you even a tiny bit of comfort? Maybe a warm drink, a soft blanket, or calling someone who cares about you?"
                ]
        
        elif technique == 'breathing_exercise':
            if any(word in user_lower for word in ['breathed', 'breathing', 'breath', 'tried it', 'did it', 'better', 'calmer', 'helped']):
                return [
                    "Wonderful that you tried the breathing exercise! How did that feel? Even a few conscious breaths can help reset your nervous system. If it helped, you can use this technique anytime you feel overwhelmed.",
                    "Good work with the breathing. Your nervous system is already starting to calm down. Let's add another layer - as you breathe, try saying 'calm' on the inhale and 'peace' on the exhale. How does that feel?",
                    "I'm glad you engaged with the breathing technique. Notice any changes in your body - maybe your shoulders dropped or your jaw relaxed? What do you notice is different now?"
                ]
        
        elif technique == 'muscle_relaxation':
            if any(word in user_lower for word in ['tensed', 'relaxed', 'tried', 'muscles', 'shoulders', 'better', 'relief']):
                return [
                    "Excellent work with the muscle relaxation! That contrast between tension and release helps your body remember what relaxation feels like. Which part of your body feels most relaxed now?",
                    "Great job trying that technique. Progressive muscle relaxation works because it gives your body a clear signal to let go of stress. How are you feeling in your body right now?",
                    "Thank you for engaging with that exercise. Your body is learning to release tension on command. What would help you remember to use this technique when stress builds up?"
                ]
        
        elif technique == 'heartbeat_grounding':
            if any(word in user_lower for word in ['heartbeat', 'chest', 'heart', 'feel', 'tried']):
                return [
                    "Perfect - connecting with your heartbeat is a powerful grounding anchor. Your heart is always there, steady and reliable. How does it feel to tune into that rhythm right now?",
                    "Good work focusing on your heartbeat. This technique works because it connects you to your body's natural rhythm. What do you notice about your heart rate now compared to when we started?",
                    "Excellent grounding work. Your heartbeat is like a built-in meditation bell - always available to bring you back to the present. How might you remember to use this when you feel scattered?"
                ]
        
        elif technique == 'journaling':
            if any(word in user_lower for word in ['wrote', 'writing', 'journal', 'thoughts', 'wrote down']):
                return [
                    "Thank you for taking the time to write. Getting thoughts out of your head and onto paper can provide real relief. What did you notice as you were writing? Sometimes the act itself is as helpful as what we write.",
                    "Great work with the journaling. Writing helps organize chaotic thoughts and feelings. Did anything surprise you about what came out on paper?",
                    "I'm glad you tried writing it down. Journaling creates distance between you and overwhelming thoughts. How do those concerns feel now that they're outside your head?"
                ]
        
        elif technique == 'physical_activity':
            if any(phrase in user_lower for phrase in ['walked', 'walk', 'tried', 'exercise', 'moved', 'didn\'t help', 'still feel', 'alternative']):
                return [
                    "I appreciate you trying to move your body. Physical activity affects everyone differently. If walking didn't shift your mood, let's try something gentler. Could you try some slow neck rolls or shoulder shrugs right where you are? Sometimes smaller movements can be more effective.",
                    "Thank you for giving movement a try. Since that approach didn't provide the relief you needed, let's explore a different angle. What about focusing on your breath while doing something with your hands - maybe organizing a small area or making a warm drink? How does that sound?"
                ]
        
        elif technique == 'social_connection':
            if any(phrase in user_lower for phrase in ['called', 'talked', 'reached out', 'texted', 'didn\'t help', 'made it worse', 'felt worse']):
                return [
                    "I hear that reaching out didn't provide the support you were hoping for. Sometimes social connection can feel draining when we're already struggling. Let's try something that focuses just on you: What's one thing you can do right now that feels nurturing to yourself?",
                    "Thank you for trying to connect with others, even though it didn't lift your spirits. Sometimes we need to fill our own cup first. What's one small act of self-compassion you could show yourself right now?"
                ]
        
        elif technique == 'self_care':
            if any(phrase in user_lower for phrase in ['tried', 'did', 'didn\'t work', 'still feel', 'not better', 'alternative']):
                return [
                    "I appreciate you attempting some self-care. When traditional self-care doesn't hit the mark, sometimes we need to go even smaller. What's the tiniest thing that might bring you a moment of comfort right now - maybe just changing your position or having a sip of water?",
                    "Thank you for trying that self-care approach. Since it didn't provide the relief you needed, let's think differently. Sometimes when we're struggling, 'self-care' needs to be more about basic needs. Have you eaten or had water recently? Sometimes our mood is connected to these fundamentals."
                ]
        
        # Universal situational follow-up - analyze what user actually said
        current_situation = OpenAIService.extract_user_situation(user_text)
        if current_situation['problems_mentioned'] or current_situation['emotions_expressed'] or current_situation['impacts_described']:
            situational_responses = OpenAIService.generate_situational_response(user_text, current_situation)
            if situational_responses:
                return situational_responses
        
        # Catch-all for any engagement that wasn't specifically matched
        engagement_indicators = ['tried', 'did', 'attempted', 'i', 'but', 'however', 'still', 'didn\'t', 'not', 'alternative', 'else', 'other']
        if any(indicator in user_lower for indicator in engagement_indicators):
            return [
                "I can hear that you tried something, and it sounds like you're looking for what might work better for you. That's actually really insightful - recognizing when something isn't quite right is important. What feels most challenging for you right now in this moment?",
                "Thank you for engaging with my suggestion, even if it didn't provide the relief you were hoping for. Everyone responds differently to coping strategies. What's one thing that has helped you feel even slightly better in the past?",
                "I appreciate you trying that approach. Since it didn't quite meet your needs, let's explore together. What would 'feeling a bit better' look like for you right now? Even a small shift?"
            ]
        
        return None  # No specific follow-up found
    
    @staticmethod
    def generate_response(text, conversation_history=None):
        """Generate a mental health response based on input text and conversation history."""
        try:
            # Detect crisis situations first
            if OpenAIService.detect_crisis(text):
                response = random.choice(OpenAIService.RESPONSES['crisis'])
                return {
                    'response': response,
                    'requires_escalation': True,
                    'risk_level': 'high',
                    'concerns': ['suicidal ideation', 'self-harm risk']
                }
            
            # Get conversation context
            context = OpenAIService.analyze_conversation_context(conversation_history or [])
            print(f"DEBUG: Context analysis: {context}")
            
            # Check if this is a follow-up to a suggested technique
            if context.get('awaiting_followup'):
                followup_responses = OpenAIService.generate_followup_response(text, context)
                if followup_responses:
                    response = random.choice(followup_responses)
                    print(f"DEBUG: Using contextual follow-up response for {context.get('suggested_technique')}")
                    
                    # Analyze safety for follow-up
                    safety_analysis = OpenAIService.analyze_safety(text)
                    
                    return {
                        'response': response,
                        'requires_escalation': safety_analysis['requires_escalation'],
                        'risk_level': safety_analysis['risk_level'],
                        'concerns': safety_analysis['concerns'],
                        'topic': 'followup',
                        'context_aware': True
                    }
                else:
                    print(f"DEBUG: No follow-up match found for technique {context.get('suggested_technique')} with user input: {text[:50]}...")
            
            # Detect topic and generate appropriate response
            topic = OpenAIService.detect_topic(text)
            print(f"DEBUG: Final detected topic: '{topic}'")
            
            # Select response based on topic - now includes all new mental health categories
            if topic == 'coping':
                response = random.choice(OpenAIService.COPING_RESPONSES)
                print("DEBUG: Using coping responses")
            elif topic in OpenAIService.RESPONSES:
                response = random.choice(OpenAIService.RESPONSES[topic])
                print(f"DEBUG: Using {topic} responses")
            else:
                # Check if it's a book request that wasn't caught by topic detection
                current_situation = OpenAIService.extract_user_situation(text)
                if 'books_reading' in current_situation['problems_mentioned']:
                    response = random.choice(OpenAIService.RESPONSES['books_reading'])
                    topic = 'books_reading'
                    print("DEBUG: Using books_reading responses")
                else:
                    response = random.choice(OpenAIService.RESPONSES['general'])
                    print("DEBUG: Using general responses")
                print("DEBUG: Using general responses")
            
            print(f"DEBUG: Selected response: '{response[:100]}...'")
            
            # Analyze safety
            safety_analysis = OpenAIService.analyze_safety(text)
            
            # Ensure current_situation is defined for context_aware check
            if 'current_situation' not in locals():
                current_situation = OpenAIService.extract_user_situation(text)
            
            return {
                'response': response,
                'requires_escalation': safety_analysis['requires_escalation'],
                'risk_level': safety_analysis['risk_level'],
                'concerns': safety_analysis['concerns'],
                'topic': topic,
                'context_aware': topic == 'books_reading' or bool(current_situation.get('problems_mentioned'))
            }
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Even in error cases, try to be contextual
            try:
                situation = OpenAIService.extract_user_situation(text)
                if 'books_reading' in situation.get('problems_mentioned', []):
                    fallback_response = "I'd love to help you find some good books. What kind of reading are you interested in - something calming, inspiring, or practical?"
                elif situation.get('emotions_expressed'):
                    emotion = situation['emotions_expressed'][0]
                    fallback_response = f"I can hear that you're feeling {emotion}. What's the most challenging part of what you're experiencing right now?"
                elif situation.get('problems_mentioned'):
                    problem = situation['problems_mentioned'][0]
                    fallback_response = f"It sounds like you're dealing with {problem} issues. What aspect of this situation feels most overwhelming?"
                else:
                    fallback_response = "I'm here to support you. What's on your mind today?"
            except Exception:
                fallback_response = "I'm here to support you. What's on your mind today?"
            
            return {
                'response': fallback_response,
                'requires_escalation': False,
                'risk_level': 'none',
                'concerns': [],
                'topic': 'general',
                'context_aware': True
            }
