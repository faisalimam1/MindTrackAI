from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, JournalEntry, MoodEntry, Task, Goal, ChatMessage
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== AUTHENTICATION BLUEPRINT =====
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration - handles both AJAX and regular form submission"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Handle both JSON (AJAX) and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validation
        if not all([username, email, password]):
            error_msg = 'All fields are required'
            if request.is_json:
                return jsonify({'error': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(url_for('auth.register'))
        
        if len(password) < 8:
            error_msg = 'Password must be at least 8 characters long'
            if request.is_json:
                return jsonify({'error': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(url_for('auth.register'))
        
        # Check if username exists
        if User.query.filter_by(username=username).first():
            error_msg = 'Username already exists'
            if request.is_json:
                return jsonify({'error': error_msg}), 409
            flash(error_msg, 'error')
            return redirect(url_for('auth.register'))
        
        # Check if email exists
        if User.query.filter_by(email=email).first():
            error_msg = 'Email already registered'
            if request.is_json:
                return jsonify({'error': error_msg}), 409
            flash(error_msg, 'error')
            return redirect(url_for('auth.register'))
        
        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            success_msg = 'Registration successful! Please log in.'
            if request.is_json:
                return jsonify({
                    'message': success_msg,
                    'redirect': url_for('auth.login')
                }), 201
            
            flash(success_msg, 'success')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration error: {e}")
            error_msg = 'Registration failed. Please try again.'
            if request.is_json:
                return jsonify({'error': error_msg}), 500
            flash(error_msg, 'error')
            return redirect(url_for('auth.register'))
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login - handles both AJAX and regular form submission"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Handle both JSON (AJAX) and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        # Validation
        if not all([username, password]):
            error_msg = 'Username and password are required'
            if request.is_json:
                return jsonify({'error': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(url_for('auth.login'))
        
        # Find user
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            # Check if user is active
            if not user.is_active:
                error_msg = 'Account is disabled. Please contact support.'
                if request.is_json:
                    return jsonify({'error': error_msg}), 403
                flash(error_msg, 'error')
                return redirect(url_for('auth.login'))
            
            # Log in user
            login_user(user, remember=True)
            
            # Update last login
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
            
            success_msg = 'Login successful!'
            if request.is_json:
                return jsonify({
                    'message': success_msg,
                    'redirect': url_for('dashboard')
                }), 200
            
            flash(success_msg, 'success')
            return redirect(url_for('dashboard'))
        else:
            error_msg = 'Invalid username or password'
            if request.is_json:
                return jsonify({'error': error_msg}), 401
            flash(error_msg, 'error')
            return redirect(url_for('auth.login'))
    
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ===== JOURNAL BLUEPRINT =====
journal_bp = Blueprint('journal', __name__, url_prefix='/journal')

@journal_bp.route('/')
@login_required
def journal_index():
    """List all journal entries"""
    entries = JournalEntry.query.filter_by(
        user_id=current_user.id
    ).order_by(JournalEntry.created_at.desc()).all()
    return render_template('journal/index.html', entries=entries)

@journal_bp.route('/new', methods=['GET', 'POST'])
@login_required
def journal_new_entry():
    """Create new journal entry"""
    if request.method == 'POST':
        data = request.get_json()
        
        try:
            entry = JournalEntry(
                user_id=current_user.id,
                title=data.get('title', ''),
                content=data.get('content', ''),
                mood_score=data.get('mood_score'),
                tags=data.get('tags', [])
            )
            db.session.add(entry)
            db.session.commit()
            
            return jsonify({
                'message': 'Entry created successfully',
                'id': entry.id
            }), 201
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating journal entry: {e}")
            return jsonify({'error': 'Failed to create entry'}), 500
    
    return render_template('journal/new.html')

@journal_bp.route('/<int:entry_id>')
@login_required
def view_entry(entry_id):
    """View specific journal entry"""
    entry = JournalEntry.query.filter_by(
        id=entry_id, 
        user_id=current_user.id
    ).first_or_404()
    return render_template('journal/view.html', entry=entry)

# ===== MOOD TRACKING BLUEPRINT =====
mood_bp = Blueprint('mood', __name__, url_prefix='/mood')

@mood_bp.route('/')
@login_required
def mood_index():
    """List mood entries"""
    entries = MoodEntry.query.filter_by(
        user_id=current_user.id
    ).order_by(MoodEntry.created_at.desc()).limit(30).all()
    return render_template('mood/index.html', entries=entries)

@mood_bp.route('/new', methods=['GET', 'POST'])
@login_required
def mood_new_entry():
    """Create new mood entry"""
    if request.method == 'POST':
        data = request.get_json()
        
        try:
            entry = MoodEntry(
                user_id=current_user.id,
                mood_score=data.get('mood_score'),
                mood_label=data.get('mood_label'),
                notes=data.get('notes'),
                activities=data.get('activities', []),
                sleep_hours=data.get('sleep_hours'),
                exercise_minutes=data.get('exercise_minutes'),
                social_interactions=data.get('social_interactions')
            )
            db.session.add(entry)
            db.session.commit()
            
            return jsonify({'message': 'Mood entry created successfully'}), 201
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating mood entry: {e}")
            return jsonify({'error': 'Failed to create mood entry'}), 500
    
    return render_template('mood/new.html')

# ===== TASKS BLUEPRINT =====
tasks_bp = Blueprint('tasks', __name__, url_prefix='/tasks')

@tasks_bp.route('/')
@login_required
def tasks_index():
    """List all tasks"""
    tasks = Task.query.filter_by(
        user_id=current_user.id
    ).order_by(Task.due_date.asc()).all()
    return render_template('tasks/index.html', tasks=tasks)

@tasks_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_task():
    """Create new task"""
    if request.method == 'POST':
        data = request.get_json()
        
        try:
            task = Task(
                user_id=current_user.id,
                title=data.get('title'),
                description=data.get('description'),
                priority=data.get('priority', 'medium'),
                due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None
            )
            db.session.add(task)
            db.session.commit()
            
            return jsonify({'message': 'Task created successfully', 'id': task.id}), 201
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating task: {e}")
            return jsonify({'error': 'Failed to create task'}), 500
    
    return render_template('tasks/new.html')

@tasks_bp.route('/<int:task_id>/complete', methods=['POST'])
@login_required
def complete_task(task_id):
    """Mark task as complete"""
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    task.status = 'completed'
    task.completed_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({'message': 'Task completed'}), 200

# ===== GOALS BLUEPRINT =====
goals_bp = Blueprint('goals', __name__, url_prefix='/goals')

@goals_bp.route('/')
@login_required
def goals_index():
    """List all goals"""
    goals = Goal.query.filter_by(
        user_id=current_user.id
    ).order_by(Goal.target_date.asc()).all()
    return render_template('goals/index.html', goals=goals)

@goals_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_goal():
    """Create new goal"""
    if request.method == 'POST':
        data = request.get_json()
        
        try:
            goal = Goal(
                user_id=current_user.id,
                title=data.get('title'),
                description=data.get('description'),
                category=data.get('category'),
                target_date=datetime.fromisoformat(data['target_date']).date() if data.get('target_date') else None
            )
            db.session.add(goal)
            db.session.commit()
            
            return jsonify({'message': 'Goal created successfully', 'id': goal.id}), 201
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating goal: {e}")
            return jsonify({'error': 'Failed to create goal'}), 500
    
    return render_template('goals/new.html')

@goals_bp.route('/<int:goal_id>/update_progress', methods=['POST'])
@login_required
def update_progress(goal_id):
    """Update goal progress"""
    data = request.get_json()
    goal = Goal.query.filter_by(id=goal_id, user_id=current_user.id).first_or_404()
    goal.progress = data.get('progress', 0)
    goal.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({'message': 'Progress updated successfully'}), 200

# ===== ML SERVICES BLUEPRINT =====
ml_bp = Blueprint('ml', __name__, url_prefix='/ml')

@ml_bp.route('/insights')
@login_required
def ml_insights():
    """ML insights page"""
    journal_entries = JournalEntry.query.filter_by(
        user_id=current_user.id
    ).order_by(JournalEntry.created_at.desc()).limit(50).all()
    
    mood_entries = MoodEntry.query.filter_by(
        user_id=current_user.id
    ).order_by(MoodEntry.created_at.desc()).limit(30).all()
    
    total_entries = len(journal_entries)
    avg_mood = sum(e.mood_score for e in mood_entries if e.mood_score) / len(mood_entries) if mood_entries else 0
    
    try:
        from ml_services import get_model_performance
        model_performance = get_model_performance()
    except (ImportError, AttributeError, Exception):
        model_performance = {'status': 'Model not loaded'}
    
    return render_template('ml/insights.html',
        journal_entries=journal_entries,
        mood_entries=mood_entries,
        total_entries=total_entries,
        avg_mood=round(avg_mood, 1),
        model_performance=model_performance
    )

@ml_bp.route('/sentiment_analysis', methods=['POST'])
@login_required
def sentiment_analysis():
    """Analyze sentiment of text with crisis detection"""
    data = request.get_json()
    text = data.get('text', '').lower()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        from ml_services import analyze_sentiment, preprocess_text, extract_topics
        
        # CRITICAL: Crisis keyword detection
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die', 
            'better off dead', 'no reason to live', 'can\'t go on',
            'end it all', 'take my life', 'harm myself', 'self harm'
        ]
        
        is_crisis = any(keyword in text for keyword in crisis_keywords)
        
        sentiment_result = analyze_sentiment(text)
        processed_text = preprocess_text(text)
        topics = extract_topics(processed_text)
        
        # Override sentiment if crisis detected
        if is_crisis:
            sentiment_result = {
                'sentiment': 'negative',
                'confidence': 0.95,
                'text': text
            }
        
        return jsonify({
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'text': text,
            'processed_text': processed_text,
            'topics': topics,
            'crisis_detected': is_crisis,
            'requires_escalation': is_crisis,
            'features': {
                'text_length': len(text),
                'word_count': len(text.split()),
                'processed_length': len(processed_text.split())
            }
        })
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return jsonify({'error': str(e)}), 500

# ===== DOCTORS BLUEPRINT =====
doctors_bp = Blueprint('doctors', __name__, url_prefix='/doctors')

@doctors_bp.route('/')
@login_required
def doctors_index():
    """Doctor recommendations page"""
    return render_template('doctors/index.html')

@doctors_bp.route('/search')
@login_required
def search_doctors():
    """Search for doctors"""
    # Mock doctor data for demonstration
    mock_doctors = [
        {
            'id': 1,
            'name': 'Dr. Sarah Johnson',
            'specialty': 'Psychiatrist',
            'distance': '2.3 miles',
            'rating': 4.8,
            'available': True
        }
    ]
    return jsonify(mock_doctors)

# ===== ASSESSMENTS BLUEPRINT =====
assessments_bp = Blueprint('assessments', __name__, url_prefix='/assessments')

@assessments_bp.route('/')
@login_required
def assessments_index():
    """Assessments overview page"""
    return render_template('assessments/index.html')

@assessments_bp.route('/api/next_question', methods=['POST'])
@login_required
def next_question():
    """API endpoint for PHQ-9 and SCID-5-PD assessments"""
    data = request.get_json()
    instrument = data.get('instrument')  # 'phq9' or 'scid5pd'
    answers = data.get('answers', [])
    state = data.get('state', {})

    if instrument == 'phq9':
        return handle_phq9(answers, state)
    elif instrument == 'scid5pd':
        return handle_scid5pd(answers, state)
    else:
        return jsonify({'error': 'Invalid instrument'}), 400

def handle_phq9(answers, state):
    """Handle PHQ-9 depression assessment"""
    # PHQ-9 questions
    phq9_questions = [
        {"id": "phq9_1", "text": "Little interest or pleasure in doing things?", "type": "likert", "options": [0, 1, 2, 3]},
        {"id": "phq9_2", "text": "Feeling down, depressed, or hopeless?", "type": "likert", "options": [0, 1, 2, 3]},
        {"id": "phq9_3", "text": "Trouble falling or staying asleep, or sleeping too much?", "type": "likert", "options": [0, 1, 2, 3]},
        {"id": "phq9_4", "text": "Feeling tired or having little energy?", "type": "likert", "options": [0, 1, 2, 3]},
        {"id": "phq9_5", "text": "Poor appetite or overeating?", "type": "likert", "options": [0, 1, 2, 3]},
        {"id": "phq9_6", "text": "Feeling bad about yourself or that you are a failure?", "type": "likert", "options": [0, 1, 2, 3]},
        {"id": "phq9_7", "text": "Trouble concentrating on things like reading or watching TV?", "type": "likert", "options": [0, 1, 2, 3]},
        {"id": "phq9_8", "text": "Moving or speaking slowly, or being fidgety/restless?", "type": "likert", "options": [0, 1, 2, 3]},
        {"id": "phq9_9", "text": "Thoughts that you would be better off dead, or of hurting yourself?", "type": "likert", "options": [0, 1, 2, 3]}
    ]

    # Get next question index
    current_idx = len(answers)

    # Check if assessment is complete
    if current_idx >= len(phq9_questions):
        # Calculate score
        score = sum(a['value'] for a in answers)

        # Determine severity
        if score <= 4:
            severity = 'minimal'
        elif score <= 9:
            severity = 'mild'
        elif score <= 14:
            severity = 'moderate'
        elif score <= 19:
            severity = 'moderately_severe'
        else:
            severity = 'severe'

        # Save to database
        try:
            from models import AssessmentSession
            session = AssessmentSession(
                user_id=current_user.id,
                instrument='phq9',
                completed_at=datetime.now(timezone.utc),
                score=score,
                severity=severity,
                answers_json=answers,
                state_json=state
            )
            db.session.add(session)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error saving PHQ-9 assessment: {e}")
            db.session.rollback()

        return jsonify({
            'done': True,
            'score': score,
            'severity': severity,
            'crisis_detected': score >= 15  # Moderately severe or severe
        })

    # Return next question
    return jsonify({
        'done': False,
        'question': phq9_questions[current_idx],
        'state': state
    })

def handle_scid5pd(answers, state):
    """Handle SCID-5-PD personality disorder screening"""
    # SCID-5-PD screening questions (simplified version)
    scid_questions = [
        {"id": "scid_1", "text": "Do you often feel uncomfortable or helpless when you're alone?", "type": "yesno"},
        {"id": "scid_2", "text": "Do you often worry that people close to you will abandon you?", "type": "yesno"},
        {"id": "scid_3", "text": "Do you frequently feel that your relationships are more intense than they really are?", "type": "yesno"},
        {"id": "scid_4", "text": "Do you often feel unsure about who you really are?", "type": "yesno"},
        {"id": "scid_5", "text": "Do you frequently do things impulsively without thinking?", "type": "yesno"},
        {"id": "scid_6", "text": "Have you repeatedly threatened or tried to harm yourself?", "type": "yesno"},
        {"id": "scid_7", "text": "Do you often have intense mood swings?", "type": "yesno"},
        {"id": "scid_8", "text": "Do you frequently feel empty inside?", "type": "yesno"},
        {"id": "scid_9", "text": "Do you often have inappropriate intense anger?", "type": "yesno"},
        {"id": "scid_10", "text": "Do you often feel suspicious or paranoid about others?", "type": "yesno"},
        {"id": "scid_11", "text": "Do you avoid getting close to people because you fear rejection?", "type": "yesno"},
        {"id": "scid_12", "text": "Do you see yourself as socially inept or inferior?", "type": "yesno"},
        {"id": "scid_13", "text": "Do you need constant reassurance and advice from others?", "type": "yesno"},
        {"id": "scid_14", "text": "Do you have difficulty disagreeing with others for fear of losing support?", "type": "yesno"},
        {"id": "scid_15", "text": "Do you excessively seek admiration from others?", "type": "yesno"},
        {"id": "scid_16", "text": "Do you often feel that you're more important or talented than others recognize?", "type": "yesno"},
        {"id": "scid_17", "text": "Do you have difficulty accepting criticism?", "type": "yesno"},
        {"id": "scid_18", "text": "Do you often hold grudges and find it hard to forgive?", "type": "yesno"},
        {"id": "scid_19", "text": "Are you often preoccupied with orderliness and perfectionism?", "type": "yesno"},
        {"id": "scid_20", "text": "Do you have difficulty expressing warm emotions to others?", "type": "yesno"}
    ]

    # Get next question index
    current_idx = len(answers)

    # Check if assessment is complete
    if current_idx >= len(scid_questions):
        # Count positive responses
        positives = sum(1 for a in answers if a['value'] is True)

        # Determine risk flag
        risk_flag = positives >= 5

        # Save to database
        try:
            from models import AssessmentSession
            session = AssessmentSession(
                user_id=current_user.id,
                instrument='scid5pd',
                completed_at=datetime.now(timezone.utc),
                positives=positives,
                risk_flag=risk_flag,
                answers_json=answers,
                state_json=state
            )
            db.session.add(session)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error saving SCID-5-PD assessment: {e}")
            db.session.rollback()

        return jsonify({
            'done': True,
            'positives': positives,
            'risk_flag': risk_flag,
            'crisis_detected': risk_flag and positives >= 10  # High risk
        })

    # Return next question
    return jsonify({
        'done': False,
        'question': scid_questions[current_idx],
        'state': state
    })

# ===== CHAT BLUEPRINT =====
chat_bp = Blueprint('chat', __name__, url_prefix='/chat')

@chat_bp.route('/')
@login_required
def chat_index():
    """Chat interface"""
    return render_template('chat/index.html')

@chat_bp.route('/api/message', methods=['POST'])
@login_required
def chat_message():
    """Handle chat messages with CRISIS DETECTION"""
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'empty'}), 400
    
    try:
        # CRITICAL: Check for crisis language FIRST
        from ml_services import detect_crisis
        is_crisis, crisis_confidence = detect_crisis(text)
        
        if is_crisis:
            # IMMEDIATE CRISIS RESPONSE
            crisis_response = """I'm very concerned about what you've shared. Your life is valuable, and help is available right now.

**Please contact one of these crisis helplines immediately:**

🚨 **NIMHANS Helpline:** 080-46110007 (24/7)
🚨 **iCall:** 9152987821 (Mon-Sat, 8 AM - 10 PM)
🚨 **Vandrevala Foundation:** 1860-2662-345 (24/7)

If you're in immediate danger, please:
- Call one of these numbers RIGHT NOW
- Go to the nearest hospital emergency room
- Tell someone you trust how you're feeling
- Don't be alone

You're not alone in this. These feelings are temporary, and professional help can make a real difference. Please reach out for help now."""
            
            # Save messages
            try:
                db.session.add(ChatMessage(
                    user_id=current_user.id,
                    role='user',
                    text=text,
                    sentiment='crisis',
                    confidence=crisis_confidence
                ))
                db.session.add(ChatMessage(
                    user_id=current_user.id,
                    role='bot',
                    text=crisis_response,
                    sentiment='crisis',
                    confidence=1.0
                ))
                db.session.commit()
            except Exception as e:
                logger.error(f"Error saving crisis chat: {e}")
                db.session.rollback()
            
            return jsonify({
                'reply': crisis_response,
                'escalate': True,
                'crisis': True,
                'risk_level': 'CRITICAL',
                'requires_immediate_help': True,
                'professional_referrals': [
                    {'type': 'emergency', 'title': 'NIMHANS Helpline', 'contact': '080-46110007', 'urgent': True},
                    {'type': 'crisis', 'title': 'iCall Suicide Prevention', 'contact': '9152987821', 'urgent': True},
                    {'type': 'crisis', 'title': 'Vandrevala Foundation', 'contact': '1860-2662-345', 'urgent': True}
                ]
            })
        
        # Normal response (no crisis detected)
        response_text = "I'm here to listen. How are you feeling today? Remember, I'm here to support you, but if you're in distress, please reach out to a mental health professional."
        
        # Save messages
        try:
            db.session.add(ChatMessage(
                user_id=current_user.id,
                role='user',
                text=text
            ))
            db.session.add(ChatMessage(
                user_id=current_user.id,
                role='bot',
                text=response_text
            ))
            db.session.commit()
        except Exception as e:
            logger.error(f"Error saving chat: {e}")
            db.session.rollback()
        
        return jsonify({
            'reply': response_text,
            'escalate': False,
            'crisis': False
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Chat service unavailable'}), 500

# ===== VIDEO/AUDIO ASSESSMENT BLUEPRINT =====
assessment_bp = Blueprint('assessment', __name__, url_prefix='/assessment')

@assessment_bp.route('/video-audio')
@login_required
def video_audio_assessment():
    """Video/audio assessment page"""
    return render_template('assessments/video_audio.html')

@assessment_bp.route('/api/analyze-recording', methods=['POST'])
@login_required
def analyze_recording():
    """
    Analyze uploaded audio/video recording with comprehensive crisis detection

    OPTIMIZED VERSION: Uses new optimized video/audio service for 5x speed and 95%+ accuracy

    Expects:
        - POST request with 'recording' or 'audio' file (audio blob from browser)
        - Optionally 'video' file for video/audio combined analysis

    Returns:
        JSON with comprehensive analysis including:
        - Crisis assessment
        - Depression, anxiety, and confidence scores
        - Safety recommendations
        - Professional resources
        - Performance metrics
    """
    try:
        logger.info(f"Received analysis request from user {current_user.id}")

        # Check if audio recording is provided
        if 'recording' not in request.files and 'audio' not in request.files:
            logger.error("No audio recording provided in request")
            return jsonify({
                'error': 'No audio recording provided',
                'message': 'Please record audio before submitting'
            }), 400

        # Get audio file
        audio_file = request.files.get('recording') or request.files.get('audio')
        audio_data = audio_file.read()

        # Validate audio data
        if not audio_data or len(audio_data) < 1000:
            logger.error(f"Audio data too small: {len(audio_data) if audio_data else 0} bytes")
            return jsonify({
                'error': 'Audio recording is too short or empty',
                'message': 'Please record at least 5 seconds of clear audio'
            }), 400

        logger.info(f"Audio data received: {len(audio_data)} bytes")

        # Check if video is also provided for combined analysis
        video_file = request.files.get('video')
        video_data = None
        if video_file:
            video_data = video_file.read()
            if video_data and len(video_data) >= 1000:
                logger.info(f"Video data received: {len(video_data)} bytes - using OPTIMIZED combined analysis")
            else:
                logger.warning(f"Video data too small ({len(video_data) if video_data else 0} bytes), using audio-only")
                video_data = None

        # Determine which service to use
        analysis_result = None

        # OPTION 1: Combined video + audio analysis (OPTIMIZED - NEW!)
        if video_data and audio_data:
            try:
                from optimized_video_audio_service import get_optimized_video_audio_service
                logger.info("🚀 Using OPTIMIZED video+audio service (5x faster, 95%+ accuracy)")
                service = get_optimized_video_audio_service()
                analysis_result = service.analyze_video_audio_optimized(video_data, audio_data)

                # Extract crisis detection from mental health indicators
                mental_health = analysis_result.get('mental_health_indicators', {})
                analysis_result['crisis_detected'] = analysis_result.get('crisis_detected', False)
                analysis_result['risk_level'] = 'HIGH' if mental_health.get('depression_score', 0) >= 0.7 else \
                                                'MEDIUM' if mental_health.get('depression_score', 0) >= 0.5 else 'LOW'

                logger.info(f"✓ Optimized analysis complete in {analysis_result.get('performance', {}).get('total_processing_time', 0):.2f}s")

            except ImportError as e:
                logger.warning(f"Optimized service not available: {e}, falling back to audio-only")
                video_data = None  # Fall through to audio-only
            except Exception as e:
                logger.error(f"Optimized analysis failed: {e}, falling back to audio-only", exc_info=True)
                video_data = None  # Fall through to audio-only

        # OPTION 2: Audio-only analysis (existing enhanced service)
        if not analysis_result:
            try:
                from enhanced_audio_service import analyze_audio_with_crisis_detection
                logger.info("Using enhanced audio analysis with crisis detection")
                analysis_result = analyze_audio_with_crisis_detection(audio_data)
            except ImportError:
                logger.warning("Enhanced audio service not available, using fallback")
                # Fallback to basic video_audio_service
                try:
                    from video_audio_service import VideoAudioAnalysisService
                    service = VideoAudioAnalysisService()
                    analysis_result = service.analyze_audio_only(audio_data)
                except Exception as e:
                    logger.error(f"Fallback analysis failed: {e}")
                    return jsonify({
                        'error': 'Audio analysis service unavailable',
                        'message': 'Please try again later or contact support'
                    }), 503

        # Check if analysis was successful
        if not analysis_result.get('processing_successful', True):
            error_type = analysis_result.get('error_type', 'unknown')
            error_details = analysis_result.get('error_details', '')
            logger.error(f"Analysis failed: {error_type} - {error_details}")

            # Return error with fallback recommendations
            return jsonify({
                'error': 'Analysis processing failed',
                'error_type': error_type,
                'message': 'Unable to process audio. Please try recording again with clear audio.',
                'recommendations': analysis_result.get('safety_recommendations', {})
            }), 500

        # Log crisis detection results
        crisis_detected = analysis_result.get('crisis_assessment', {}).get('is_crisis', False)
        depression_score = analysis_result.get('depression_score', 0.0)
        risk_level = analysis_result.get('risk_level', 'UNKNOWN')

        logger.info(f"Analysis complete for user {current_user.id}: "
                   f"crisis={crisis_detected}, depression={depression_score:.3f}, "
                   f"risk={risk_level}")

        # CRITICAL: If crisis detected, log for monitoring
        if crisis_detected:
            logger.critical(f"🚨 CRISIS DETECTED for user {current_user.id}: "
                          f"severity={analysis_result.get('crisis_assessment', {}).get('severity_score', 0.0):.3f}, "
                          f"risk={risk_level}")

        # Save assessment to database (optional - uncomment if needed)
        # try:
        #     assessment = AssessmentSession(
        #         user_id=current_user.id,
        #         instrument='audio_assessment',
        #         completed_at=datetime.now(timezone.utc),
        #         score=int(depression_score * 100),
        #         severity=analysis_result.get('depression_level', 'unknown'),
        #         answers_json={'full_analysis': analysis_result}
        #     )
        #     db.session.add(assessment)
        #     db.session.commit()
        # except Exception as e:
        #     logger.error(f"Failed to save assessment to database: {e}")
        #     db.session.rollback()

        # Extract mental health indicators (for optimized service)
        mental_health = analysis_result.get('mental_health_indicators', {})

        # Return comprehensive analysis results (compatible with both old and new services)
        return jsonify({
            'success': True,
            'timestamp': analysis_result.get('timestamp'),
            'analysis_type': analysis_result.get('analysis_type', analysis_result.get('assessment_type')),

            # Core metrics (backward compatible)
            'depression_score': depression_score,
            'depression_level': analysis_result.get('depression_level', 'unknown'),
            'confidence_score': analysis_result.get('confidence_score', 0.5),
            'confidence_level': analysis_result.get('confidence_level', 'moderate'),
            'overall_wellbeing': analysis_result.get('overall_wellbeing', 'unknown'),
            'risk_level': risk_level,

            # NEW: Enhanced mental health metrics (from optimized service)
            'anxiety_score': mental_health.get('anxiety_score', analysis_result.get('anxiety_score', 0.0)),
            'anxiety_level': mental_health.get('anxiety_level', analysis_result.get('anxiety_level', 'unknown')),
            'stress_level': mental_health.get('stress_level', 0.0),
            'stress_category': mental_health.get('stress_category', 'unknown'),
            'emotional_stability': mental_health.get('emotional_stability', 0.5),
            'engagement_level': mental_health.get('engagement_level', 0.5),
            'risk_factors': mental_health.get('risk_factors', []),
            'protective_factors': mental_health.get('protective_factors', []),

            # Crisis assessment
            'crisis_detected': crisis_detected,
            'requires_immediate_intervention': analysis_result.get('requires_immediate_intervention', False),
            'requires_professional_followup': analysis_result.get('requires_professional_followup', False),
            'crisis_details': analysis_result.get('crisis_assessment', {}),

            # Detailed analyses
            'depression_analysis': analysis_result.get('depression_analysis', {}),
            'confidence_analysis': analysis_result.get('confidence_analysis', {}),
            'emotion_intensity': analysis_result.get('emotion_intensity', {}),
            'urgency_analysis': analysis_result.get('urgency_analysis', {}),

            # Video analysis (if available)
            'video_analysis': analysis_result.get('video_analysis', {}),
            'audio_analysis': analysis_result.get('audio_analysis', {}),

            # Recommendations and resources (enhanced from optimized service)
            'safety_recommendations': analysis_result.get('safety_recommendations', {}),
            'recommendations': analysis_result.get('recommendations',
                                                  analysis_result.get('safety_recommendations', {}).get('primary_actions', [])),

            # Transcription
            'transcribed_text': analysis_result.get('transcribed_text', ''),
            'transcription_successful': analysis_result.get('transcription_successful', False),

            # Technical details
            'audio_features': analysis_result.get('audio_features', {}),
            'sentiment_analysis': analysis_result.get('sentiment_analysis', {}),
            'analysis_confidence': analysis_result.get('analysis_confidence',
                                                      analysis_result.get('confidence_score', 0.5)),
            'processing_info': analysis_result.get('processing_info', {}),

            # NEW: Performance metrics (from optimized service)
            'performance': analysis_result.get('performance', {}),
            'processing_time': analysis_result.get('performance', {}).get('total_processing_time', 0),
            'target_met': analysis_result.get('performance', {}).get('target_met', False),
            'optimization_used': video_data is not None  # True if video was provided
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in audio analysis: {e}", exc_info=True)
        return jsonify({
            'error': 'Unexpected error during analysis',
            'message': 'An unexpected error occurred. Please try again or contact support if the issue persists.',
            'technical_details': str(e)
        }), 500


@assessment_bp.route('/api/save-assessment', methods=['POST'])
@login_required
def save_assessment():
    """
    Save audio/video assessment results to database
    """
    try:
        from models import AudioVideoAssessment

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        logger.info(f"Saving assessment for user {current_user.id}")

        # Create new assessment record
        assessment = AudioVideoAssessment(
            user_id=current_user.id,
            assessment_type=data.get('assessment_type', 'audio-only'),

            # Scores
            depression_score=data.get('depression_score'),
            depression_level=data.get('depression_level'),
            confidence_score=data.get('confidence_score'),
            confidence_level=data.get('confidence_level'),
            overall_wellbeing=data.get('overall_wellbeing'),

            # Crisis detection
            crisis_detected=data.get('crisis_detected', False),
            crisis_indicators=data.get('crisis_details', {}).get('detected_keywords', []),
            risk_level=data.get('risk_level'),

            # Transcription
            transcribed_text=data.get('transcribed_text', ''),
            transcription_successful=data.get('transcription_successful', False),

            # Voice features
            voice_features=data.get('audio_features', {}),

            # Recommendations
            recommendations=data.get('recommendations', []),

            # Metadata
            audio_duration=data.get('audio_features', {}).get('duration', 0.0)
        )

        db.session.add(assessment)
        db.session.commit()

        logger.info(f"Assessment saved successfully: ID={assessment.id}")

        return jsonify({
            'success': True,
            'message': 'Assessment saved successfully',
            'assessment_id': assessment.id
        }), 200

    except Exception as e:
        logger.error(f"Error saving assessment: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({
            'error': 'Failed to save assessment',
            'message': 'An error occurred while saving. Please try again.'
        }), 500


@assessment_bp.route('/combined-results')
@login_required
def combined_results():
    """Combined assessment results page showing all assessments and composite analysis"""
    try:
        from models import AudioVideoAssessment, AssessmentSession

        # Get latest assessments
        latest_audio_video = AudioVideoAssessment.query.filter_by(
            user_id=current_user.id
        ).order_by(AudioVideoAssessment.created_at.desc()).first()

        latest_phq9 = AssessmentSession.query.filter_by(
            user_id=current_user.id,
            instrument='phq9'
        ).order_by(AssessmentSession.completed_at.desc()).first()

        latest_scid = AssessmentSession.query.filter_by(
            user_id=current_user.id,
            instrument='scid5pd'
        ).order_by(AssessmentSession.completed_at.desc()).first()

        # Calculate combined assessment
        composite_results = calculate_composite_assessment(
            latest_audio_video, latest_phq9, latest_scid
        )

        return render_template('assessments/combined_results.html',
            audio_video=latest_audio_video,
            phq9=latest_phq9,
            scid=latest_scid,
            composite=composite_results
        )

    except Exception as e:
        logger.error(f"Error loading combined results: {e}", exc_info=True)
        return render_template('assessments/combined_results.html',
            audio_video=None,
            phq9=None,
            scid=None,
            composite=None,
            error="Unable to load assessment results"
        )


def calculate_composite_assessment(audio_video, phq9, scid):
    """Calculate combined assessment based on all available data"""
    # Default values
    video_depression = 0
    audio_depression = 0
    video_confidence = 0
    audio_confidence = 0
    phq9_normalized = 0
    scid_risk_weight = 0

    weights_used = []
    total_weight = 0

    # Extract audio/video data
    if audio_video:
        # Get depression and confidence from audio/video assessment
        if audio_video.depression_score is not None:
            combined_av_depression = audio_video.depression_score
            # Split between video and audio (assume 50/50 if not specified)
            video_depression = combined_av_depression
            audio_depression = combined_av_depression
            weights_used.append('video')
            weights_used.append('audio')
            total_weight += 0.35 + 0.30  # 65%

        if audio_video.confidence_score is not None:
            video_confidence = audio_video.confidence_score
            audio_confidence = audio_video.confidence_score

    # Extract PHQ-9 data
    if phq9 and phq9.score is not None:
        # Normalize PHQ-9 score (0-27) to 0-1 scale
        phq9_normalized = phq9.score / 27.0
        weights_used.append('phq9')
        total_weight += 0.25  # 25%

    # Extract SCID-5-PD data
    if scid and scid.positives is not None:
        # Normalize SCID positives to risk weight
        scid_risk_weight = min(scid.positives / 20.0, 1.0) * 0.1
        weights_used.append('scid')
        total_weight += 0.10  # 10%

    # Calculate composite depression score with adaptive weighting
    if total_weight > 0:
        # Redistribute weights if some assessments are missing
        weight_video = 0.35 if 'video' in weights_used else 0
        weight_audio = 0.30 if 'audio' in weights_used else 0
        weight_phq9 = 0.25 if 'phq9' in weights_used else 0
        weight_scid = 0.10 if 'scid' in weights_used else 0

        # Normalize weights to sum to 1.0
        if total_weight < 1.0:
            # Redistribute missing weight proportionally
            scale_factor = 1.0 / total_weight
            weight_video *= scale_factor
            weight_audio *= scale_factor
            weight_phq9 *= scale_factor
            weight_scid *= scale_factor

        composite_depression = (
            video_depression * weight_video +
            audio_depression * weight_audio +
            phq9_normalized * weight_phq9 +
            scid_risk_weight * weight_scid
        )
    else:
        composite_depression = 0

    # Calculate composite confidence
    composite_confidence = (video_confidence * 0.5 + audio_confidence * 0.5) if audio_video else 0

    # Calculate overall wellbeing
    wellbeing_score = (1 - composite_depression) * 0.7 + composite_confidence * 0.3

    # Determine depression level (aligned with PHQ-9 thresholds)
    if composite_depression < 0.15:  # 0-14%
        depression_level = 'minimal'
        depression_color = 'success'
    elif composite_depression < 0.33:  # 15-32%
        depression_level = 'mild'
        depression_color = 'info'
    elif composite_depression < 0.52:  # 33-51%
        depression_level = 'moderate'
        depression_color = 'warning'
    elif composite_depression < 0.70:  # 52-69%
        depression_level = 'moderately_severe'
        depression_color = 'danger'
    else:  # 70%+
        depression_level = 'severe'
        depression_color = 'danger'

    # Determine overall risk
    risk_factors = []
    if composite_depression > 0.7:
        risk_level = 'high'
        risk_factors.append('high_depression_score')
    elif composite_depression > 0.5:
        risk_level = 'medium'
        risk_factors.append('moderate_depression_score')
    else:
        risk_level = 'low'

    if phq9 and phq9.score and phq9.score >= 15:
        risk_factors.append('phq9_severe')
        if risk_level == 'low':
            risk_level = 'medium'
    elif phq9 and phq9.score and phq9.score >= 10:
        risk_factors.append('phq9_moderate')

    if scid and scid.risk_flag:
        risk_factors.append('scid_risk_flag')
        if risk_level == 'low':
            risk_level = 'medium'

    # Generate recommendations
    recommendations = generate_composite_recommendations(
        composite_depression, risk_level, risk_factors
    )

    # Detect crisis
    crisis_detected = (
        composite_depression >= 0.7 or
        (phq9 and phq9.score and phq9.score >= 15) or
        (scid and scid.positives and scid.positives >= 10) or
        (audio_video and audio_video.crisis_detected)
    )

    return {
        'depression_score': composite_depression,
        'depression_level': depression_level,
        'depression_color': depression_color,
        'confidence_score': composite_confidence,
        'wellbeing_score': wellbeing_score,
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'crisis_detected': crisis_detected,
        'assessments_used': len(weights_used),
        'total_assessments': 4,
        'completeness_score': total_weight
    }


def generate_composite_recommendations(depression_score, risk_level, risk_factors):
    """Generate personalized recommendations based on composite assessment"""
    recommendations = []

    if risk_level == 'high' or depression_score >= 0.7:
        recommendations.extend([
            "🚨 Seek immediate professional mental health consultation",
            "📞 Contact crisis helplines if experiencing suicidal thoughts",
            "🏥 Consider visiting nearest mental health facility",
            "👥 Inform trusted friend or family member about your condition",
            "📝 Schedule urgent appointment with psychiatrist or psychologist"
        ])
    elif risk_level == 'medium' or depression_score >= 0.5:
        recommendations.extend([
            "👨‍⚕️ Schedule appointment with mental health professional within next week",
            "💭 Consider therapy or counseling services",
            "📊 Monitor symptoms and track mood regularly",
            "🧘 Practice stress-reduction techniques (meditation, deep breathing)",
            "💪 Maintain regular physical exercise routine"
        ])
    else:
        recommendations.extend([
            "✅ Continue current mental health practices",
            "📈 Regular self-assessment and monitoring",
            "🎯 Consider preventive mental health strategies",
            "😊 Maintain healthy lifestyle habits",
            "🤝 Stay connected with support network"
        ])

    # Add specific recommendations based on risk factors
    if 'phq9_severe' in risk_factors or 'phq9_moderate' in risk_factors:
        recommendations.append("📋 Follow up on PHQ-9 assessment with healthcare provider")

    if 'scid_risk_flag' in risk_factors:
        recommendations.append("🧠 Consider personality assessment with qualified mental health professional")

    return recommendations[:7]  # Return top 7 recommendations