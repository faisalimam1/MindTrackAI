from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, JournalEntry, MoodEntry, Task, Goal, ChatMessage
from datetime import datetime, timezone
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to convert numpy types to Python native types for JSON serialization
def convert_to_native(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

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
        phone_number = data.get('phoneNumber', '').strip()
        emergency_contact_phone = data.get('emergencyContactPhone', '').strip()
        emergency_contact_name = data.get('emergencyContactName', '').strip()
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
            user = User(
                username=username,
                email=email,
                phone_number=phone_number or None,
                emergency_contact_phone=emergency_contact_phone or None,
                emergency_contact_name=emergency_contact_name or None
            )
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
    ).order_by(MoodEntry.created_at.desc()).all()[:30]
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
    ).order_by(JournalEntry.created_at.desc()).all()[:50]

    mood_entries = MoodEntry.query.filter_by(
        user_id=current_user.id
    ).order_by(MoodEntry.created_at.desc()).all()[:30]
    
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

        # Check for critical score (≥80%) and create alert
        from critical_alert_service import check_and_create_critical_alert, send_admin_notification
        is_critical, critical_alert, comforting_message = check_and_create_critical_alert(
            user_id=current_user.id,
            assessment_type='phq9',
            score=score,
            severity=severity
        )

        # Send admin notification if critical
        if is_critical and critical_alert:
            send_admin_notification(critical_alert)

        response_data = {
            'done': True,
            'score': score,
            'severity': severity,
            'crisis_detected': score >= 15,  # Moderately severe or severe
            'critical_alert': is_critical,
            'comforting_message': comforting_message if is_critical else None
        }

        return jsonify(response_data)

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

        # Determine severity based on positives
        if positives >= 15:
            severity = 'severe'
        elif positives >= 10:
            severity = 'high'
        elif positives >= 5:
            severity = 'moderate'
        else:
            severity = 'mild'

        # Check for critical score (≥80%) and create alert
        from critical_alert_service import check_and_create_critical_alert, send_admin_notification
        is_critical, critical_alert, comforting_message = check_and_create_critical_alert(
            user_id=current_user.id,
            assessment_type='scid5pd',
            score=positives,
            severity=severity
        )

        # Send admin notification if critical
        if is_critical and critical_alert:
            send_admin_notification(critical_alert)

        response_data = {
            'done': True,
            'positives': positives,
            'risk_flag': risk_flag,
            'crisis_detected': risk_flag and positives >= 10,  # High risk
            'critical_alert': is_critical,
            'comforting_message': comforting_message if is_critical else None
        }

        return jsonify(response_data)

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
        
        # Advanced Mental Health Chatbot - Using pattern matching and context
        from mental_health_chatbot import chatbot

        # Get recent conversation history for context
        recent_messages = ChatMessage.query.filter_by(
            user_id=current_user.id
        ).order_by(ChatMessage.created_at.desc()).all()[:5]

        conversation_history = [msg.message for msg in reversed(recent_messages)]

        # Analyze the message using advanced chatbot
        analysis = chatbot.analyze_message(text, conversation_history)

        # Generate contextual response
        response_text = chatbot.generate_response(analysis)

        # Legacy fallback removed - using advanced chatbot exclusively

        # If for some reason the chatbot didn't generate a response, use default
        if not response_text:
            response_text = """I'm here to support you with your mental health journey. You can talk to me about:

• How you're feeling emotionally
• Stress, anxiety, or depression
• Sleep or relationship issues
• Coping strategies and self-care
• Finding professional help

What's on your mind today? I'm listening without judgment, and everything you share is confidential."""

        # Remove old keyword-based code below
        '''
        elif any(word in text_lower for word in ['anxious', 'worried', 'stressed', 'panic', 'nervous', 'fear']):
            response_text = """I understand you're experiencing anxiety. Anxiety can feel overwhelming, but there are ways to manage it.

Try these grounding techniques:
• Take slow, deep breaths (4 seconds in, 4 seconds hold, 4 seconds out)
• Focus on your 5 senses - what can you see, hear, touch, smell, taste?
• Progressive muscle relaxation
• Remind yourself: "This feeling will pass"

What's causing your anxiety right now? I'm here to listen without judgment."""

        elif any(word in text_lower for word in ['sleep', 'insomnia', 'tired', 'exhausted', 'fatigue']):
            response_text = """Sleep issues can really affect how we feel. Let's explore some sleep hygiene tips:

• Keep a consistent sleep schedule
• Avoid screens 1 hour before bed
• Create a calming bedtime routine
• Keep your bedroom cool and dark
• Avoid caffeine after 2 PM
• Try relaxation exercises before bed

How long have you been experiencing sleep difficulties? Poor sleep can impact mental health, so it's important to address."""

        elif any(word in text_lower for word in ['lonely', 'alone', 'isolated', 'no friends']):
            response_text = """Feeling lonely can be really painful. Please know that you're not alone in feeling this way.

Ways to combat loneliness:
• Reach out to one person today - even a text message counts
• Join online or local communities with shared interests
• Volunteer - helping others can create connections
• Practice self-connection through journaling or meditation

Remember: It's quality, not quantity of connections that matters. Would you like to talk about what's making you feel isolated?"""

        elif any(word in text_lower for word in ['angry', 'frustrated', 'mad', 'irritated', 'rage']):
            response_text = """It sounds like you're feeling angry or frustrated. These are valid emotions, and it's important to process them healthily.

Ways to manage anger:
• Take a timeout - step away from the situation
• Physical activity can help release tension
• Write down what's making you angry
• Practice the "STOP" technique: Stop, Take a breath, Observe, Proceed mindfully

What's triggering these feelings? Let's explore constructive ways to address what's bothering you."""

        elif any(word in text_lower for word in ['help', 'support', 'therapy', 'therapist', 'counseling']):
            response_text = """It's a positive step to seek help. Here are some resources:

• Use our "Find Doctors" feature to locate mental health professionals near you
• Consider online therapy platforms if in-person isn't accessible
• Many employers offer Employee Assistance Programs (EAP)
• Community mental health centers often provide affordable services

Would you like me to guide you to our doctor recommendation feature? Professional support can make a significant difference."""

        elif any(word in text_lower for word in ['thank', 'thanks', 'grateful', 'appreciate']):
            response_text = """You're very welcome! I'm glad I could provide some support. Remember:

• Your mental health matters
• Seeking help is a sign of strength
• Small steps forward are still progress
• You deserve care and compassion

Is there anything else you'd like to talk about or explore?"""

        elif any(word in text_lower for word in ['better', 'improving', 'good day', 'feeling good', 'happy']):
            response_text = """That's wonderful to hear! It's important to celebrate these positive moments.

To maintain your wellbeing:
• Keep track of what's contributing to these good feelings
• Continue any healthy habits you've developed
• Stay connected with your support system
• Remember this feeling during difficult times

What's been helping you feel better? Identifying these factors can be valuable."""

        '''
        # End of old keyword-based code (commented out)

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

        # Check if analysis failed due to silent/invalid audio
        if analysis_result.get('error'):
            error_msg = analysis_result.get('error')
            logger.error(f"Analysis error: {error_msg}")

            return jsonify({
                'error': 'Audio quality issue',
                'message': error_msg,
                'is_silent': analysis_result.get('is_silent', False),
                'troubleshooting': [
                    'Check your microphone is working properly',
                    'Make sure you speak clearly during recording',
                    'Record in a quiet environment',
                    'Ensure the recording is at least 5 seconds long',
                    'Try speaking louder and closer to the microphone'
                ]
            }), 400

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
                'troubleshooting': [
                    'Ensure FFmpeg is installed (required for audio conversion)',
                    'Check that your audio file format is supported',
                    'Try recording with a different device or browser',
                    'Contact support if the problem persists'
                ],
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
        # Convert numpy types to native Python types for JSON serialization
        response_data = {
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
        }

        # Convert all numpy types to native Python types
        return jsonify(convert_to_native(response_data)), 200

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

        # Check for critical score (≥80%) and create alert
        from critical_alert_service import check_and_create_critical_alert, send_admin_notification

        # Get depression score (0-1.0 scale)
        depression_score = data.get('depression_score', 0)

        is_critical, critical_alert, comforting_message = check_and_create_critical_alert(
            user_id=current_user.id,
            assessment_type='audio_video',
            score=depression_score,
            severity=data.get('depression_level', 'unknown')
        )

        # Send admin notification if critical
        if is_critical and critical_alert:
            send_admin_notification(critical_alert)

        return jsonify({
            'success': True,
            'message': 'Assessment saved successfully',
            'assessment_id': assessment.id,
            'critical_alert': is_critical,
            'comforting_message': comforting_message if is_critical else None
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

        # Generate key findings
        key_findings = None
        if composite_results:
            key_findings = generate_key_findings(
                latest_audio_video,
                latest_phq9,
                latest_scid,
                composite_results.get('depression_score', 0)
            )

        return render_template('assessments/combined_results.html',
            audio_video=latest_audio_video,
            phq9=latest_phq9,
            scid=latest_scid,
            composite=composite_results,
            key_findings=key_findings,
            user=current_user
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

    # Generate personalized recommendations with actual assessment data
    recommendations = generate_composite_recommendations(
        composite_depression, risk_level, risk_factors,
        phq9=phq9, scid=scid, audio_video=audio_video
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


def generate_key_findings(audio_video, phq9, scid, composite_depression):
    """Generate Key Findings highlighting positive and negative indicators from assessments"""
    positives = []
    negatives = []

    # Analyze PHQ-9 results
    if phq9 and phq9.answers_json:
        answers = phq9.answers_json if isinstance(phq9.answers_json, list) else []

        # Check for specific positive indicators
        if len(answers) > 0 and answers[0].get('value', 3) <= 1:
            positives.append("Shows interest and pleasure in activities")
        if len(answers) > 7 and answers[7].get('value', 3) == 0:
            positives.append("No restlessness or psychomotor issues reported")
        if len(answers) > 8 and answers[8].get('value', 3) == 0:
            positives.append("No thoughts of self-harm")

        # Check for negative indicators
        if len(answers) > 1 and answers[1].get('value', 0) >= 2:
            negatives.append("Frequent feelings of depression or hopelessness")
        if len(answers) > 2 and answers[2].get('value', 0) >= 2:
            negatives.append("Persistent sleep disturbances")
        if len(answers) > 3 and answers[3].get('value', 0) >= 2:
            negatives.append("Low energy and fatigue")
        if len(answers) > 4 and answers[4].get('value', 0) >= 2:
            negatives.append("Changes in appetite or eating patterns")
        if len(answers) > 6 and answers[6].get('value', 0) >= 2:
            negatives.append("Difficulty concentrating")

    # Analyze SCID-5-PD results
    if scid and scid.positives:
        if scid.positives <= 3:
            positives.append("Stable personality indicators with low risk factors")
        if scid.positives >= 10:
            negatives.append("Multiple personality disorder screening indicators detected")
        elif scid.positives >= 5:
            negatives.append("Some personality disorder risk factors present")

    # Analyze Audio/Video results
    if audio_video:
        if audio_video.confidence_score and audio_video.confidence_score >= 0.6:
            positives.append("Good confidence levels detected in speech patterns")
        elif audio_video.confidence_score and audio_video.confidence_score < 0.4:
            negatives.append("Low confidence detected in voice tone and speech")

        if audio_video.depression_score and audio_video.depression_score < 0.3:
            positives.append("Voice and facial analysis shows healthy emotional expression")
        elif audio_video.depression_score and audio_video.depression_score >= 0.6:
            negatives.append("Voice patterns suggest signs of depression")

    # Composite analysis
    if composite_depression < 0.15:
        positives.append("Overall mental health is in excellent range")
    elif composite_depression < 0.3:
        positives.append("Mental wellbeing is in healthy range")

    # Default messages if no specific findings
    if not positives:
        positives.append("Taking this assessment shows self-awareness and proactive care")
        positives.append("Seeking to understand your mental health is a positive step")

    if not negatives:
        negatives.append("No significant concerns detected in assessments")

    return {
        'positives': positives[:4],  # Limit to top 4
        'negatives': negatives[:4]   # Limit to top 4
    }


def generate_composite_recommendations(depression_score, risk_level, risk_factors, phq9=None, scid=None, audio_video=None):
    """Generate personalized, actionable recommendations based on actual user responses"""
    recommendations = []

    # Analyze PHQ-9 specific responses
    phq9_issues = []
    if phq9 and phq9.answers_json:
        answers = phq9.answers_json if isinstance(phq9.answers_json, list) else []

        # PHQ-9 Questions mapping
        phq9_questions = [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself",
            "Trouble concentrating",
            "Moving or speaking slowly or being restless",
            "Thoughts of self-harm"
        ]

        for i, answer in enumerate(answers):
            if i < len(phq9_questions) and answer.get('value', 0) >= 2:  # More than half the days or nearly every day
                phq9_issues.append({
                    'question': phq9_questions[i],
                    'severity': answer.get('value', 0),
                    'index': i
                })

    # Analyze SCID-5 responses
    scid_issues = []
    if scid and scid.answers_json:
        answers = scid.answers_json if isinstance(scid.answers_json, list) else []
        for i, answer in enumerate(answers):
            if answer.get('value') == True:  # They answered yes
                scid_issues.append(i)

    # Analyze audio/video transcription
    audio_concerns = []
    if audio_video and audio_video.transcribed_text:
        transcription_lower = audio_video.transcribed_text.lower()

        # Detect specific concerns from transcription
        if any(word in transcription_lower for word in ['suicide', 'kill myself', 'die', 'end it']):
            audio_concerns.append('suicidal_ideation')
        if any(word in transcription_lower for word in ['anxious', 'panic', 'worried', 'nervous']):
            audio_concerns.append('anxiety')
        if any(word in transcription_lower for word in ['alone', 'lonely', 'isolated', 'no friends']):
            audio_concerns.append('loneliness')
        if any(word in transcription_lower for word in ['sleep', 'insomnia', 'tired', 'exhausted']):
            audio_concerns.append('sleep_issues')
        if any(word in transcription_lower for word in ['work', 'job', 'stress', 'overwhelmed']):
            audio_concerns.append('work_stress')

    # Generate personalized recommendations based on specific user responses

    # 1. Address PHQ-9 specific issues with personalized recommendations
    for issue in phq9_issues:
        q_index = issue['index']
        severity = issue['severity']
        question = issue['question']

        if q_index == 0:  # Lost interest/pleasure
            recommendations.append(
                f"📊 **From your PHQ-9 assessment:** You indicated experiencing '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "This is called anhedonia - a core symptom of depression where you lose the ability to feel pleasure.\n\n"
                "**Understanding the problem:**\n"
                "Anhedonia happens because depression affects brain chemistry, reducing dopamine (the 'pleasure chemical'). Activities that once brought joy now feel pointless or exhausting.\n\n"
                "**Coping strategies that work:**\n\n"
                "1️⃣ **Behavioral Activation (Most Effective):**\n"
                "   • Schedule ONE small enjoyable activity daily, even if you don't feel like it\n"
                "   • Start tiny: 5-minute walk, favorite song, petting a cat, warm shower\n"
                "   • Track it: Mark calendar when you do it - seeing progress helps\n"
                "   • Don't wait to 'feel like it' - action comes first, motivation follows\n\n"
                "2️⃣ **Pleasure Retraining:**\n"
                "   • List 10 things you used to enjoy (even if they feel pointless now)\n"
                "   • Try 1 per week for just 10 minutes\n"
                "   • Notice even tiny moments of interest (a pretty sky, good smell)\n"
                "   • Keep a 'small joys' log - retrain your brain to notice pleasure\n\n"
                "3️⃣ **Social Activation:**\n"
                "   • Accept social invitations even when you don't want to (often helps afterward)\n"
                "   • Ask someone to join you in activities\n"
                "   • Social connection can spark interest when solo activities don't\n\n"
                "4️⃣ **Try New Things:**\n"
                "   • Sometimes old interests are too associated with 'before depression'\n"
                "   • Try completely new: art class, cooking new recipe, different walking route\n"
                "   • Novelty can bypass the 'nothing interests me' block\n\n"
                "5️⃣ **Professional Help:**\n"
                "   • Therapy (CBT or Behavioral Activation Therapy) is very effective\n"
                "   • Medication may help restore pleasure capacity\n"
                "   • This symptom often improves significantly with treatment"
            )

        elif q_index == 1:  # Feeling down/depressed/hopeless
            recommendations.append(
                f"📊 **From your PHQ-9 assessment:** You reported '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "This persistent low mood with hopelessness is a core depression symptom that needs attention.\n\n"
                "**Understanding the problem:**\n"
                "Persistent depressed mood happens when brain neurotransmitters (serotonin, dopamine, norepinephrine) are out of balance. The hopelessness you feel is a symptom, not reality.\n\n"
                "**Evidence-based coping strategies:**\n\n"
                "1️⃣ **Cognitive Behavioral Techniques:**\n"
                "   • **Thought Record**: Write down negative thoughts, identify distortions, create balanced alternatives\n"
                "   • Example: 'I'm worthless' → 'I'm struggling right now, but I have value' (list 3 pieces of evidence)\n"
                "   • Challenge 'always/never' thinking - look for exceptions\n"
                "   • Ask: 'What would I tell a friend thinking this?'\n\n"
                "2️⃣ **Mood Monitoring:**\n"
                "   • Track mood 3x daily (morning, afternoon, evening) on 1-10 scale\n"
                "   • Note what you were doing, who you were with, what you were thinking\n"
                "   • Identify patterns: worst times of day, triggering situations\n"
                "   • Use this data to plan better days\n\n"
                "3️⃣ **Physical Interventions:**\n"
                "   • Exercise: 30 min cardio 3-4x/week (as effective as medication for mild-moderate depression)\n"
                "   • Sunlight: 15-30 min outdoor exposure daily (boosts serotonin)\n"
                "   • Sleep regulation: Same wake time daily, even on bad days\n\n"
                "4️⃣ **Social Connection (Even When You Don't Feel Like It):**\n"
                "   • Schedule 1 social interaction daily, even brief (text, call, coffee)\n"
                "   • Isolation worsens depression - connection is medicine\n"
                "   • Tell trusted people you're struggling - hiding it makes it worse\n\n"
                "5️⃣ **Crisis Planning:**\n"
                "   • If hopelessness includes thoughts of suicide, tell someone TODAY\n"
                "   • Create safety plan: list of people to call, reasons to live, coping strategies\n"
                "   • Remove means of harm from environment\n\n"
                "6️⃣ **Professional Treatment (IMPORTANT):**\n"
                "   • This level of depression typically needs professional treatment\n"
                "   • Therapy options: CBT, IPT (Interpersonal Therapy), or MBCT (Mindfulness-Based)\n"
                "   • Medication: Antidepressants are effective for moderate-severe depression\n"
                "   • Combination (therapy + medication) often works best\n"
                "   • **Action: Schedule appointment within 1 week**"
            )

        elif q_index == 2:  # Sleep problems
            recommendations.append(
                f"📊 **From your PHQ-9 assessment:** You're experiencing '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "Sleep disruption is both a cause and effect of depression - a vicious cycle that needs breaking.\n\n"
                "**Understanding the problem:**\n"
                "Depression disrupts sleep architecture (REM/deep sleep patterns). Poor sleep then worsens mood, concentration, and emotional regulation. This creates a downward spiral.\n\n"
                "**Comprehensive Sleep Strategy (CBT-I Principles):**\n\n"
                "1️⃣ **Sleep Schedule (Most Important):**\n"
                "   • Same wake time EVERY day (even weekends, even after bad night)\n"
                "   • Only go to bed when actually sleepy (not just tired)\n"
                "   • If not asleep in 20 min, get up - return when sleepy\n"
                "   • No napping (builds sleep pressure for night)\n\n"
                "2️⃣ **Bedtime Routine (Wind-Down Protocol):**\n"
                "   • Start 60-90 min before bed\n"
                "   • Dim lights (signals melatonin production)\n"
                "   • No screens (blue light suppresses melatonin)\n"
                "   • Calming activities: reading, stretching, meditation, warm bath\n"
                "   • Write tomorrow's to-do list (clears racing thoughts)\n\n"
                "3️⃣ **Sleep Environment:**\n"
                "   • Cool temperature (65-68°F / 18-20°C)\n"
                "   • Dark (blackout curtains or eye mask)\n"
                "   • Quiet (white noise if needed)\n"
                "   • Bed for sleep/sex only (not work, TV, phone)\n\n"
                "4️⃣ **Daytime Habits:**\n"
                "   • Sunlight exposure: 15-30 min within 1 hour of waking (regulates circadian rhythm)\n"
                "   • Exercise: 30 min daily, but NOT 3 hours before bed\n"
                "   • Caffeine cutoff: None after 2 PM\n"
                "   • Avoid alcohol (disrupts sleep quality even if you fall asleep faster)\n\n"
                "5️⃣ **For Racing Thoughts:**\n"
                "   • Keep 'worry journal' by bed - write it down, deal tomorrow\n"
                "   • Breathing: 4-7-8 technique (inhale 4, hold 7, exhale 8)\n"
                "   • Body scan meditation\n"
                "   • Guided sleep meditation apps\n\n"
                "6️⃣ **When to Get Professional Help:**\n"
                "   • If insomnia persists 3+ weeks despite trying these strategies\n"
                "   • If sleepiness interferes with daily functioning\n"
                "   • CBT-I (Cognitive Behavioral Therapy for Insomnia) - 80-90% effective\n"
                "   • Sleep study if suspect sleep apnea (snoring, gasping, still tired after full sleep)\n"
                "   • Medication short-term while building good habits"
            )

        elif q_index == 3:  # Fatigue/low energy
            recommendations.append(
                f"📊 **From your PHQ-9 assessment:** You indicated '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "Fatigue is one of the most challenging depression symptoms.\n\n"
                "**Recommended actions:**\n"
                "• Gentle movement - even a 10-minute walk can boost energy\n"
                "• Check for medical causes - vitamin deficiencies, thyroid issues\n"
                "• Break tasks into tiny steps - accomplishing small things builds momentum\n"
                "• Prioritize rest without guilt - depression is exhausting"
            )

        elif q_index == 4:  # Appetite changes
            recommendations.append(
                f"📊 **From your PHQ-9 assessment:** You're dealing with '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "Appetite changes are common in depression.\n\n"
                "**Recommended actions:**\n"
                "• Keep easy, nutritious snacks available\n"
                "• Set meal reminders if forgetting to eat\n"
                "• Eat with others when possible - social eating helps\n"
                "• If significant weight change, consult a healthcare provider"
            )

        elif q_index == 5:  # Negative self-perception
            recommendations.append(
                f"📊 **From your PHQ-9 assessment:** You reported '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "These self-critical thoughts are depression symptoms, not facts.\n\n"
                "**Recommended actions:**\n"
                "• Practice self-compassion - talk to yourself like a good friend would\n"
                "• Challenge negative thoughts: 'What evidence supports/contradicts this?'\n"
                "• Keep a 'evidence log' of your positive qualities and achievements\n"
                "• Therapy (especially CBT) is very effective for changing thought patterns"
            )

        elif q_index == 6:  # Concentration problems
            recommendations.append(
                f"📊 **From your PHQ-9 assessment:** You're experiencing '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "Concentration difficulties are cognitive symptoms of depression.\n\n"
                "**Recommended actions:**\n"
                "• Use external memory aids - lists, reminders, notes\n"
                "• Break work into smaller chunks with breaks\n"
                "• Reduce multitasking - focus on one thing at a time\n"
                "• This typically improves with depression treatment"
            )

        elif q_index == 7:  # Psychomotor changes
            recommendations.append(
                f"📊 **From your PHQ-9 assessment:** You noted '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "These physical manifestations indicate depression severity.\n\n"
                "**Recommended actions:**\n"
                "• Seek professional evaluation soon - these are significant symptoms\n"
                "• Gentle movement or relaxation exercises may help\n"
                "• Medication may be particularly helpful for these symptoms"
            )

        elif q_index == 8:  # Self-harm thoughts
            recommendations.append(
                f"🚨 **CRITICAL - From your PHQ-9 assessment:** You indicated experiencing '{question.lower()}' {'nearly every day' if severity == 3 else 'more than half the days'}. "
                "This requires immediate professional attention.\n\n"
                "**IMMEDIATE ACTIONS:**\n"
                "• Call crisis helpline NOW: NIMHANS 080-46110007 or iCall 9152987821\n"
                "• See a mental health professional THIS WEEK\n"
                "• Tell someone you trust immediately\n"
                "• Create a safety plan - remove means of harm\n"
                "• Use our 'Find Doctors' feature to locate professionals near you"
            )

    # 2. Address Audio/Video specific concerns
    if 'suicidal_ideation' in audio_concerns:
        recommendations.append(
            "🚨 **CRITICAL - From your audio/video assessment:** Your recording indicated thoughts of suicide or self-harm. "
            "This is a mental health emergency.\n\n"
            "**IMMEDIATE ACTIONS:**\n"
            "• Contact crisis support NOW: NIMHANS 080-46110007 (24/7)\n"
            "• DO NOT be alone - call someone immediately\n"
            "• Go to nearest emergency room if feelings intensify\n"
            "• Your life has value - these feelings can be treated"
        )

    if 'anxiety' in audio_concerns:
        recommendations.append(
            "🎙️ **From your audio/video assessment:** You mentioned feeling anxious, worried, or nervous. "
            "Your voice patterns also suggest elevated stress levels.\n\n"
            "**Recommended actions:**\n"
            "• Practice grounding techniques: 5-4-3-2-1 method (5 things you see, 4 you touch, etc.)\n"
            "• Deep breathing: 4-4-4-4 box breathing\n"
            "• Limit caffeine and ensure adequate sleep\n"
            "• Consider therapy (CBT is very effective for anxiety)\n"
            "• If panic attacks are frequent, see a mental health professional"
        )

    if 'loneliness' in audio_concerns:
        recommendations.append(
            "🎙️ **From your audio/video assessment:** You expressed feelings of loneliness or isolation. "
            "Social connection is crucial for mental health.\n\n"
            "**Recommended actions:**\n"
            "• Reach out to one person today - even a brief text counts\n"
            "• Join communities around your interests (online or in-person)\n"
            "• Volunteer - helping others creates connections\n"
            "• Consider group therapy or support groups\n"
            "• Quality over quantity - one good connection makes a difference"
        )

    if 'sleep_issues' in audio_concerns:
        recommendations.append(
            "🎙️ **From your audio/video assessment:** You mentioned sleep difficulties. "
            "Poor sleep significantly impacts mental health.\n\n"
            "**Recommended actions:**\n"
            "• Maintain consistent sleep schedule (same time every day)\n"
            "• Create relaxing bedtime routine\n"
            "• Avoid screens 1 hour before bed\n"
            "• If insomnia lasts 2+ weeks, see a healthcare provider\n"
            "• CBT for Insomnia (CBT-I) is the gold standard treatment"
        )

    if 'work_stress' in audio_concerns:
        recommendations.append(
            "🎙️ **From your audio/video assessment:** You mentioned work-related stress or feeling overwhelmed. "
            "Chronic work stress can lead to burnout.\n\n"
            "**Recommended actions:**\n"
            "• Set clear work-life boundaries\n"
            "• Take regular breaks during workday\n"
            "• Practice saying 'no' to unreasonable demands\n"
            "• Consider discussing workload with supervisor\n"
            "• If burnout symptoms present, professional help recommended"
        )

    # 3. Address SCID-5 concerns if significant
    if len(scid_issues) >= 10:
        recommendations.append(
            f"📋 **From your SCID-5-PD assessment:** You answered 'yes' to {len(scid_issues)} questions, "
            "suggesting possible personality-related patterns that may benefit from professional evaluation.\n\n"
            "**Recommended actions:**\n"
            "• Consult with a mental health professional for comprehensive personality assessment\n"
            "• Dialectical Behavior Therapy (DBT) or Schema Therapy can be very helpful\n"
            "• Understanding your patterns is the first step to managing them\n"
            "• Many personality-related challenges are highly treatable"
        )
    elif len(scid_issues) >= 5:
        recommendations.append(
            f"📋 **From your SCID-5-PD assessment:** You answered 'yes' to {len(scid_issues)} questions. "
            "While this doesn't indicate a disorder, it suggests some patterns worth exploring.\n\n"
            "**Recommended actions:**\n"
            "• Consider therapy to explore these patterns\n"
            "• Self-awareness is valuable - journaling can help identify triggers\n"
            "• Professional guidance can provide coping strategies"
        )

    # If no specific issues found, add general wellness recommendations
    if not recommendations:
        if depression_score < 0.15:  # Good mental health
            recommendations.append(
                "✅ **Overall Assessment:** Your mental health appears to be in a healthy range. "
                "Continue maintaining your positive habits.\n\n"
                "**Recommended actions:**\n"
                "• Continue regular self-care practices\n"
                "• Maintain social connections\n"
                "• Keep up with physical activity and healthy sleep\n"
                "• Use our mood tracker to monitor ongoing wellbeing"
            )
        else:
            recommendations.append(
                "💡 **General Wellness Recommendations:**\n\n"
                "• Monitor your mental health regularly using our assessment tools\n"
                "• Maintain healthy sleep schedule (7-9 hours)\n"
                "• Stay physically active (30 minutes, 3-4 times per week)\n"
                "• Stay connected with supportive people\n"
                "• Practice stress management techniques\n"
                "• Consider professional consultation if symptoms worsen"
            )

    # Return all personalized recommendations (they're already specific to user's responses)
    return recommendations


# ===== DOCTOR RECOMMENDATIONS BLUEPRINT =====
doctors_bp = Blueprint('doctors', __name__, url_prefix='/doctors')

@doctors_bp.route('/')
@login_required
def index():
    """Show doctor recommendations page"""
    return render_template('doctors/index.html')

@doctors_bp.route('/search')
@login_required
def search_doctors():
    """Search for mental health professionals based on criteria"""
    specialty = request.args.get('specialty', '')
    location = request.args.get('location', '')

    # Mental health professionals data for Bangalore
    # In production, this would integrate with real APIs or database
    doctors_list = [
        {
            'id': 1,
            'name': 'Dr. Kapur B, MD',
            'specialty': 'Psychiatrist',
            'subspecialty': 'Depression & Schizophrenia',
            'distance': 'Hebbal',
            'rating': 4.8,
            'review_count': 127,
            'address': 'Hebbal, Manipal Hospital, Bangalore, Karnataka 560036',
            'phone': '8046808476',
            'email': 'NA',
            'available': True,
            'next_available': 'Visit Website',
            'accepts_insurance': True,
            'languages': ['English', 'Hindi', 'Punjabi'],
            'years_experience': 47,
            'education': 'AFMC, PUNE',
            'certifications': ['Board Certified in Psychiatry', 'Fellow of American Psychiatric Association']
        },
        {
            'id': 2,
            'name': 'Dr. Krishen Ranganath',
            'specialty': 'Psychiatrist',
            'subspecialty': 'Autism, Dyslexia, Eating Disorders, Mood Disorders, PTSD',
            'distance': 'Seshadripuram / Basaveshwara Nagar',
            'rating': 4.9,
            'review_count': 167,
            'address': 'Apollo Hospitals Sheshadripuram & BINDIG MINDCARE, Bangalore',
            'phone': '+91 80 4668 8888',
            'email': 'NA',
            'available': True,
            'next_available': 'Book via Practo / Clinic inquiry',
            'accepts_insurance': False,
            'languages': ['English', 'Hindi', 'Kannada', 'Tamil', 'Telugu'],
            'years_experience': 18,
            'education': 'MBBS, MRCPsych (UK), Diploma in Clinical Psychiatry (Ireland), PG Dip Clinical Neuropsychiatry (Birmingham, UK)',
            'certifications': [
                'Medical Registration Verified',
                'Certificate (Part 1) in Clinical Psychopharmacology – BAP',
                'Internship in Medical Leadership (UK)'
            ]
        },
        {
            'id': 3,
            'name': 'Dr. Bhupendra Chaudhry',
            'specialty': 'Psychiatrist',
            'subspecialty': 'Depression, Anxiety Disorders, OCD, Schizophrenia, Addiction, Psychiatric Emergencies, Child & Adolescent Issues',
            'distance': 'Koramangala / Old Airport Road / Kumara Park West',
            'rating': 4.6,
            'review_count': 124,
            'address': 'Apollo Medical Centre; Manipal Hospital — Old Airport Road; Mallige Medical Centre, Bangalore',
            'phone': '18001024647',
            'email': 'NA',
            'available': True,
            'next_available': 'Book via Practo or Apollo platform',
            'accepts_insurance': False,
            'languages': ['English', 'Hindi', 'Kannada'],
            'years_experience': 33,
            'education': 'MBBS (Kanpur University), MD Psychiatry (SNMC, Agra)',
            'certifications': [
                'Karnataka Medical Council Reg 79231',
                'Member of Indian Psychiatric Society'
            ]
        },
        {
            'id': 4,
            'name': 'Dr. Chandra Shekar M',
            'specialty': 'Psychiatrist',
            'subspecialty': 'General Psychiatry, Child Psychiatry, De-addiction',
            'distance': 'RT Nagar / Horamavu',
            'rating': 4.5,
            'review_count': 18,
            'address': 'Medax Hospitals (RT Nagar); Trust-In Hospital (Horamavu); Sridi Sai Hospital — various clinics in Bangalore',
            'phone': 'On-call via Practo/clinic inquiry',
            'email': 'NA',
            'available': True,
            'next_available': 'Book via Practo or hospital portal',
            'accepts_insurance': False,
            'languages': ['English', 'Hindi'],
            'years_experience': 31,
            'education': 'MBBS; DPM Psychiatry (NIMHANS); DNB Psychiatry (NIMHANS)',
            'certifications': [
                'Karnataka Medical Council Reg 39712',
                'Indian Psychiatric Society',
                'Karnataka Psychiatric Society'
            ]
        }
    ]

    # Filter by specialty if provided
    if specialty:
        doctors_list = [d for d in doctors_list if specialty.lower() in d['specialty'].lower()]

    # Filter by location if provided
    if location:
        doctors_list = [d for d in doctors_list if location.lower() in d['distance'].lower()]

    return jsonify(doctors_list)

@doctors_bp.route('/<int:doctor_id>')
@login_required
def doctor_detail(doctor_id):
    """Show detailed information about a specific doctor"""
    # In production, this would fetch from a database or API
    return render_template('doctors/detail.html', doctor_id=doctor_id)

@doctors_bp.route('/appointment', methods=['POST'])
@login_required
def book_appointment():
    """Book an appointment with a doctor"""
    data = request.get_json()
    doctor_id = data.get('doctor_id')
    appointment_date = data.get('appointment_date')
    appointment_time = data.get('appointment_time')
    reason = data.get('reason')

    # In production, this would:
    # 1. Validate the appointment slot
    # 2. Create appointment record in database
    # 3. Send confirmation email/SMS
    # 4. Integrate with doctor's calendar system

    return jsonify({
        'success': True,
        'message': 'Appointment request submitted successfully',
        'appointment': {
            'doctor_id': doctor_id,
            'date': appointment_date,
            'time': appointment_time,
            'status': 'pending'
        }
    })


# ===== ADMIN BLUEPRINT =====
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


def check_admin_access():
    """Check if current user has admin access"""
    import os
    if not current_user.is_authenticated:
        return False

    # Check if user is admin (either by username or by admin flag)
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')
    return current_user.username == admin_username


@admin_bp.route('/')
@login_required
def admin_dashboard():
    """Admin dashboard showing critical alerts"""
    if not check_admin_access():
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))

    try:
        from critical_alert_service import get_all_critical_alerts, get_critical_alerts_count

        # Get filter parameters
        show_resolved = request.args.get('show_resolved', 'false').lower() == 'true'

        # Get all critical alerts
        alerts = get_all_critical_alerts(include_resolved=show_resolved)
        unviewed_count = get_critical_alerts_count()

        # Statistics
        total_alerts = len(alerts)
        resolved_count = sum(1 for a in alerts if a.resolved)
        unresolved_count = total_alerts - resolved_count

        return render_template('admin/dashboard.html',
            alerts=alerts,
            unviewed_count=unviewed_count,
            total_alerts=total_alerts,
            resolved_count=resolved_count,
            unresolved_count=unresolved_count,
            show_resolved=show_resolved
        )

    except Exception as e:
        logger.error(f"Error loading admin dashboard: {e}", exc_info=True)
        flash('Error loading admin dashboard', 'error')
        return redirect(url_for('dashboard'))


@admin_bp.route('/api/alerts')
@login_required
def get_alerts_api():
    """API endpoint to get critical alerts (for real-time updates)"""
    if not check_admin_access():
        return jsonify({'error': 'Access denied'}), 403

    try:
        from critical_alert_service import get_unviewed_critical_alerts, get_critical_alerts_count

        # Get unviewed alerts
        alerts = get_unviewed_critical_alerts()
        count = get_critical_alerts_count()

        # Convert to JSON
        alerts_data = []
        for alert in alerts:
            alerts_data.append({
                'id': alert.id,
                'username': alert.username,
                'phone_number': alert.phone_number or 'N/A',
                'email': alert.email or 'N/A',
                'assessment_type': alert.assessment_type,
                'score': alert.score,
                'raw_score': alert.raw_score,
                'severity': alert.severity,
                'created_at': alert.created_at.isoformat() if alert.created_at else None,
                'alert_sent': alert.alert_sent
            })

        return jsonify({
            'success': True,
            'alerts': alerts_data,
            'count': count
        })

    except Exception as e:
        logger.error(f"Error fetching alerts API: {e}")
        return jsonify({'error': 'Failed to fetch alerts'}), 500


@admin_bp.route('/api/alert/<int:alert_id>/view', methods=['POST'])
@login_required
def mark_alert_viewed_api(alert_id):
    """Mark an alert as viewed"""
    if not check_admin_access():
        return jsonify({'error': 'Access denied'}), 403

    try:
        from critical_alert_service import mark_alert_viewed

        data = request.get_json() or {}
        notes = data.get('notes')

        success = mark_alert_viewed(alert_id, admin_notes=notes)

        if success:
            return jsonify({'success': True, 'message': 'Alert marked as viewed'})
        else:
            return jsonify({'error': 'Alert not found'}), 404

    except Exception as e:
        logger.error(f"Error marking alert as viewed: {e}")
        return jsonify({'error': 'Failed to update alert'}), 500


@admin_bp.route('/api/alert/<int:alert_id>/resolve', methods=['POST'])
@login_required
def mark_alert_resolved_api(alert_id):
    """Mark an alert as resolved"""
    if not check_admin_access():
        return jsonify({'error': 'Access denied'}), 403

    try:
        from critical_alert_service import mark_alert_resolved

        data = request.get_json() or {}
        notes = data.get('notes')

        success = mark_alert_resolved(alert_id, admin_notes=notes)

        if success:
            return jsonify({'success': True, 'message': 'Alert marked as resolved'})
        else:
            return jsonify({'error': 'Alert not found'}), 404

    except Exception as e:
        logger.error(f"Error marking alert as resolved: {e}")
        return jsonify({'error': 'Failed to update alert'}), 500