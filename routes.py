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
    Analyze uploaded audio recording with comprehensive crisis detection

    Expects:
        - POST request with 'recording' file (audio blob from browser)
        - Optionally 'video' file for video/audio combined analysis

    Returns:
        JSON with comprehensive analysis including:
        - Crisis assessment
        - Depression and confidence scores
        - Safety recommendations
        - Professional resources
    """
    try:
        logger.info(f"Received audio analysis request from user {current_user.id}")

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

        # Import enhanced audio service
        try:
            from enhanced_audio_service import analyze_audio_with_crisis_detection
            enhanced_service_available = True
        except ImportError:
            logger.warning("Enhanced audio service not available, using fallback")
            enhanced_service_available = False

        # Perform comprehensive audio analysis with crisis detection
        if enhanced_service_available:
            logger.info("Using enhanced audio analysis with crisis detection")
            analysis_result = analyze_audio_with_crisis_detection(audio_data)
        else:
            # Fallback to basic video_audio_service
            logger.warning("Using basic video_audio_service as fallback")
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

        # Return comprehensive analysis results
        return jsonify({
            'success': True,
            'timestamp': analysis_result.get('timestamp'),
            'analysis_type': analysis_result.get('analysis_type'),

            # Core metrics
            'depression_score': depression_score,
            'depression_level': analysis_result.get('depression_level', 'unknown'),
            'confidence_score': analysis_result.get('confidence_score', 0.5),
            'confidence_level': analysis_result.get('confidence_level', 'moderate'),
            'overall_wellbeing': analysis_result.get('overall_wellbeing', 'unknown'),
            'risk_level': risk_level,

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

            # Recommendations and resources
            'safety_recommendations': analysis_result.get('safety_recommendations', {}),
            'recommendations': analysis_result.get('safety_recommendations', {}).get('primary_actions', []),

            # Transcription
            'transcribed_text': analysis_result.get('transcribed_text', ''),
            'transcription_successful': analysis_result.get('transcription_successful', False),

            # Technical details
            'audio_features': analysis_result.get('audio_features', {}),
            'sentiment_analysis': analysis_result.get('sentiment_analysis', {}),
            'analysis_confidence': analysis_result.get('analysis_confidence', 0.5),
            'processing_info': analysis_result.get('processing_info', {})
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