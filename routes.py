from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token, create_refresh_token
from models import db, User, JournalEntry, MoodEntry, Task, Goal, AssessmentSession, ChatMessage
from openai_service import OpenAIService, init_openai
from video_audio_service import VideoAudioAnalysisService
from video_audio_service_high_accuracy import analyze_video_audio_high_accuracy, analyze_audio_high_accuracy, analyze_video_high_accuracy, get_high_accuracy_model_info
import os
from datetime import datetime, timezone
import random
from typing import Dict, Any, List
import json
import os
from werkzeug.utils import secure_filename
# Composite assessment computation combining PHQ-9, SCID-5-PD, and AV analysis
def _compute_composite_assessment(av_results: Dict[str, Any], phq9_session, scid_session) -> Dict[str, Any]:
    """
    Comprehensive assessment combining all four assessment types:
    - Audio analysis (voice patterns, speech sentiment)
    - Video analysis (facial expressions, emotions)
    - PHQ-9 (Patient Health Questionnaire-9)
    - SCID-5-PD (Structured Clinical Interview for DSM-5 Personality Disorders)
    """
    try:
        # Extract AV metrics with better error handling
        av_dep = float(av_results.get('depression_score', 0.5))
        av_conf = float(av_results.get('confidence_score', 0.5))
        
        # Extract detailed AV metrics for better weighting
        video_dep = float(av_results.get('video_analysis', {}).get('depression_indicators', {}).get('score', av_dep))
        audio_dep = float(av_results.get('audio_analysis', {}).get('depression_indicators', {}).get('score', av_dep))
        video_conf = float(av_results.get('video_analysis', {}).get('confidence_indicators', {}).get('score', av_conf))
        audio_conf = float(av_results.get('audio_analysis', {}).get('confidence_indicators', {}).get('score', av_conf))

        # Extract PHQ-9 data
        phq9_score = None
        phq9_severity = None
        phq9_risk_level = None
        if phq9_session and getattr(phq9_session, 'score', None) is not None:
            phq9_score = float(getattr(phq9_session, 'score', 0) or 0)
            phq9_severity = getattr(phq9_session, 'severity', None)
            # Determine PHQ-9 risk level
            if phq9_score >= 20:
                phq9_risk_level = 'severe'
            elif phq9_score >= 15:
                phq9_risk_level = 'moderately_severe'
            elif phq9_score >= 10:
                phq9_risk_level = 'moderate'
            elif phq9_score >= 5:
                phq9_risk_level = 'mild'
            else:
                phq9_risk_level = 'minimal'

        # Extract SCID-5-PD data
        scid_positive = None
        scid_risk = None
        scid_risk_level = None
        if scid_session:
            scid_positive = int(getattr(scid_session, 'positives', 0) or 0)
            scid_risk = bool(getattr(scid_session, 'risk_flag', False))
            # Determine SCID-5 risk level
            if scid_positive >= 15:
                scid_risk_level = 'high'
            elif scid_positive >= 10:
                scid_risk_level = 'moderate'
            elif scid_positive >= 5:
                scid_risk_level = 'low'
            else:
                scid_risk_level = 'minimal'

        # Normalize PHQ-9 to 0-1 scale
        phq9_norm = None
        if phq9_score is not None:
            phq9_norm = min(max(phq9_score / 27.0, 0.0), 1.0)

        # Calculate SCID-5 risk weight
        scid_risk_weight = 0.0
        if scid_positive is not None:
            if scid_positive >= 15:  # High risk
                scid_risk_weight = 0.25
            elif scid_positive >= 10:  # Medium-high risk
                scid_risk_weight = 0.15
            elif scid_positive >= 5:  # Medium risk
                scid_risk_weight = 0.08
            elif scid_positive >= 2:  # Low risk
                scid_risk_weight = 0.03
        
        if scid_risk:
            scid_risk_weight = max(scid_risk_weight, 0.15)  # Ensure minimum risk if flagged

        # === COMPREHENSIVE DEPRESSION CALCULATION ===
        # Weighted combination based on assessment availability and reliability
        dep_components = []
        dep_weights = []
        dep_sources = []
        
        # Video depression (35% weight - most reliable for real-time assessment)
        dep_components.append(video_dep)
        dep_weights.append(0.35)
        dep_sources.append('video_emotions')
        
        # Audio depression (30% weight - good for voice patterns)
        dep_components.append(audio_dep)
        dep_weights.append(0.30)
        dep_sources.append('audio_voice')
        
        # PHQ-9 depression (25% weight if available - clinical standard)
        if phq9_norm is not None:
            dep_components.append(phq9_norm)
            dep_weights.append(0.25)
            dep_sources.append('phq9_questionnaire')
        else:
            # Redistribute weight if PHQ-9 not available
            dep_weights[0] += 0.125  # Video gets 47.5%
            dep_weights[1] += 0.125  # Audio gets 42.5%
        
        # Calculate weighted average
        if dep_components:
            base_depression = sum(comp * weight for comp, weight in zip(dep_components, dep_weights)) / sum(dep_weights)
        else:
            base_depression = av_dep
        
        # Add SCID-5 risk boost (10% of total weight)
        scid_boost = scid_risk_weight * 0.1
        composite_dep = min(base_depression + scid_boost, 1.0)

        # === COMPREHENSIVE CONFIDENCE CALCULATION ===
        conf_components = []
        conf_weights = []
        conf_sources = []
        
        # Video confidence (45% weight)
        conf_components.append(video_conf)
        conf_weights.append(0.45)
        conf_sources.append('video_confidence')
        
        # Audio confidence (45% weight)
        conf_components.append(audio_conf)
        conf_weights.append(0.45)
        conf_sources.append('audio_confidence')
        
        # PHQ-9 confidence boost (10% weight if available)
        if phq9_norm is not None:
            phq9_confidence_boost = (1 - phq9_norm) * 0.1  # Lower depression = higher confidence
            conf_components.append(phq9_confidence_boost)
            conf_weights.append(0.10)
            conf_sources.append('phq9_confidence')
        else:
            # Redistribute weight
            conf_weights[0] += 0.05  # Video gets 50%
            conf_weights[1] += 0.05  # Audio gets 50%
        
        # Calculate weighted average
        if conf_components:
            composite_conf = sum(comp * weight for comp, weight in zip(conf_components, conf_weights)) / sum(conf_weights)
        else:
            composite_conf = av_conf

        # === COMPREHENSIVE WELLBEING CALCULATION ===
        # Wellbeing = (1 - depression) * 0.7 + confidence * 0.3
        # This emphasizes mental health over confidence
        wellbeing = (1 - composite_dep) * 0.7 + composite_conf * 0.3
        
        # Enhanced wellbeing categorization
        if wellbeing > 0.8:
            wellbeing_label = 'excellent'
            wellbeing_color = 'green'
        elif wellbeing > 0.65:
            wellbeing_label = 'good'
            wellbeing_color = 'light-green'
        elif wellbeing > 0.5:
            wellbeing_label = 'moderate'
            wellbeing_color = 'yellow'
        elif wellbeing > 0.35:
            wellbeing_label = 'concerning'
            wellbeing_color = 'orange'
        elif wellbeing > 0.2:
            wellbeing_label = 'serious'
            wellbeing_color = 'red'
        else:
            wellbeing_label = 'critical'
            wellbeing_color = 'dark-red'

        # Enhanced depression level categorization
        if composite_dep < 0.15:
            dep_label = 'minimal'
            dep_color = 'green'
            dep_urgency = 'low'
        elif composite_dep < 0.35:
            dep_label = 'mild'
            dep_color = 'light-green'
            dep_urgency = 'low'
        elif composite_dep < 0.55:
            dep_label = 'moderate'
            dep_color = 'yellow'
            dep_urgency = 'medium'
        elif composite_dep < 0.75:
            dep_label = 'severe'
            dep_color = 'orange'
            dep_urgency = 'high'
        else:
            dep_label = 'critical'
            dep_color = 'red'
            dep_urgency = 'urgent'

        # Enhanced confidence level categorization
        if composite_conf > 0.8:
            conf_label = 'very_high'
            conf_color = 'green'
        elif composite_conf > 0.65:
            conf_label = 'high'
            conf_color = 'light-green'
        elif composite_conf > 0.5:
            conf_label = 'moderate'
            conf_color = 'yellow'
        elif composite_conf > 0.35:
            conf_label = 'low'
            conf_color = 'orange'
        else:
            conf_label = 'very_low'
            conf_color = 'red'

        # === RISK ASSESSMENT ===
        overall_risk = 'low'
        risk_factors = []
        
        if composite_dep > 0.7:
            overall_risk = 'high'
            risk_factors.append('high_depression_score')
        elif composite_dep > 0.5:
            overall_risk = 'medium'
            risk_factors.append('moderate_depression_score')
        
        if phq9_score and phq9_score >= 15:
            overall_risk = 'high'
            risk_factors.append('phq9_severe')
        elif phq9_score and phq9_score >= 10:
            if overall_risk == 'low':
                overall_risk = 'medium'
            risk_factors.append('phq9_moderate')
        
        if scid_positive and scid_positive >= 10:
            overall_risk = 'high'
            risk_factors.append('scid_high_positive')
        elif scid_positive and scid_positive >= 5:
            if overall_risk == 'low':
                overall_risk = 'medium'
            risk_factors.append('scid_moderate_positive')
        
        if scid_risk:
            overall_risk = 'high'
            risk_factors.append('scid_risk_flag')

        # === RECOMMENDATIONS ===
        recommendations = []
        
        if overall_risk == 'high' or composite_dep > 0.7:
            recommendations.extend([
                "Immediate professional mental health consultation recommended",
                "Consider crisis intervention services if experiencing suicidal thoughts",
                "Regular monitoring and follow-up assessments needed"
            ])
        elif overall_risk == 'medium' or composite_dep > 0.5:
            recommendations.extend([
                "Schedule appointment with mental health professional",
                "Consider therapy or counseling services",
                "Monitor symptoms and track mood regularly"
            ])
        else:
            recommendations.extend([
                "Continue current mental health practices",
                "Regular self-assessment and monitoring",
                "Consider preventive mental health strategies"
            ])

        return {
            # Primary Results
            'depression_score': round(composite_dep, 3),
            'depression_level': dep_label,
            'depression_color': dep_color,
            'depression_urgency': dep_urgency,
            'confidence_score': round(composite_conf, 3),
            'confidence_level': conf_label,
            'confidence_color': conf_color,
            'overall_wellbeing': wellbeing_label,
            'wellbeing_color': wellbeing_color,
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            
            # Assessment Sources
            'inputs_used': {
                'video_analysis': True,
                'audio_analysis': True,
                'phq9_used': phq9_norm is not None,
                'scid5pd_used': scid_session is not None,
                'total_assessments': sum([True, True, phq9_norm is not None, scid_session is not None])
            },
            
            # Component Details
            'component_scores': {
                'video_depression': round(video_dep, 3),
                'audio_depression': round(audio_dep, 3),
                'video_confidence': round(video_conf, 3),
                'audio_confidence': round(audio_conf, 3),
                'phq9_normalized': round(phq9_norm, 3) if phq9_norm else None,
                'scid_risk_weight': round(scid_risk_weight, 3)
            },
            
            # PHQ-9 Details
            'phq9': {
                'score': phq9_score,
                'severity': phq9_severity,
                'risk_level': phq9_risk_level,
                'normalized_score': phq9_norm
            } if phq9_score is not None else None,
            
            # SCID-5-PD Details
            'scid5pd': {
                'positives': scid_positive,
                'risk_flag': scid_risk,
                'risk_level': scid_risk_level,
                'risk_weight': round(scid_risk_weight, 3)
            } if scid_session else None,
            
            # Weighting Information
            'weighting': {
                'video_weight': round(dep_weights[0], 3),
                'audio_weight': round(dep_weights[1], 3),
                'phq9_weight': round(dep_weights[2], 3) if len(dep_weights) > 2 else 0,
                'scid_weight': round(scid_boost, 3)
            },
            
            # Metadata
            'assessment_timestamp': datetime.now().isoformat(),
            'assessment_version': '2.0',
            'completeness_score': round(sum([True, True, phq9_norm is not None, scid_session is not None]) / 4, 3)
        }
        
    except Exception as e:
        print(f"Composite assessment error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'depression_score': av_results.get('depression_score', 0.5),
            'depression_level': av_results.get('depression_level', 'moderate'),
            'confidence_score': av_results.get('confidence_score', 0.5),
            'confidence_level': av_results.get('confidence_level', 'moderate'),
            'overall_wellbeing': av_results.get('overall_wellbeing', 'moderate'),
            'overall_risk': 'unknown',
            'inputs_used': {'av_analysis': True, 'phq9_used': False, 'scid_risk': False},
            'error': str(e),
            'assessment_timestamp': datetime.now().isoformat()
        }

# Authentication Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            flash('All fields are required', 'error')
            return redirect(url_for('auth.register'))
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('auth.register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('auth.register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'error')
            return redirect(url_for('auth.register'))
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        # Handle both form data and JSON data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
            
        username = data.get('username')
        password = data.get('password')
        
        if not all([username, password]):
            if request.is_json:
                return jsonify({'error': 'Username and password are required'}), 400
            else:
                flash('Username and password are required', 'error')
                return redirect(url_for('auth.login'))
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            if request.is_json:
                return jsonify({'message': 'Login successful!', 'redirect': url_for('dashboard')})
            else:
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
        else:
            if request.is_json:
                return jsonify({'error': 'Invalid username or password'}), 401
            else:
                flash('Invalid username or password', 'error')
                return redirect(url_for('auth.login'))
    
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# JWT Authentication endpoints
@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    """API login endpoint returning JWT tokens"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not all([username, password]):
        return jsonify({'error': 'Username and password required'}), 400
    
    user = User.query.filter_by(username=username).first()
    
    if user and user.check_password(password):
        
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)
        
        return jsonify({
            'access_token': access_token,
            'refresh_token': refresh_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@auth_bp.route('/api/refresh', methods=['POST'])
@jwt_required(refresh=True)
def api_refresh():
    """Refresh JWT access token"""
    current_user_id = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user_id)
    
    return jsonify({'access_token': new_access_token})

@auth_bp.route('/api/profile')
@jwt_required()
def api_profile():
    """Get user profile using JWT"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.isoformat()
    })

# Journal Blueprint
journal_bp = Blueprint('journal', __name__, url_prefix='/journal')

@journal_bp.route('/')
@login_required
def index():
    entries = JournalEntry.query.filter_by(user_id=current_user.id).order_by(JournalEntry.created_at.desc()).all()
    return render_template('journal/index.html', entries=entries)

@journal_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_entry():
    if request.method == 'POST':
        data = request.get_json()
        entry = JournalEntry(
            user_id=current_user.id,
            title=data.get('title'),
            content=data.get('content'),
            mood_score=data.get('mood_score'),
            tags=json.dumps(data.get('tags', []))
        )
        db.session.add(entry)
        db.session.commit()
        
        return jsonify({'message': 'Entry created successfully', 'id': entry.id}), 201
    
    return render_template('journal/new.html')

@journal_bp.route('/<int:entry_id>')
@login_required
def view_entry(entry_id):
    entry = JournalEntry.query.filter_by(id=entry_id, user_id=current_user.id).first_or_404()
    return render_template('journal/view.html', entry=entry)

# Mood Tracking Blueprint
mood_bp = Blueprint('mood', __name__, url_prefix='/mood')

@mood_bp.route('/')
@login_required
def index():
    entries = MoodEntry.query.filter_by(user_id=current_user.id).order_by(MoodEntry.created_at.desc()).limit(30).all()
    return render_template('mood/index.html', entries=entries)

@mood_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_entry():
    if request.method == 'POST':
        data = request.get_json()
        entry = MoodEntry(
            user_id=current_user.id,
            mood_score=data.get('mood_score'),
            mood_label=data.get('mood_label'),
            notes=data.get('notes'),
            activities=json.dumps(data.get('activities', [])),
            sleep_hours=data.get('sleep_hours'),
            exercise_minutes=data.get('exercise_minutes'),
            social_interactions=data.get('social_interactions')
        )
        db.session.add(entry)
        db.session.commit()
        return jsonify({'message': 'Mood entry created successfully'}), 201
    
    return render_template('mood/new.html')

# Tasks Blueprint
tasks_bp = Blueprint('tasks', __name__, url_prefix='/tasks')

@tasks_bp.route('/')
@login_required
def index():
    tasks = Task.query.filter_by(user_id=current_user.id).order_by(Task.due_date.asc()).all()
    return render_template('tasks/index.html', tasks=tasks)

@tasks_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_task():
    if request.method == 'POST':
        data = request.get_json()
        task = Task(
            user_id=current_user.id,
            title=data.get('title'),
            description=data.get('description'),
            priority=data.get('priority', 'medium'),
            due_date=datetime.fromisoformat(data.get('due_date')) if data.get('due_date') else None
        )
        db.session.add(task)
        db.session.commit()
        return jsonify({'message': 'Task created successfully'}), 201
    
    return render_template('tasks/new.html')

@tasks_bp.route('/<int:task_id>/complete')
@login_required
def complete_task(task_id):
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    task.status = 'completed'
    task.completed_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'message': 'Task completed'}), 200

# Goals Blueprint
goals_bp = Blueprint('goals', __name__, url_prefix='/goals')

@goals_bp.route('/')
@login_required
def index():
    goals = Goal.query.filter_by(user_id=current_user.id).order_by(Goal.target_date.asc()).all()
    return render_template('goals/index.html', goals=goals)

@goals_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_goal():
    if request.method == 'POST':
        data = request.get_json()
        goal = Goal(
            user_id=current_user.id,
            title=data.get('title'),
            description=data.get('description'),
            category=data.get('category'),
            target_date=datetime.fromisoformat(data.get('target_date')).date() if data.get('target_date') else None
        )
        db.session.add(goal)
        db.session.commit()
        return jsonify({'message': 'Goal created successfully'}), 201
    
    return render_template('goals/new.html')

@goals_bp.route('/<int:goal_id>/update_progress', methods=['POST'])
@login_required
def update_progress(goal_id):
    data = request.get_json()
    goal = Goal.query.filter_by(id=goal_id, user_id=current_user.id).first_or_404()
    goal.progress = data.get('progress', 0)
    db.session.commit()
    return jsonify({'message': 'Progress updated successfully'}), 200

# ML Services Blueprint
ml_bp = Blueprint('ml', __name__, url_prefix='/ml')

@ml_bp.route('/insights')
@login_required
def ml_insights():
    """Show ML insights and model performance"""
    # Get user's journal entries for analysis
    journal_entries = JournalEntry.query.filter_by(user_id=current_user.id).order_by(JournalEntry.created_at.desc()).limit(50).all()
    
    # Get mood entries
    mood_entries = MoodEntry.query.filter_by(user_id=current_user.id).order_by(MoodEntry.created_at.desc()).limit(30).all()
    
    # Calculate basic insights
    total_entries = len(journal_entries)
    avg_mood = sum(entry.mood_score for entry in mood_entries) / len(mood_entries) if mood_entries else 0
    
    # Get model performance
    from ml_services import get_model_performance
    model_performance = get_model_performance()
    
    return render_template('ml/insights.html',
                         journal_entries=journal_entries,
                         mood_entries=mood_entries,
                         total_entries=total_entries,
                         avg_mood=round(avg_mood, 1),
                         model_performance=model_performance)

@ml_bp.route('/sentiment_analysis', methods=['POST'])
@login_required
def sentiment_analysis():
    """Analyze sentiment of text using ML models"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Use enhanced ML service
    from ml_services import analyze_sentiment, preprocess_text, extract_topics
    
    # Basic sentiment analysis
    sentiment_result = analyze_sentiment(text)
    
    # Preprocessing
    processed_text = preprocess_text(text)
    
    # Topic extraction
    topics = extract_topics(processed_text)
    
    # Simple sentiment analysis (placeholder for ML service)
    result = {
        'sentiment': sentiment_result['sentiment'],
        'confidence': sentiment_result['confidence'],
        'text': text,
        'processed_text': processed_text,
        'topics': topics,
        'features': {
            'text_length': len(text),
            'word_count': len(text.split()),
            'processed_length': len(processed_text.split())
        }
    }
    
    return jsonify(result)

@ml_bp.route('/train_model', methods=['POST'])
@login_required
def train_ml_model():
    """Train the baseline ML model with sample data"""
    try:
        from ml_services import train_sample_model
        result = train_sample_model()
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@ml_bp.route('/model_performance')
@login_required
def get_ml_performance():
    """Get current ML model performance metrics"""
    try:
        from ml_services import get_model_performance, baseline_model
        
        # Try to get performance if model exists
        try:
            baseline_model.load_model()
            # Model exists, get actual performance
            performance = {
                'status': 'Model loaded successfully',
                'message': 'Model is ready for predictions',
                'model_info': {
                    'type': 'TF-IDF + Random Forest',
                    'features': '5000 TF-IDF features',
                    'algorithm': 'Random Forest (100 estimators)'
                }
            }
        except:
            performance = get_model_performance()
            
        return jsonify(performance)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@ml_bp.route('/predict', methods=['POST'])
@login_required
def ml_predict():
    """Make prediction using trained ML model"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        from ml_services import baseline_model
        
        # Make prediction
        prediction = baseline_model.predict(text)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'text': text
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}',
            'suggestion': 'Train the model first using /train_model endpoint'
        }), 500

@ml_bp.route('/preprocess', methods=['POST'])
@login_required
def preprocess_text():
    """Preprocess text using the full pipeline"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        from ml_services import preprocess_text, extract_topics
        
        # Apply preprocessing
        processed = preprocess_text(text)
        topics = extract_topics(processed)
        
        return jsonify({
            'original_text': text,
            'processed_text': processed,
            'topics': topics,
            'features': {
                'original_length': len(text),
                'processed_length': len(processed),
                'word_reduction': len(text.split()) - len(processed.split())
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Doctor Recommendations Blueprint
doctors_bp = Blueprint('doctors', __name__, url_prefix='/doctors')

@doctors_bp.route('/')
@login_required
def index():
    """Show doctor recommendations page"""
    return render_template('doctors/index.html')

@doctors_bp.route('/search')
@login_required
def search_doctors():
    """Search for doctors based on criteria"""
    specialty = request.args.get('specialty', '')
    location = request.args.get('location', '')
    
    # Mock doctor data - in production, this would integrate with real APIs
    mock_doctors = [
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
        mock_doctors = [d for d in mock_doctors if specialty.lower() in d['specialty'].lower()]
    
    return jsonify(mock_doctors)

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
    
    # In production, this would create an appointment record
    # For now, just return success
    return jsonify({
        'message': 'Appointment request submitted successfully',
        'appointment_id': 12345,
        'status': 'pending_confirmation'
    }), 201


# Assessments Blueprint (PHQ-9 + SCID-5-PD screening with branching)
assessments_bp = Blueprint('assessments', __name__, url_prefix='/assessments')


def _phq9_bank() -> List[Dict[str, Any]]:
    # Minimal PHQ-9 item bank
    items = [
        {'id': 'phq1', 'text': 'Little interest or pleasure in doing things', 'type': 'likert', 'options': [0, 1, 2, 3]},
        {'id': 'phq2', 'text': 'Feeling down, depressed, or hopeless', 'type': 'likert', 'options': [0, 1, 2, 3]},
        {'id': 'phq3', 'text': 'Trouble falling or staying asleep, or sleeping too much', 'type': 'likert', 'options': [0, 1, 2, 3]},
        {'id': 'phq4', 'text': 'Feeling tired or having little energy', 'type': 'likert', 'options': [0, 1, 2, 3]},
        {'id': 'phq5', 'text': 'Poor appetite or overeating', 'type': 'likert', 'options': [0, 1, 2, 3]},
        {'id': 'phq6', 'text': 'Feeling bad about yourself — or that you are a failure or have let yourself or your family down', 'type': 'likert', 'options': [0, 1, 2, 3]},
        {'id': 'phq7', 'text': 'Trouble concentrating on things, such as reading or watching television', 'type': 'likert', 'options': [0, 1, 2, 3]},
        {'id': 'phq8', 'text': 'Moving or speaking so slowly that other people could have noticed, or being so fidgety or restless that you have been moving a lot more than usual', 'type': 'likert', 'options': [0, 1, 2, 3]},
        {'id': 'phq9', 'text': 'Thoughts that you would be better off dead or of hurting yourself', 'type': 'likert', 'options': [0, 1, 2, 3]},
    ]
    return items


def _scid5pd_bank() -> List[Dict[str, Any]]:
    # Screening subset (~24 items) yes/no style, covering multiple domains; not the full copyrighted text
    items = [
        {'id': 'scid1', 'text': 'Do you often feel a pervasive pattern of distrust and suspicion of others?', 'type': 'bool'},
        {'id': 'scid2', 'text': 'Do you prefer being alone and have little interest in close relationships?', 'type': 'bool'},
        {'id': 'scid3', 'text': 'Do you believe you have special powers or unusual perceptual experiences?', 'type': 'bool'},
        {'id': 'scid4', 'text': 'Do you avoid social situations because of fears of criticism or rejection?', 'type': 'bool'},
        {'id': 'scid5', 'text': 'Do you need to be the center of attention and feel uncomfortable when you are not?', 'type': 'bool'},
        {'id': 'scid6', 'text': 'Do you lack empathy and often exploit others for your own benefit?', 'type': 'bool'},
        {'id': 'scid7', 'text': 'Do you act impulsively and have difficulty planning ahead?', 'type': 'bool'},
        {'id': 'scid8', 'text': 'Do you experience unstable and intense relationships with rapid mood changes?', 'type': 'bool'},
        {'id': 'scid9', 'text': 'Do you have a pervasive pattern of detachment and limited emotional expression?', 'type': 'bool'},
        {'id': 'scid10', 'text': 'Are you excessively devoted to work and productivity to the exclusion of leisure and friendships?', 'type': 'bool'},
        {'id': 'scid11', 'text': 'Are you preoccupied with orderliness, perfectionism, and control?', 'type': 'bool'},
        {'id': 'scid12', 'text': 'Do you fear being alone and go to great lengths to obtain nurturance and support from others?', 'type': 'bool'},
        {'id': 'scid13', 'text': 'Do you have an inflated sense of self-importance and need for admiration?', 'type': 'bool'},
        {'id': 'scid14', 'text': 'Do you frequently disregard social norms or the rights of others?', 'type': 'bool'},
        {'id': 'scid15', 'text': 'Do you feel uncomfortable unless others take responsibility for most areas of your life?', 'type': 'bool'},
        {'id': 'scid16', 'text': 'Do you often have odd beliefs or magical thinking that influences behavior?', 'type': 'bool'},
        {'id': 'scid17', 'text': 'Do you often hold grudges and perceive benign remarks as attacks?', 'type': 'bool'},
        {'id': 'scid18', 'text': 'Do you engage in self-damaging acts or have recurrent suicidal behavior?', 'type': 'bool'},
        {'id': 'scid19', 'text': 'Do you avoid making decisions without excessive advice and reassurance?', 'type': 'bool'},
        {'id': 'scid20', 'text': 'Are you preoccupied with fantasies of unlimited success, power, brilliance, or beauty?', 'type': 'bool'},
        {'id': 'scid21', 'text': 'Do you have unstable self-image or sense of self?', 'type': 'bool'},
        {'id': 'scid22', 'text': 'Do you often act in ways that are reckless or show little regard for safety?', 'type': 'bool'},
        {'id': 'scid23', 'text': 'Do you feel constrained by rules and prefer flexibility to structure?', 'type': 'bool'},
        {'id': 'scid24', 'text': 'Do you find it hard to discard worn-out or worthless items even with no sentimental value?', 'type': 'bool'},
    ]
    return items


@assessments_bp.route('/')
@login_required
def assessments_index():
    return render_template('assessments/index.html')


@assessments_bp.route('/api/next_question', methods=['POST'])
@login_required
def next_question():
    payload = request.get_json(force=True) or {}
    instrument = payload.get('instrument', 'phq9')
    answers = payload.get('answers', [])  # list of {id, value}
    state = payload.get('state') or {}

    if instrument not in ('phq9', 'scid5pd'):
        return jsonify({'error': 'invalid instrument'}), 400

    if instrument == 'phq9':
        bank = _phq9_bank()
        id_to_item = {q['id']: q for q in bank}
        if not state.get('order'):
            order = [q['id'] for q in bank]
            random.shuffle(order)
            state['order'] = order
            state['index'] = 0
        # Branching: if item phq9 answered > 0, schedule a safety follow-up
        answered = {a['id']: a['value'] for a in answers}
        if 'phq9' in answered and answered['phq9'] and not state.get('safety_added'):
            state['safety_added'] = True
            # insert a follow-up immediately next
            state['order'].insert(state['index'] + 1, 'phq9_safety')
            id_to_item['phq9_safety'] = {
                'id': 'phq9_safety',
                'text': 'Have you had any thoughts or plans to harm yourself in the past two weeks?',
                'type': 'bool'
            }
        # Advance to next
        while state['index'] < len(state['order']):
            qid = state['order'][state['index']]
            if qid not in answered:
                item = id_to_item.get(qid)
                state['index'] += 1
                return jsonify({'question': item, 'state': state})
            state['index'] += 1
        # Completed -> score
        score = sum(int(v) for v in answered.values() if isinstance(v, (int, float)))
        severity = (
            'none-minimal' if score <= 4 else
            'mild' if score <= 9 else
            'moderate' if score <= 14 else
            'moderately severe' if score <= 19 else
            'severe'
        )
        # persist session
        try:
            sess = AssessmentSession(
                user_id=current_user.id,
                instrument='phq9',
                completed_at=datetime.utcnow(),
                score=score,
                severity=severity,
                answers_json=json.dumps(answers),
                state_json=json.dumps(state),
            )
            db.session.add(sess)
            db.session.commit()
        except Exception:
            db.session.rollback()
        return jsonify({'done': True, 'score': score, 'severity': severity, 'state': state})

    # SCID-5-PD screening logic
    bank = _scid5pd_bank()
    id_to_item = {q['id']: q for q in bank}
    if not state.get('order'):
        # select at least 20 questions randomly
        ids = [q['id'] for q in bank]
        random.shuffle(ids)
        state['order'] = ids[:max(20, len(ids))] if len(ids) >= 20 else ids
        state['index'] = 0
    answered = {a['id']: a['value'] for a in answers}

    # Branching examples: if scid8 (borderline) yes, add follow-up on self-harm if not already asked
    if answered.get('scid8') is True and 'scid8_follow' not in answered and not state.get('scid8_follow_added'):
        state['scid8_follow_added'] = True
        state['order'].insert(state['index'] + 1, 'scid8_follow')
        id_to_item['scid8_follow'] = {
            'id': 'scid8_follow',
            'text': 'Have mood changes led to impulsive acts or self-harm?',
            'type': 'bool'
        }

    # Iterate to next unanswered question
    while state['index'] < len(state['order']):
        qid = state['order'][state['index']]
        if qid not in answered:
            item = id_to_item.get(qid)
            state['index'] += 1
            return jsonify({'question': item, 'state': state})
        state['index'] += 1

    # Finished: return simple domain tallies
    positive = sum(1 for k, v in answered.items() if str(k).startswith('scid') and v is True)
    risk_flag = any(answered.get(k) for k in ('scid8_follow', 'scid18'))
    try:
        sess = AssessmentSession(
            user_id=current_user.id,
            instrument='scid5pd',
            completed_at=datetime.utcnow(),
            positives=positive,
            risk_flag=risk_flag,
            answers_json=json.dumps(answers),
            state_json=json.dumps(state),
        )
        db.session.add(sess)
        db.session.commit()
    except Exception:
        db.session.rollback()
    return jsonify({'done': True, 'positives': positive, 'risk_flag': risk_flag, 'state': state})


# Chatbot Blueprint
chat_bp = Blueprint('chat', __name__, url_prefix='/chat')


@chat_bp.route('/')
@login_required
def chat_index():
    return render_template('chat/index.html')


@chat_bp.route('/api/chat/history', methods=['GET'])
@login_required
def chat_history():
    """Get chat message history for current user"""
    try:
        messages = ChatMessage.query.filter_by(
            user_id=current_user.id
        ).order_by(ChatMessage.created_at.asc()).limit(50).all()
        
        return jsonify({
            'messages': [{
                'text': msg.text,
                'role': msg.role,
                'created_at': msg.created_at.isoformat() if msg.created_at else None
            } for msg in messages]
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching chat history: {str(e)}")
        return jsonify({'messages': []}), 500

@chat_bp.route('/api/message', methods=['POST'])
@login_required
def chat_message():
    data = request.get_json(force=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'empty'}), 400
    
    # Initialize OpenAI client
    try:
        init_openai()
    except ValueError as e:
        current_app.logger.error(f"OpenAI initialization error: {str(e)}")
        return jsonify({
            'error': 'Chat service is currently unavailable. Please try again later.',
            'requires_escalation': True
        }), 503
    
    # Get conversation history for context
    conversation_history = []
    try:
        # Get last 12 messages for richer context
        last_messages = ChatMessage.query.filter_by(
            user_id=current_user.id
        ).order_by(ChatMessage.created_at.desc()).limit(12).all()
        
        # Format messages for OpenAI (oldest first)
        for msg in reversed(last_messages):
            role = 'user' if msg.role == 'user' else 'assistant'
            conversation_history.append({
                'role': role,
                'content': msg.text
            })
    except Exception as e:
        current_app.logger.error(f"Error fetching conversation history: {str(e)}")
    
    # Generate response using OpenAI
    try:
        response = OpenAIService.generate_response(text, conversation_history)
        
        # Save messages to database
        try:
            db.session.add(ChatMessage(
                user_id=current_user.id,
                role='user',
                text=text,
                sentiment=response.get('risk_level', 'neutral'),
                confidence=0.8 if response.get('requires_escalation', False) else 0.5
            ))
            
            db.session.add(ChatMessage(
                user_id=current_user.id,
                role='bot',
                text=response['response'],
                sentiment=response.get('risk_level', 'neutral'),
                confidence=0.9
            ))
            
            db.session.commit()
        except Exception as e:
            current_app.logger.error(f"Error saving chat messages: {str(e)}")
            db.session.rollback()
        
        # Prepare response with professional referrals if needed
        chat_response = {
            'reply': response['response'],
            'escalate': response.get('requires_escalation', False),
            'risk_level': response.get('risk_level', 'none'),
            'concerns': response.get('concerns', [])
        }
        
        # Add professional referrals for severe cases
        if response.get('requires_escalation', False) or response.get('risk_level') in ['high', 'moderate']:
            chat_response['professional_referrals'] = [
                {
                    'type': 'emergency',
                    'title': 'NIMHANS Helpline',
                    'contact': '080-46110007',
                    'urgent': True
                },
                {
                    'type': 'crisis',
                    'title': 'Suicide Prevention Helpline',
                    'contact': '9152987821',
                    'urgent': True
                },
                {
                    'type': 'professional',
                    'title': 'Mental Health Professionals',
                    'contact': '/doctors/',
                    'urgent': False
                }
            ]
        
        return jsonify(chat_response)
        
    except Exception as e:
        current_app.logger.error(f"Error in chat message handling: {str(e)}")
        return jsonify({
            'error': 'I apologize, but I encountered an error processing your message. Please try again.',
            'requires_escalation': False
        }), 500


# Video/Audio Assessment Blueprint
assessment_bp = Blueprint('assessment', __name__, url_prefix='/assessment')

@assessment_bp.route('/video-audio')
def video_audio_assessment():
    """Render video/audio assessment page"""
    return render_template('assessments/video_audio.html')

@assessment_bp.route('/api/analyze-recording', methods=['POST'])
def analyze_recording():
    """Analyze uploaded video/audio recording for mental health indicators"""
    try:
        print(f"=== ANALYZE RECORDING REQUEST ===")
        print(f"Files received: {list(request.files.keys())}")
        print(f"Form data: {dict(request.form)}")
        
        # Check for recording file
        if 'recording' not in request.files:
            return jsonify({'error': 'No recording file provided'}), 400
        
        recording_file = request.files['recording']
        assessment_type = request.form.get('assessment_type', 'audio-only')
        
        print(f"Recording file: {recording_file.filename}")
        print(f"Assessment type: {assessment_type}")
        
        if recording_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the recording data
        recording_data = recording_file.read()
        print(f"Recording data size: {len(recording_data)} bytes")
        
        # Select fast mode vs high-accuracy
        fast_mode = request.form.get('fast', os.getenv('FAST_MODE', '1')) in ['1', 'true', 'True']
        if fast_mode:
            print("FAST_MODE enabled: using lightweight analysis")
            service = VideoAudioAnalysisService()
            if assessment_type == 'video-audio':
                results = service.analyze_video_audio(recording_data, recording_data)
            else:
                # Reuse audio path via combined API
                results = service.analyze_video_audio(b"", recording_data)
        else:
            print("Using high-accuracy models for analysis...")
            if assessment_type == 'video-audio':
                results = analyze_video_audio_high_accuracy(recording_data, recording_data)
            else:
                results = analyze_audio_high_accuracy(recording_data)
        
        print(f"Analysis results: {results.get('depression_score', 'N/A')} depression, {results.get('confidence_score', 'N/A')} confidence")
        
        # Get latest PHQ-9 and SCID-5-PD sessions for combined assessment
        latest_phq9 = None
        latest_scid = None
        if current_user and current_user.is_authenticated:
            print(f"User authenticated: {current_user.id}")
            latest_phq9 = AssessmentSession.query.filter_by(
                user_id=current_user.id,
                instrument='phq9'
            ).order_by(AssessmentSession.created_at.desc()).first()
            latest_scid = AssessmentSession.query.filter_by(
                user_id=current_user.id,
                instrument='scid5pd'
            ).order_by(AssessmentSession.created_at.desc()).first()
            
            print(f"Latest PHQ-9: {latest_phq9.score if latest_phq9 else 'None'}")
            print(f"Latest SCID-5: {latest_scid.positives if latest_scid else 'None'}")
        else:
            print("User not authenticated, using AV analysis only")

        # Compute comprehensive combined assessment
        print("Computing combined assessment...")
        composite = _compute_composite_assessment(results, latest_phq9, latest_scid)
        print(f"Combined assessment: {composite.get('depression_score', 'N/A')} depression, {composite.get('overall_risk', 'N/A')} risk")
        
        # Update results with combined assessment
        results.update({
            'composite_assessment': composite,
            'depression_score': composite.get('depression_score', results.get('depression_score', 0.5)),
            'depression_level': composite.get('depression_level', results.get('depression_level', 'moderate')),
            'confidence_score': composite.get('confidence_score', results.get('confidence_score', 0.5)),
            'confidence_level': composite.get('confidence_level', results.get('confidence_level', 'moderate')),
            'overall_wellbeing': composite.get('overall_wellbeing', results.get('overall_wellbeing', 'moderate')),
            'overall_risk': composite.get('overall_risk', 'unknown'),
            'risk_factors': composite.get('risk_factors', []),
            'recommendations': composite.get('recommendations', []),
            'inputs_used': composite.get('inputs_used', {}),
            'component_scores': composite.get('component_scores', {}),
            'weighting': composite.get('weighting', {}),
            'completeness_score': composite.get('completeness_score', 0.5)
        })
        
        # Add escalation hints for frontend if severe
        if composite.get('overall_risk') == 'high' or composite.get('depression_level') in ['severe', 'critical']:
            results['requires_escalation'] = True
            results['professional_referrals'] = [
                {'type': 'emergency', 'title': 'NIMHANS Helpline', 'contact': '080-46110007', 'urgent': True},
                {'type': 'crisis', 'title': 'Indian Suicide Prevention', 'contact': '9152987821', 'urgent': True},
                {'type': 'professional', 'title': 'Mental Health Professionals', 'contact': '/doctors/', 'urgent': False}
            ]

        # Save assessment results to database
        try:
            assessment_session = AssessmentSession(
                user_id=current_user.id if current_user.is_authenticated else None,
                assessment_type='video_audio_analysis',
                results=json.dumps(results),
                confidence=composite.get('confidence_score', 0.5),
                created_at=datetime.now(timezone.utc)
            )
            db.session.add(assessment_session)
            db.session.commit()
            print("Assessment saved to database")
        except Exception as e:
            print(f"Error saving assessment: {e}")
            db.session.rollback()
        
        print(f"=== FINAL RESULTS ===")
        print(f"Depression: {results.get('depression_score', 'N/A')} ({results.get('depression_level', 'N/A')})")
        print(f"Confidence: {results.get('confidence_score', 'N/A')} ({results.get('confidence_level', 'N/A')})")
        print(f"Wellbeing: {results.get('overall_wellbeing', 'N/A')}")
        print(f"Risk: {results.get('overall_risk', 'N/A')}")
        print(f"Completeness: {results.get('completeness_score', 'N/A')}")
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error analyzing recording: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to analyze recording',
            'depression_score': None,
            'confidence_score': None,
            'depression_level': 'unknown',
            'confidence_level': 'unknown',
            'overall_wellbeing': 'unknown',
            'overall_risk': 'unknown',
            'recommendations': [
                'Please try the assessment again',
                'Consider speaking with a mental health professional if you continue to experience difficulties'
            ]
        }), 500

@assessment_bp.route('/api/test-combined-assessment', methods=['POST'])
def test_combined_assessment():
    """Test endpoint to verify combined assessment is working"""
    try:
        print("=== TESTING COMBINED ASSESSMENT ===")
        
        # Create mock AV results
        av_results = {
            'depression_score': 0.6,
            'confidence_score': 0.4,
            'video_analysis': {
                'depression_indicators': {'score': 0.65},
                'confidence_indicators': {'score': 0.35}
            },
            'audio_analysis': {
                'depression_indicators': {'score': 0.55},
                'confidence_indicators': {'score': 0.45}
            }
        }
        
        # Get latest PHQ-9 and SCID-5 sessions
        latest_phq9 = None
        latest_scid = None
        if current_user and current_user.is_authenticated:
            latest_phq9 = AssessmentSession.query.filter_by(
                user_id=current_user.id,
                instrument='phq9'
            ).order_by(AssessmentSession.created_at.desc()).first()
            latest_scid = AssessmentSession.query.filter_by(
                user_id=current_user.id,
                instrument='scid5pd'
            ).order_by(AssessmentSession.created_at.desc()).first()
        
        print(f"PHQ-9 available: {latest_phq9 is not None}")
        print(f"SCID-5 available: {latest_scid is not None}")
        
        # Compute combined assessment
        composite = _compute_composite_assessment(av_results, latest_phq9, latest_scid)
        
        print(f"Combined result: {composite}")
        
        return jsonify({
            'success': True,
            'message': 'Combined assessment test completed',
            'av_results': av_results,
            'composite_assessment': composite,
            'phq9_available': latest_phq9 is not None,
            'scid5_available': latest_scid is not None
        })
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@assessment_bp.route('/api/save-assessment', methods=['POST'])
def save_assessment():
    """Save assessment results to user's profile"""
    try:
        if not current_user.is_authenticated:
            return jsonify({'error': 'User not authenticated'}), 401
        
        data = request.get_json()
        
        # Create assessment session record
        assessment_session = AssessmentSession(
            user_id=current_user.id,
            assessment_type='video_audio_manual_save',
            results=json.dumps(data),
            confidence=data.get('confidence_score', 0.5),
            created_at=datetime.now(timezone.utc)
        )
        
        db.session.add(assessment_session)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Assessment saved successfully'})
        
    except Exception as e:
        print(f"Error saving assessment: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to save assessment'}), 500

@assessment_bp.route('/api/model-info')
def get_model_info():
    """Get information about high-accuracy models"""
    try:
        model_info = get_high_accuracy_model_info()
        return jsonify(model_info)
    except Exception as e:
        print(f"Error getting model info: {e}")
        return jsonify({
            'error': 'Failed to get model information',
            'high_accuracy_available': False
        }), 500


