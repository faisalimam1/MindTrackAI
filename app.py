from flask import Flask, render_template, jsonify, redirect, url_for
from flask_login import login_required, current_user
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ===== CONFIGURATION =====
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')

# PostgreSQL Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/mindmate'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key-here')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = 2592000  # 30 days

# Upload Configuration
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# ===== INITIALIZE EXTENSIONS =====
from extensions import db, migrate, login_manager, jwt  # noqa: E402

db.init_app(app)
migrate.init_app(app, db)
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

try:
    jwt.init_app(app)
except Exception as e:
    print(f"JWT initialization warning: {e}")

# ===== JINJA2 FILTERS FOR POSTGRESQL JSON SUPPORT =====
@app.template_filter('from_json')
def from_json(value):
    """Handle both JSON strings and native JSON objects"""
    if value:
        try:
            if isinstance(value, (dict, list)):
                return value
            if isinstance(value, str):
                return json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
    return []

@app.template_filter('strftime')
def strftime_filter(value, format='%Y-%m-%d'):
    """Format datetime objects"""
    if value is None:
        return ''
    try:
        if hasattr(value, 'strftime'):
            return value.strftime(format)
        return str(value)
    except (AttributeError, ValueError, TypeError):
        return str(value)

# ===== IMPORT MODELS =====
from models import User, JournalEntry, MoodEntry, Task, Goal, AssessmentSession, ChatMessage, CriticalUser  # noqa: E402

# ===== USER LOADER =====
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ===== IMPORT AND REGISTER BLUEPRINTS =====
try:
    from routes import (
        auth_bp, journal_bp, mood_bp, tasks_bp, goals_bp,
        ml_bp, doctors_bp, assessments_bp, chat_bp, assessment_bp, admin_bp
    )

    app.register_blueprint(auth_bp)
    app.register_blueprint(journal_bp)
    app.register_blueprint(mood_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(goals_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(doctors_bp)
    app.register_blueprint(assessments_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(assessment_bp)
    app.register_blueprint(admin_bp)

    print("[OK] All blueprints registered successfully")
except Exception as e:
    print(f"[WARNING] Blueprint registration error: {e}")

# ===== MAIN ROUTES =====
@app.route('/')
def index():
    """Homepage"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with comprehensive data"""
    try:
        # Get user's data
        journal_entries = JournalEntry.query.filter_by(
            user_id=current_user.id
        ).order_by(JournalEntry.created_at.desc()).all()[:10]

        mood_entries = MoodEntry.query.filter_by(
            user_id=current_user.id
        ).order_by(MoodEntry.created_at.desc()).all()[:30]

        tasks = Task.query.filter_by(
            user_id=current_user.id,
            status='pending'
        ).order_by(Task.due_date.asc()).all()[:5]

        goals = Goal.query.filter_by(
            user_id=current_user.id,
            status='active'
        ).order_by(Goal.target_date.asc()).all()[:5]

        # Calculate statistics
        total_journals = JournalEntry.query.filter_by(user_id=current_user.id).count()
        total_moods = MoodEntry.query.filter_by(user_id=current_user.id).count()
        active_tasks = Task.query.filter_by(user_id=current_user.id, status='pending').count()
        active_goals = Goal.query.filter_by(user_id=current_user.id, status='active').count()

        # Calculate average mood for last 7 days
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        recent_moods = MoodEntry.query.filter(
            MoodEntry.user_id == current_user.id,
            MoodEntry.created_at >= week_ago
        ).all()
        
        avg_mood = 0
        if recent_moods:
            avg_mood = sum(entry.mood_score for entry in recent_moods) / len(recent_moods)

        # Generate recommendations
        recommendations = generate_recommendations(
            current_user.id, journal_entries, mood_entries, tasks, goals
        )

        # Assessment summaries
        last_phq9 = AssessmentSession.query.filter_by(
            user_id=current_user.id, 
            instrument='phq9'
        ).order_by(AssessmentSession.completed_at.desc()).first()
        
        last_scid = AssessmentSession.query.filter_by(
            user_id=current_user.id, 
            instrument='scid5pd'
        ).order_by(AssessmentSession.completed_at.desc()).first()

        # Recent chat messages
        recent_chat = ChatMessage.query.filter_by(
            user_id=current_user.id,
            role='bot'
        ).order_by(ChatMessage.created_at.desc()).all()[:5]

        return render_template('dashboard.html',
            journal_entries=journal_entries,
            mood_entries=mood_entries,
            tasks=tasks,
            goals=goals,
            total_journals=total_journals,
            total_moods=total_moods,
            active_tasks=active_tasks,
            active_goals=active_goals,
            avg_mood=round(avg_mood, 1),
            recommendations=recommendations,
            last_phq9=last_phq9,
            last_scid=last_scid,
            recent_chat=recent_chat
        )
    except Exception as e:
        print(f"Dashboard error: {e}")
        return render_template('dashboard.html',
            journal_entries=[],
            mood_entries=[],
            tasks=[],
            goals=[],
            total_journals=0,
            total_moods=0,
            active_tasks=0,
            active_goals=0,
            avg_mood=0,
            recommendations=[],
            last_phq9=None,
            last_scid=None,
            recent_chat=[]
        )

def generate_recommendations(user_id, journal_entries, mood_entries, tasks, goals):
    """Generate personalized recommendations"""
    recommendations = []
    
    # Analyze mood patterns
    if mood_entries:
        recent_moods = mood_entries[:7]
        avg_recent_mood = sum(e.mood_score for e in recent_moods) / len(recent_moods)
        
        if avg_recent_mood < 5:
            recommendations.append({
                'type': 'mood',
                'priority': 'high',
                'title': 'Consider Professional Help',
                'description': 'Your mood has been consistently low. Consider speaking with a mental health professional.',
                'action': 'Find nearby therapists',
                'icon': 'fas fa-user-md',
                'color': 'danger'
            })
        
        if avg_recent_mood < 6:
            recommendations.append({
                'type': 'self_care',
                'priority': 'medium',
                'title': 'Self-Care Activities',
                'description': 'Try engaging in activities that usually boost your mood.',
                'action': 'Schedule self-care time',
                'icon': 'fas fa-heart',
                'color': 'warning'
            })

    # Sleep analysis
    sleep_entries = [e for e in mood_entries if e.sleep_hours]
    if sleep_entries:
        avg_sleep = sum(e.sleep_hours for e in sleep_entries) / len(sleep_entries)
        if avg_sleep < 7:
            recommendations.append({
                'type': 'health',
                'priority': 'medium',
                'title': 'Improve Sleep Quality',
                'description': f'Your average sleep is {avg_sleep:.1f} hours. Aim for 7-9 hours.',
                'action': 'Set bedtime routine',
                'icon': 'fas fa-moon',
                'color': 'info'
            })

    # Exercise analysis
    exercise_entries = [e for e in mood_entries if e.exercise_minutes]
    if exercise_entries:
        avg_exercise = sum(e.exercise_minutes for e in exercise_entries) / len(exercise_entries)
        if avg_exercise < 30:
            recommendations.append({
                'type': 'health',
                'priority': 'medium',
                'title': 'Increase Physical Activity',
                'description': f'Your average exercise is {avg_exercise:.0f} minutes. Aim for 30+ minutes daily.',
                'action': 'Plan exercise routine',
                'icon': 'fas fa-dumbbell',
                'color': 'success'
            })

    # Social interactions
    social_entries = [e for e in mood_entries if e.social_interactions]
    if social_entries:
        avg_social = sum(e.social_interactions for e in social_entries) / len(social_entries)
        if avg_social < 2:
            recommendations.append({
                'type': 'social',
                'priority': 'medium',
                'title': 'Increase Social Connections',
                'description': "You've had limited social interactions. Consider reaching out to friends or family.",
                'action': 'Schedule social time',
                'icon': 'fas fa-users',
                'color': 'primary'
            })

    # Overdue tasks
    overdue_tasks = [t for t in tasks if t.due_date and t.due_date < datetime.now(timezone.utc)]
    if overdue_tasks:
        recommendations.append({
            'type': 'productivity',
            'priority': 'high',
            'title': 'Overdue Tasks',
            'description': f'You have {len(overdue_tasks)} overdue task(s). Consider rescheduling or completing them.',
            'action': 'Review overdue tasks',
            'icon': 'fas fa-exclamation-triangle',
            'color': 'danger'
        })

    # Goals needing attention
    goals_needing_attention = [
        g for g in goals 
        if g.progress < 30 and g.target_date and 
        (g.target_date - datetime.now(timezone.utc).date()).days < 30
    ]
    if goals_needing_attention:
        recommendations.append({
            'type': 'goals',
            'priority': 'medium',
            'title': 'Goals Need Attention',
            'description': 'Some of your goals are behind schedule. Review and adjust your plans.',
            'action': 'Review goals',
            'icon': 'fas fa-bullseye',
            'color': 'warning'
        })

    return recommendations[:5]

# ===== API ROUTES =====
@app.route('/api/recommendations')
@login_required
def get_recommendations():
    """API endpoint for recommendations"""
    journal_entries = JournalEntry.query.filter_by(
        user_id=current_user.id
    ).order_by(JournalEntry.created_at.desc()).all()[:10]

    mood_entries = MoodEntry.query.filter_by(
        user_id=current_user.id
    ).order_by(MoodEntry.created_at.desc()).all()[:30]
    
    tasks = Task.query.filter_by(user_id=current_user.id, status='pending').all()
    goals = Goal.query.filter_by(user_id=current_user.id, status='active').all()
    
    recommendations = generate_recommendations(
        current_user.id, journal_entries, mood_entries, tasks, goals
    )
    
    return jsonify(recommendations)

# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('403.html'), 403

# ===== CREATE UPLOAD FOLDER =====
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ===== DATABASE INITIALIZATION =====
if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("[OK] Database tables created successfully!")
        except Exception as e:
            print(f"[WARNING] Database error: {e}")

    # Run the application
    print("[STARTING] MindTrack AI...")
    print("Running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)