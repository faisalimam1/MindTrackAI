#!/usr/bin/env python3
"""
Lightweight startup script for MindTrackAI - avoids heavy ML imports
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import jwt
from functools import wraps
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///mindtrack_ai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key-here')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = 2592000  # 30 days

# Import extensions
from extensions import db, migrate, login_manager, jwt

# Initialize extensions with app
db.init_app(app)
migrate.init_app(app, db)
login_manager.init_app(app)
jwt.init_app(app)

# Custom Jinja2 filters
@app.template_filter('from_json')
def from_json(value):
    if value:
        try:
            import json
            return json.loads(value)
        except:
            return []
    return []

@app.template_filter('strftime')
def strftime_filter(value, format='%Y-%m-%d'):
    """Custom filter to format datetime objects"""
    if value is None:
        return ''
    try:
        if hasattr(value, 'strftime'):
            return value.strftime(format)
        return str(value)
    except:
        return str(value)

# Import models first
from models import User, JournalEntry, MoodEntry, Task, Goal, AssessmentSession, ChatMessage

# Import only basic routes (avoid heavy ML routes for now)
from routes import auth_bp, journal_bp, mood_bp, tasks_bp, goals_bp, doctors_bp, chat_bp

app.register_blueprint(auth_bp)
app.register_blueprint(journal_bp)
app.register_blueprint(mood_bp)
app.register_blueprint(tasks_bp)
app.register_blueprint(goals_bp)
app.register_blueprint(doctors_bp)
app.register_blueprint(chat_bp)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's data for dashboard
    journal_entries = JournalEntry.query.filter_by(user_id=current_user.id).order_by(JournalEntry.created_at.desc()).limit(10).all()
    mood_entries = MoodEntry.query.filter_by(user_id=current_user.id).order_by(MoodEntry.created_at.desc()).limit(30).all()
    tasks = Task.query.filter_by(user_id=current_user.id, status='pending').order_by(Task.due_date.asc()).limit(5).all()
    goals = Goal.query.filter_by(user_id=current_user.id, status='active').order_by(Goal.target_date.asc()).limit(5).all()
    
    # Calculate statistics
    total_journals = len(JournalEntry.query.filter_by(user_id=current_user.id).all())
    total_moods = len(MoodEntry.query.filter_by(user_id=current_user.id).all())
    active_tasks = len(Task.query.filter_by(user_id=current_user.id, status='pending').all())
    active_goals = len(Goal.query.filter_by(user_id=current_user.id, status='active').all())
    
    # Calculate average mood for the last 7 days
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent_moods = MoodEntry.query.filter(
        MoodEntry.user_id == current_user.id,
        MoodEntry.created_at >= week_ago
    ).all()
    
    avg_mood = 0
    if recent_moods:
        avg_mood = sum(entry.mood_score for entry in recent_moods) / len(recent_moods)
    
    # Generate personalized recommendations
    recommendations = generate_recommendations(current_user.id, journal_entries, mood_entries, tasks, goals)

    # Assessments summary
    last_phq9 = AssessmentSession.query.filter_by(user_id=current_user.id, instrument='phq9').order_by(AssessmentSession.completed_at.desc()).first()
    last_scid = AssessmentSession.query.filter_by(user_id=current_user.id, instrument='scid5pd').order_by(AssessmentSession.completed_at.desc()).first()

    # Recent chatbot messages (bot replies)
    recent_chat = ChatMessage.query.filter_by(user_id=current_user.id, role='bot').order_by(ChatMessage.created_at.desc()).limit(5).all()

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
                         recent_chat=recent_chat)

def generate_recommendations(user_id, journal_entries, mood_entries, tasks, goals):
    """Generate personalized recommendations based on user data"""
    recommendations = []
    
    # Analyze mood patterns
    if mood_entries:
        recent_moods = mood_entries[:7]  # Last 7 entries
        avg_recent_mood = sum(entry.mood_score for entry in recent_moods) / len(recent_moods)
        
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
    
    # Check for overdue tasks
    overdue_tasks = [task for task in tasks if task.due_date and task.due_date < datetime.now(timezone.utc)]
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
    
    return recommendations[:5]  # Return top 5 recommendations

@app.route('/api/recommendations')
@login_required
def get_recommendations():
    """API endpoint for getting recommendations"""
    journal_entries = JournalEntry.query.filter_by(user_id=current_user.id).order_by(JournalEntry.created_at.desc()).limit(10).all()
    mood_entries = MoodEntry.query.filter_by(user_id=current_user.id).order_by(MoodEntry.created_at.desc()).limit(30).all()
    tasks = Task.query.filter_by(user_id=current_user.id, status='pending').all()
    goals = Goal.query.filter_by(user_id=current_user.id, status='active').all()
    
    recommendations = generate_recommendations(current_user.id, journal_entries, mood_entries, tasks, goals)
    return jsonify(recommendations)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

def init_database():
    """Initialize the database with tables"""
    print("Initializing database...")
    try:
        with app.app_context():
            # Create all tables
            db.create_all()
            print("Database tables created successfully")
            
            # Check if admin user exists
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                # Create admin user
                admin_user = User(
                    username='admin',
                    email='admin@ai-mental-health.com',
                    password_hash=generate_password_hash('admin123')
                )
                db.session.add(admin_user)
                db.session.commit()
                print("Admin user created (username: admin, password: admin123)")
            
            print("Database initialization complete!")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("MindTrackAI - Lightweight Version")
    print("=" * 50)
    
    # Initialize database
    if not init_database():
        print("\n❌ Database initialization failed.")
        return
    
    print("\nStarting MindTrackAI...")
    print("Web application will be available at: http://localhost:5000")
    print("Admin login: admin / admin123")
    print("\nTo stop the application, press Ctrl+C")
    print("=" * 50)
    
    # Start the Flask application
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\nMindTrackAI stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")

if __name__ == '__main__':
    main()
