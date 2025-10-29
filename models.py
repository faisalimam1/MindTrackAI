from extensions import db
from flask_login import UserMixin
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.postgresql import JSON

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(20))
    emergency_contact_phone = db.Column(db.String(20))  # Trusted friend/family for crisis support
    emergency_contact_name = db.Column(db.String(100))  # Name of trusted friend/family
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    journal_entries = db.relationship('JournalEntry', backref='user', lazy=True)
    mood_entries = db.relationship('MoodEntry', backref='user', lazy=True)
    tasks = db.relationship('Task', backref='user', lazy=True)
    goals = db.relationship('Goal', backref='user', lazy=True)
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)

class JournalEntry(db.Model):
    __tablename__ = 'journal_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    mood_score = db.Column(db.Integer)  # 1-10 scale
    tags = db.Column(JSON)  # Changed to JSON for PostgreSQL
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # AI Analysis fields
    sentiment_score = db.Column(db.Float)
    emotion_labels = db.Column(JSON)  # Changed to JSON for PostgreSQL
    key_topics = db.Column(JSON)  # Changed to JSON for PostgreSQL
    ai_insights = db.Column(db.Text)

class MoodEntry(db.Model):
    __tablename__ = 'mood_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    mood_score = db.Column(db.Integer, nullable=False)  # 1-10 scale
    mood_label = db.Column(db.String(50))  # e.g., "Happy", "Sad", "Anxious"
    notes = db.Column(db.Text)
    activities = db.Column(JSON)  # Changed to JSON for PostgreSQL
    sleep_hours = db.Column(db.Float)
    exercise_minutes = db.Column(db.Integer)
    social_interactions = db.Column(db.Integer)  # Number of social interactions
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class Task(db.Model):
    __tablename__ = 'tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    priority = db.Column(db.String(20), default='medium')  # low, medium, high
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, completed
    due_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)

class Goal(db.Model):
    __tablename__ = 'goals'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))  # e.g., "Health", "Career", "Personal"
    target_date = db.Column(db.Date)
    progress = db.Column(db.Integer, default=0)  # 0-100 percentage
    status = db.Column(db.String(20), default='active')  # active, completed, abandoned
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AssessmentSession(db.Model):
    __tablename__ = 'assessment_sessions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    instrument = db.Column(db.String(50), nullable=False)  # 'phq9' or 'scid5pd'
    started_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)
    # Results (PHQ-9)
    score = db.Column(db.Integer)
    severity = db.Column(db.String(50))
    # Results (SCID-5-PD)
    positives = db.Column(db.Integer)
    risk_flag = db.Column(db.Boolean)
    # Raw payloads (JSON for PostgreSQL)
    answers_json = db.Column(JSON)  # Changed to JSON for PostgreSQL
    state_json = db.Column(JSON)  # Changed to JSON for PostgreSQL


class AudioVideoAssessment(db.Model):
    __tablename__ = 'audio_video_assessments'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    assessment_type = db.Column(db.String(20), nullable=False)  # 'audio-only' or 'video-audio'

    # Scores and Results
    depression_score = db.Column(db.Float)  # 0.0 to 1.0
    depression_level = db.Column(db.String(20))  # minimal, mild, moderate, severe, critical
    confidence_score = db.Column(db.Float)  # 0.0 to 1.0
    confidence_level = db.Column(db.String(20))  # low, moderate, high
    overall_wellbeing = db.Column(db.String(20))  # excellent, good, moderate, concerning, serious, critical

    # Crisis Detection
    crisis_detected = db.Column(db.Boolean, default=False)
    crisis_indicators = db.Column(JSON)  # List of detected crisis keywords/indicators
    risk_level = db.Column(db.String(20))  # low, mild, moderate, high, critical

    # Transcription
    transcribed_text = db.Column(db.Text)
    transcription_successful = db.Column(db.Boolean, default=False)

    # Voice Features (stored as JSON)
    voice_features = db.Column(JSON)  # pitch, energy, speaking rate, etc.

    # Recommendations
    recommendations = db.Column(JSON)  # Array of recommendation objects

    # Metadata
    audio_duration = db.Column(db.Float)  # Duration in seconds
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'bot'
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class CriticalUser(db.Model):
    __tablename__ = 'critical_users'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    phone_number = db.Column(db.String(20))
    email = db.Column(db.String(120))

    # Assessment details
    assessment_type = db.Column(db.String(50), nullable=False)  # 'phq9', 'scid5pd', 'audio_video', 'composite'
    score = db.Column(db.Float, nullable=False)  # Normalized to 0-100 percentage
    raw_score = db.Column(db.String(50))  # Original score format (e.g., "21/27", "16/20", "0.85")
    severity = db.Column(db.String(50))  # 'severe', 'critical', etc.

    # Alert management
    alert_sent = db.Column(db.Boolean, default=False)
    alert_sent_at = db.Column(db.DateTime)
    admin_viewed = db.Column(db.Boolean, default=False)
    admin_viewed_at = db.Column(db.DateTime)
    admin_notes = db.Column(db.Text)
    resolved = db.Column(db.Boolean, default=False)
    resolved_at = db.Column(db.DateTime)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to user
    user = db.relationship('User', backref='critical_alerts', lazy=True)