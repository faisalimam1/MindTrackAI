# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MindTrack AI is a mental health tracking and analysis platform that uses Flask, PostgreSQL, and machine learning models to provide sentiment analysis, mood tracking, journal analysis, and crisis detection. The application includes critical safety features for detecting suicidal ideation and severe depression indicators.

## Architecture

### Core Application Structure

The application uses Flask blueprints for modular organization:

- **app.py**: Main application entry point, initializes Flask app, database, extensions, and registers blueprints
- **models.py**: SQLAlchemy database models (User, JournalEntry, MoodEntry, Task, Goal, AssessmentSession, ChatMessage)
- **routes.py**: All Flask blueprints and route handlers (auth, journal, mood, tasks, goals, ml, doctors, assessments, chat)
- **extensions.py**: Flask extension initialization (SQLAlchemy, Flask-Migrate, Flask-Login, JWT)
- **ml_services.py**: Machine learning services including **CRITICAL SAFETY FEATURES** for crisis detection

### Database Models (PostgreSQL with JSON Support)

All JSON fields use `sqlalchemy.dialects.postgresql.JSON` type:
- **User**: Authentication and user management
- **JournalEntry**: Text entries with AI analysis (sentiment_score, emotion_labels, key_topics, ai_insights)
- **MoodEntry**: Daily mood tracking with activities, sleep_hours, exercise_minutes, social_interactions
- **Task**: Task management with priority and status tracking
- **Goal**: Goal setting with progress tracking (0-100%)
- **AssessmentSession**: PHQ-9 and SCID-5-PD assessment results
- **ChatMessage**: Chat history with sentiment and crisis detection

### Blueprint Organization

Routes are organized into blueprints with URL prefixes:
- `/auth` - User registration, login, logout
- `/journal` - Journal entry CRUD operations
- `/mood` - Mood tracking entries
- `/tasks` - Task management
- `/goals` - Goal setting and progress tracking
- `/ml` - ML insights and sentiment analysis
- `/doctors` - Doctor recommendations
- `/assessments` - Assessment overview
- `/chat` - AI chat interface with crisis detection
- `/assessment` - Video/audio assessment tools

### Machine Learning Pipeline

The ML services are split across multiple files:

1. **ml_services.py**: Core sentiment analysis, crisis detection, text preprocessing, topic extraction
2. **high_accuracy_models.py**: Advanced models for higher accuracy analysis
3. **video_audio_service.py**: Video/audio analysis for depression detection
4. **video_audio_service_high_accuracy.py**: High-accuracy video/audio analysis

**CRITICAL SAFETY FEATURE**: Crisis detection runs FIRST before any other ML analysis. Keywords include "suicide", "kill myself", "end my life", etc. Crisis responses trigger immediate helpline referrals.

## Development Commands

### Database Setup

```bash
# Initialize database
python init_db.py

# Run migrations (if using Flask-Migrate)
flask db init
flask db migrate -m "Migration message"
flask db upgrade
```

### Running the Application

```bash
# Development mode (preferred method)
python app.py

# Alternative: using Flask CLI
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000

# Production mode with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# HTTPS mode (for testing)
python run_https.py
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up -d --build
```

### Testing & Development

```bash
# Install dependencies
pip install -r requirements.txt

# Scan for API usage patterns
python scan_usage.py

# Run with virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
python app.py
```

## Configuration

### Environment Variables (.env file required)

```bash
# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_ENV=development

# Database (PostgreSQL required)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mindmate

# JWT Authentication
JWT_SECRET_KEY=your-jwt-secret-key-here

# File Uploads
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=104857600  # 100MB in bytes

# Optional: Redis for caching/Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
```

### Database Configuration

The application REQUIRES PostgreSQL (not SQLite) due to:
- JSON/JSONB column types used in models
- Better performance for ML workloads
- Production-ready features

Connection format: `postgresql://username:password@host:port/database`

## Key Implementation Details

### JSON Field Handling in Templates

Use custom Jinja2 filters for PostgreSQL JSON fields:

```python
# In templates
{{ entry.tags|from_json }}
{{ entry.created_at|strftime('%Y-%m-%d') }}
```

These filters handle both JSON strings and native JSON objects from PostgreSQL.

### Authentication Flow

- Uses Flask-Login for session management
- Passwords hashed with Werkzeug security (bcrypt)
- JWT tokens available for API authentication
- `@login_required` decorator protects routes
- `current_user` provides authenticated user context

### Crisis Detection System (CRITICAL)

**ml_services.py** contains safety-critical code:

```python
CRISIS_KEYWORDS = ['suicide', 'suicidal', 'kill myself', ...]
SEVERE_DEPRESSION_KEYWORDS = ['hopeless', 'worthless', ...]

def detect_crisis(text: str) -> tuple[bool, float]:
    # Returns (is_crisis, confidence_level)
    # Always check this FIRST before other analysis
```

**Routes that use crisis detection:**
- `/ml/sentiment_analysis` - Text sentiment analysis with crisis override
- `/chat/api/message` - Chat messages with immediate crisis response

Crisis responses include:
- NIMHANS Helpline: 080-46110007 (24/7)
- iCall: 9152987821 (Mon-Sat, 8 AM - 10 PM)
- Vandrevala Foundation: 1860-2662-345 (24/7)

### Recommendation Engine

The `generate_recommendations()` function in **app.py** analyzes:
- Average mood scores (last 7 days)
- Sleep patterns (target: 7-9 hours)
- Exercise levels (target: 30+ minutes daily)
- Social interactions (target: 2+ daily)
- Overdue tasks
- Goals behind schedule

Returns prioritized recommendations (high/medium priority) with actionable steps.

### ML Model Architecture

**BaselineMLModel** class uses:
- TF-IDF vectorization (5000 features, bigrams)
- Random Forest classifier (100 estimators)
- Text preprocessing with spaCy or NLTK fallback
- Model persistence with pickle in `models/` directory

**Sentiment Analysis** uses multi-layer fallback:
1. Crisis detection (keyword matching)
2. NLTK VADER (if available)
3. TextBlob (fallback)
4. Neutral default (if all fail)

### Video/Audio Processing

Located in **video_audio_service.py** and **video_audio_service_high_accuracy.py**:
- Uses OpenCV for facial expression analysis
- Librosa for audio feature extraction (MFCC, spectral features)
- DeepFace and FER for emotion detection
- OpenAI Whisper for speech transcription
- Combines visual and audio cues for depression assessment

### Enhanced Audio Analysis with Crisis Detection

**NEW SYSTEM** - Located in **audio_crisis_detector.py** and **enhanced_audio_service.py**:

#### Crisis Detection System

Multi-layered crisis detection with 7 severity-weighted keyword categories:

1. **Explicit Suicidal (1.0 severity)**: "kill myself", "end my life", "want to die", "commit suicide"
2. **Self-Harm Intent (0.95)**: "hurt myself", "cut myself", "overdose", "jump off"
3. **Hopelessness (0.85)**: "no hope", "no point", "give up", "no way out"
4. **Severe Depression (0.75)**: "worthless", "hate myself", "burden", "empty inside"
5. **Pain/Suffering (0.70)**: "can't take it anymore", "unbearable pain", "suffering too much"
6. **Isolation (0.65)**: "nobody cares", "alone forever", "better off without me"
7. **Distress (0.50)**: "can't cope", "falling apart", "drowning", "exhausted"

#### Scoring Algorithm

```python
# Base severity calculation
base_score = (
    max_severity * 0.50 +        # Highest indicator
    avg_severity * 0.30 +        # Average of all
    category_factor * 0.20       # Unique categories
)

# Apply contextual modifiers
final_score = base_score × urgency_multiplier × (1 - past_tense) × (1 - protective)

# Crisis threshold: >= 0.65
```

**Contextual Modifiers**:
- **Urgency multipliers** (1.0-1.5×): "right now", "tonight", "going to", "about to"
- **Past tense reduction** (0-0.3×): "used to", "in the past", "long ago"
- **Protective factors** (0-0.25×): "getting help", "seeing therapist", "reasons to live"

#### Enhanced Depression Scoring

High-risk phrases significantly boost depression scores:
```python
if crisis_detected:
    crisis_boost = severity_score * 0.4  # Up to 0.4 boost
    depression_score = min(base_score + crisis_boost, 1.0)
```

Additional boosts from:
- High emotion intensity + negative sentiment: +0.15
- High urgency + distress: +0.10

#### Safety-Focused Recommendations

Risk-appropriate responses (NO self-help for critical cases):

**CRITICAL (≥0.85)**:
- Immediate crisis helplines (NIMHANS: 080-46110007, iCall: 9152987821)
- Emergency services
- Hospital emergency room
- **No self-care suggestions**

**HIGH (≥0.65)**:
- Urgent professional consultation (psychiatrist/psychologist within 24-48 hours)
- Trusted support person notification
- Symptom monitoring
- Limited self-care suggestions

**MODERATE (0.35-0.65)**:
- Professional therapy recommendation
- Self-care suggestions (journaling, sleep, exercise)
- Support network engagement

**LOW (<0.35)**:
- Wellness and prevention tips
- Maintain positive practices

#### Emotion Intensity Analysis

Detects emotional amplification from:
- Intensity modifiers: "extremely", "very", "completely", "totally"
- Exclamation marks (≥3 = 1.3× multiplier)
- Repeated words (emotional emphasis)
- Sentiment polarity magnitude

#### Speech Urgency Detection

Combines text and prosodic features:
- **Text urgency**: "right now", "immediately", "urgent", "emergency"
- **Prosody urgency**:
  - Fast + few pauses = high urgency (0.8)
  - Slow + many pauses = distressed urgency (0.75)
  - Moderate fast = moderate urgency (0.6)

## Template Structure

Templates use Jinja2 with Bootstrap 5:

```
templates/
├── base.html              # Base layout with navbar, imports
├── index.html             # Landing page
├── dashboard.html         # Main user dashboard
├── login.html             # Login form
├── register.html          # Registration form
├── assessments/           # Assessment templates
├── chat/                  # Chat interface
├── doctors/               # Doctor recommendations
├── goals/                 # Goal management
├── journal/               # Journal entries
├── ml/                    # ML insights
├── mood/                  # Mood tracking
└── tasks/                 # Task management
```

Each subdirectory typically contains:
- `index.html` - List view
- `new.html` - Create form
- `view.html` or `edit.html` - Detail/edit views

## Common Development Tasks

### Adding a New Route

1. Define route in appropriate blueprint in **routes.py**
2. Add database model if needed in **models.py**
3. Create corresponding template in **templates/**
4. Update navigation in **templates/base.html** if needed

### Adding ML Analysis

1. Define analysis function in **ml_services.py**
2. Ensure crisis detection runs FIRST for user safety
3. Add API endpoint in `/ml` blueprint
4. Update dashboard or insights page to display results

### Database Migrations

When models change:
```bash
flask db migrate -m "Description of changes"
flask db upgrade
```

### Adding New Dependencies

1. Add to **requirements.txt**
2. Install: `pip install -r requirements.txt`
3. Update Dockerfile if using Docker
4. Test compatibility with Python 3.11+

## Security Considerations

- Never commit `.env` file (use `env_example.txt` as template)
- Change `SECRET_KEY` and `JWT_SECRET_KEY` in production
- Database credentials should use environment variables
- User passwords are hashed, never stored in plaintext
- CSRF protection enabled by default
- Input validation on all form submissions
- Crisis detection is a safety-critical feature - never bypass or modify without thorough testing

## Performance Notes

- Database queries use pagination for large datasets (limit 30-50 items)
- Mood entries limited to last 30 days in most views
- Journal entries limited to last 10-50 in dashboard
- TF-IDF vectorization cached after first load
- ML models loaded once and reused (singleton pattern)
- Static files should be served via nginx in production

## Testing Crisis Detection

### Test Phrases and Expected Results

**CRITICAL Level (≥0.85 severity)**:
```python
"I want to kill myself right now" → CRITICAL + urgency
"I'm going to end my life tonight" → CRITICAL + urgency
"I can't go on, I want to die" → CRITICAL

Expected: Immediate crisis helplines, emergency services, NO self-care
```

**HIGH Level (0.65-0.85 severity)**:
```python
"I feel completely hopeless and worthless" → HIGH
"Nobody cares about me, I want to disappear" → HIGH
"I'm thinking about hurting myself" → HIGH

Expected: Urgent professional help, psychiatrist/psychologist referral
```

**MODERATE Level (0.35-0.65 severity)**:
```python
"I'm feeling really sad and tired lately" → MODERATE
"I can't cope with everything, I'm falling apart" → MODERATE
"Life feels meaningless and empty" → MODERATE

Expected: Therapy recommendation + self-care suggestions
```

**With Protective Factors (reduced severity)**:
```python
"I used to feel suicidal but I'm better now" → Reduced (past tense)
"I'm having dark thoughts but I'm seeing a therapist" → Reduced (protective)
"I felt hopeless before but I'm getting help" → Reduced (both)

Expected: Lower scores due to past tense and protective factors
```

**False Positive Prevention**:
```python
"The movie made me want to die laughing" → Context analysis
"I'm dying to see you" (colloquial) → Context analysis
"I'm killing it at work" (positive idiom) → Context analysis

Expected: Should NOT trigger crisis detection
```

### Testing Workflow

1. **Use test accounts** - Never test with production user data
2. **Verify helpline numbers** - Ensure all crisis contacts are current
3. **Test keyword matching** - Try various phrasings and word combinations
4. **Test contextual modifiers**:
   - Add urgency words → score should increase
   - Add past tense → score should decrease
   - Add protective factors → score should decrease
5. **Verify response appropriateness**:
   - CRITICAL: Only crisis resources, no self-help
   - HIGH: Professional referrals + limited self-care
   - MODERATE/LOW: Mixed recommendations
6. **Test audio pipeline**:
   - Record test audio with crisis phrases
   - Verify transcription accuracy
   - Confirm crisis detection triggers correctly
7. **Check logging**:
   - Critical log messages for crisis detection
   - User privacy maintained (no sensitive data in logs)
8. **Test error handling**:
   - Silent audio → graceful fallback
   - Failed transcription → voice-based analysis
   - Service unavailable → appropriate error messages

### Manual Testing Commands

```python
# Test crisis detector directly
from audio_crisis_detector import detect_crisis_from_text

# Test various phrases
test_phrases = [
    "I want to kill myself",
    "I'm feeling hopeless",
    "I used to want to die but I'm better now",
    "I'm seeing a therapist for my depression"
]

for phrase in test_phrases:
    is_crisis, confidence, details = detect_crisis_from_text(phrase)
    print(f"'{phrase}' → Crisis: {is_crisis}, Score: {confidence:.3f}, Risk: {details['risk_level']}")
```

### Safety Testing Checklist

When testing crisis detection features:
1. ✅ Use test accounts only, never production data
2. ✅ Verify helpline numbers are current and correct (NIMHANS, iCall, Vandrevala)
3. ✅ Test keyword matching with various phrasings
4. ✅ Verify contextual modifiers work correctly (urgency, past tense, protective)
5. ✅ Ensure immediate response (no delays) for CRITICAL cases
6. ✅ Verify database logging of crisis events (if enabled)
7. ✅ Test escalation flow end-to-end
8. ✅ Confirm appropriate recommendations for each risk level
9. ✅ Test false positive prevention (colloquial phrases, idioms)
10. ✅ Verify transcription accuracy with various accents and speech patterns

## Production Deployment Checklist

1. Set strong `SECRET_KEY` and `JWT_SECRET_KEY`
2. Use managed PostgreSQL database
3. Enable SSL/TLS (HTTPS)
4. Set `FLASK_ENV=production`
5. Use Gunicorn with multiple workers
6. Configure nginx as reverse proxy
7. Set up database backups
8. Enable application logging
9. Monitor crisis detection events
10. Test all helpline numbers are accessible

## File Upload Handling

- Upload directory: `uploads/` (created automatically)
- Max file size: 100MB (configurable via `MAX_CONTENT_LENGTH`)
- Supported formats: Audio (wav, mp3, m4a), Video (mp4, avi, mov)
- Files stored with sanitized names
- Clean up old uploads periodically (not automated)

## API Response Formats

Most API endpoints return JSON with consistent structure:

```json
{
  "message": "Success message",
  "data": {...},
  "error": "Error message if failed"
}
```

Crisis detection responses include:
```json
{
  "crisis": true,
  "escalate": true,
  "risk_level": "CRITICAL",
  "requires_immediate_help": true,
  "professional_referrals": [...]
}
```

## Timezone Handling

All datetime fields use UTC (`datetime.now(timezone.utc)`). Display times should be converted to user's local timezone in the frontend if needed.

## Known Limitations

- No automated Celery worker setup (background tasks commented out)
- ML models require training before first use
- Video/audio analysis requires significant CPU/GPU resources
- OpenAI Whisper model download required for transcription (~1-3GB depending on model size)
- spaCy model (`en_core_web_sm`) must be downloaded: `python -m spacy download en_core_web_sm`

## Development Workflow

1. Create feature branch from main
2. Update models if database schema changes
3. Run migrations locally and test
4. Add/update routes and templates
5. Test crisis detection if modifying ML services
6. Update this CLAUDE.md if architecture changes
7. Commit with descriptive messages
8. Test in Docker environment before deploying
