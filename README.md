# 🧠 MindTrack AI

### A multimodal mental-health platform that reads text, voice, and video to understand emotional well-being — and responds with personalized, privacy-first insights.

[![Award](https://img.shields.io/badge/🥇%201st%20Prize-INTUITE%202026-FFD700?style=for-the-badge)](#-recognition)
[![IBM](https://img.shields.io/badge/Presented-IBM%20Global%20CSR%20Summit%20South%20Asia%202025-052FAD?style=for-the-badge&logo=ibm&logoColor=white)](#-recognition)
[![IEEE](https://img.shields.io/badge/IEEE%20Xplore-Paper%20Accepted-00629B?style=for-the-badge&logo=ieee&logoColor=white)](#-recognition)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=flat)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

---

## 🏆 Recognition

- 🥇 **1st Prize — INTUITE 2026**
- 🎤 **Presented at the IBM Global CSR Summit, South Asia 2025**
- 📄 **Research paper accepted for IEEE Xplore** (publication pending)

> **Role:** Lead architect & sole owner, supported by two contributors.

---

## ✨ What It Does

MindTrack AI fuses **three modalities** to build a richer picture of mental state than text alone:

| Modality | Signals analyzed | Models / tools |
|----------|------------------|----------------|
| 📝 **Text** | Sentiment, emotion, themes from journal entries | BERT · VADER · spaCy · TextBlob |
| 🎙️ **Voice** | Transcript + vocal tone / acoustic features | Whisper · Librosa |
| 🎥 **Video** | Facial emotion & micro-expression cues | OpenCV · DeepFace · FER · MediaPipe |

A **7-tier detection logic** combines these signals into a composite, confidence-aware well-being assessment — paired with mood tracking, task/goal management, and personalized recommendations.

---

## 🛠️ Tech Stack

**ML / AI**
- **Text:** BERT (transformer sentiment/emotion), VADER, spaCy (NER), TextBlob, scikit-learn (LDA topic modeling)
- **Voice:** Whisper (speech-to-text), Librosa (audio feature extraction)
- **Video:** OpenCV, DeepFace, FER, MediaPipe (facial-emotion analysis)

**Backend & Infra**
- Flask · SQLAlchemy · PostgreSQL · Redis · Celery (async ML jobs) · Docker / Docker Compose

**Frontend**
- Bootstrap 5 · Chart.js · Font Awesome

---

## 🚀 Quick Start (Docker)

```bash
git clone https://github.com/faisalimam1/MindTrackAI.git
cd MindTrackAI
cp env_example.txt .env        # fill in your configuration
docker-compose up -d
# Web app → http://localhost:5000
```

<details>
<summary>Local development (without Docker)</summary>

```bash
pip install -r requirements.txt
createdb mindtrack_ai
redis-server

export FLASK_APP=app.py
export DATABASE_URL=postgresql://localhost/mindtrack_ai
export REDIS_URL=redis://localhost:6379/0

flask db init && flask db migrate && flask db upgrade
python app.py
# In a second terminal:
celery -A ml_services.celery_app worker --loglevel=info
```
</details>

---

## 🏗️ Architecture

```
MindTrackAI/
├── app.py            # Flask application entrypoint
├── models.py         # SQLAlchemy models (users, journals, mood, tasks, goals)
├── routes.py         # HTTP routes
├── ml_services.py    # Text/voice/video ML pipelines (Celery tasks)
├── extensions.py     # Flask extensions
├── templates/        # Jinja templates
├── static/           # CSS / JS / assets
├── docker-compose.yml
└── requirements.txt
```

**Containers:** `web` (Flask) · `postgres` · `redis` · `celery_worker` · `celery_beat` · optional `nginx`.

---

## 🔒 Privacy & Safety

- Password hashing (bcrypt), Flask-Login sessions, CSRF protection, input validation, SQLAlchemy ORM (SQL-injection safe).
- Built for **mental-health support and awareness** — not a diagnostic or clinical tool.

---

## 📊 Core Endpoints

| Area | Endpoints |
|------|-----------|
| Auth | `POST /register` · `POST /login` · `GET /logout` |
| Journal | `GET /journal/` · `POST /journal/new` · `GET /journal/<id>` |
| Mood | `GET /mood/` · `POST /mood/new` |
| Tasks / Goals | `GET /tasks/` · `POST /tasks/new` · `GET /goals/` · `POST /goals/new` |
| ML | `GET /ml/insights` · `POST /ml/sentiment_analysis` |

---

## 📞 Contact

**Faisal Imam** — [LinkedIn](https://www.linkedin.com/in/faisalimam19) · imamfaisal36@gmail.com · [Issues](https://github.com/faisalimam1/MindTrackAI/issues)

---

**MindTrack AI** — *Track your mind. Understand your soul.* 🧠✨
© Faisal Imam & Team · MIT License
