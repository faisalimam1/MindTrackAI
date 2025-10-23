from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import os

# CRITICAL: Crisis detection keywords
CRISIS_KEYWORDS = [
    'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die', 
    'better off dead', 'no reason to live', "can't go on", 'end it all',
    'take my life', 'harm myself', 'self harm', 'cut myself', 
    'overdose', 'jump off', 'hang myself', 'shoot myself'
]

SEVERE_DEPRESSION_KEYWORDS = [
    'hopeless', 'worthless', 'useless', 'hate myself', 'failure',
    'give up', 'no point', 'meaningless', 'empty', 'numb'
]

def detect_crisis(text: str):
    """Detect crisis/suicidal language - CRITICAL SAFETY FEATURE"""
    if not text:
        return False, 0.0
    
    text_lower = text.lower()
    
    # Check for crisis keywords
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return True, 1.0  # MAXIMUM CRISIS LEVEL
    
    # Check for severe depression indicators
    severe_count = sum(1 for keyword in SEVERE_DEPRESSION_KEYWORDS if keyword in text_lower)
    if severe_count >= 3:
        return True, 0.9  # HIGH CRISIS LEVEL
    
    return False, 0.0

def _ensure_nltk_data():
    try:
        from nltk.data import find
        try:
            find('sentiment/vader_lexicon.zip')
        except LookupError:
            import nltk as _nltk
            _nltk.download('vader_lexicon')
        try:
            find('tokenizers/punkt')
        except LookupError:
            import nltk as _nltk
            _nltk.download('punkt')
    except Exception:
        pass

def analyze_sentiment(text: str):
    """Return sentiment with CRISIS DETECTION"""
    if not text:
        return {'sentiment': 'neutral', 'confidence': 0.0, 'text': text, 'crisis': False}

    # CRITICAL: Check for crisis language FIRST
    is_crisis, crisis_confidence = detect_crisis(text)
    
    if is_crisis:
        return {
            'sentiment': 'negative',
            'confidence': crisis_confidence,
            'text': text,
            'crisis': True,
            'severity': 'CRITICAL',
            'requires_immediate_help': True
        }

    # Try VADER
    try:
        _ensure_nltk_data()
        from nltk.sentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = float(scores.get('compound', 0.0))
        
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Check if highly negative
        severity = 'severe' if compound < -0.5 else 'moderate' if compound < -0.2 else 'mild'
        
        return {
            'sentiment': label,
            'confidence': round(abs(compound), 3),
            'text': text,
            'crisis': False,
            'severity': severity if label == 'negative' else 'none'
        }
    except Exception:
        pass

    # Fallback: TextBlob
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = float(blob.sentiment.polarity)
        
        if polarity > 0.05:
            label = 'positive'
        elif polarity < -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        severity = 'severe' if polarity < -0.5 else 'moderate' if polarity < -0.2 else 'mild'
        
        return {
            'sentiment': label,
            'confidence': round(abs(polarity), 3),
            'text': text,
            'crisis': False,
            'severity': severity if label == 'negative' else 'none'
        }
    except Exception:
        return {
            'sentiment': 'neutral', 
            'confidence': 0.0, 
            'text': text, 
            'crisis': False
        }

def preprocess_text(text: str):
    """Apply preprocessing pipeline"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return " ".join(tokens)
    except Exception:
        try:
            _ensure_nltk_data()
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            import nltk
            try:
                nltk.download('wordnet')
                nltk.download('stopwords')
            except Exception:
                pass
            
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            return " ".join(tokens)
        except Exception:
            return text.lower().strip()

class BaselineMLModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.model_path = 'models/baseline_model.pkl'
        self.vectorizer_path = 'models/tfidf_vectorizer.pkl'
        
    def create_pipeline(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
    def train(self, texts, labels):
        if not os.path.exists('models'):
            os.makedirs('models')
        processed_texts = [preprocess_text(text) for text in texts]
        self.create_pipeline()
        self.pipeline.fit(processed_texts, labels)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        return self.evaluate(processed_texts, labels)
    
    def predict(self, text):
        if self.pipeline is None:
            self.load_model()
        processed_text = preprocess_text(text)
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
        return {
            'prediction': prediction,
            'confidence': float(max(probability)),
            'probabilities': probability.tolist()
        }
    
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            self.vectorizer = self.pipeline.named_steps['tfidf']
            self.model = self.pipeline.named_steps['classifier']
        except FileNotFoundError:
            raise Exception("Model not found. Please train the model first.")
    
    def evaluate(self, texts, labels):
        if self.pipeline is None:
            raise Exception("Model not trained yet.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(self.pipeline, texts, labels, cv=5)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'cross_validation_mean': float(cv_scores.mean()),
            'cross_validation_std': float(cv_scores.std()),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

baseline_model = BaselineMLModel()

def analyze_journal_entry(entry_id, text):
    """Enhanced analysis with crisis detection"""
    sentiment_result = analyze_sentiment(text)
    processed_text = preprocess_text(text)
    
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'processed_length': len(processed_text.split()),
        'sentiment': sentiment_result['sentiment'],
        'confidence': sentiment_result['confidence'],
        'crisis_detected': sentiment_result.get('crisis', False)
    }
    
    ml_prediction = "neutral"
    try:
        ml_prediction = baseline_model.predict(text)['prediction']
    except Exception:
        pass
    
    return {
        'entry_id': entry_id,
        'sentiment': sentiment_result['sentiment'],
        'emotions': [sentiment_result['sentiment']],
        'topics': extract_topics(processed_text),
        'features': features,
        'ml_prediction': ml_prediction,
        'crisis_detected': sentiment_result.get('crisis', False),
        'requires_escalation': sentiment_result.get('requires_immediate_help', False)
    }

def extract_topics(text):
    """Extract topics with mental health focus"""
    topics = []
    text_lower = text.lower()
    
    # Check for crisis first
    if any(keyword in text_lower for keyword in CRISIS_KEYWORDS):
        topics.append('CRISIS_SUICIDAL_IDEATION')
    
    topic_keywords = {
        'severe_depression': ['hopeless', 'worthless', 'meaningless', 'give up'],
        'anxiety': ['anxious', 'worry', 'stress', 'nervous', 'panic'],
        'depression': ['sad', 'depressed', 'tired', 'empty'],
        'sleep': ['sleep', 'insomnia', 'awake', 'rest'],
        'relationships': ['friend', 'family', 'partner', 'relationship'],
        'work': ['work', 'job', 'career', 'office'],
        'health': ['health', 'exercise', 'diet', 'physical']
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    
    return topics if topics else ['general']

def get_model_performance():
    return {
        'status': 'Model not trained yet',
        'message': 'Train the model with sample data to see performance metrics'
    }

def train_sample_model():
    sample_texts = [
        "I feel happy and excited about today",
        "I'm feeling really sad and hopeless",
        "I'm anxious about my upcoming presentation",
        "I had a great time with my friends",
        "I'm stressed about work deadlines",
        "I feel calm and peaceful",
        "I'm worried about my health",
        "I'm grateful for my family"
    ]
    
    sample_labels = [
        'positive', 'negative', 'negative', 'positive', 
        'negative', 'positive', 'negative', 'positive'
    ]
    
    try:
        results = baseline_model.train(sample_texts, sample_labels)
        return {
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': results
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }