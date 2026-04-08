"""
NLP Utilities for Public Complaints Analysis
This module contains functions for sentiment analysis, complaint categorization,
keyword extraction, and word cloud generation with Multilingual Support (English, Hindi, Hinglish).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from langdetect import detect, LangDetectException
# Lazy import for transformers - only load when needed
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from indicnlp.tokenize import indic_tokenize
import nltk
import re
import unicodedata
from collections import Counter
import warnings
import io
import base64
from fpdf import FPDF
from datetime import datetime

warnings.filterwarnings('ignore')

# Language detection cache to improve performance
_language_cache = {}

# Download required NLTK data and set up multilingual support
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Hindi stopwords (common words to filter)
HINDI_STOPWORDS = {
    'का', 'की', 'के', 'को', 'में', 'से', 'है', 'हैं', 'थे', 'था', 'थी', 'हो',
    'गया', 'गई', 'गए', 'रहा', 'रही', 'रहे', 'आप', 'मैं', 'मुझे', 'मेरा',
    'तुम', 'तुम्हारा', 'यह', 'वह', 'ये', 'वो', 'क्या', 'कौन', 'कहां',
    'कब', 'क्यों', 'कैसे', 'और', 'या', 'पर', 'बहुत', 'ज्यादा', 'कम',
    'अच्छा', 'बुरा', 'नहीं', 'हां', 'यदि', 'तो', 'जब', 'तक', 'सब',
    'सभी', 'कुछ', 'थोड़ा', 'ज्यादा', 'बिल्कुल', 'भी', 'ही', 'ने',
    'ने', 'को', 'एक', 'दो', 'तीन', 'चार', 'पांच'
}

# Hinglish common words mapping (Roman Hindi to meaning for sentiment)
HINGLISH_SENTIMENT_WORDS = {
    # Positive words
    'acha': 'good', 'achha': 'good', 'badhiya': 'excellent', 'badiya': 'excellent',
    'sahi': 'correct', 'mast': 'awesome', 'zabardast': 'amazing', 'shandar': 'wonderful',
    'khoobsurat': 'beautiful', 'pyara': 'lovely', 'best': 'best', 'nice': 'nice',
    'happy': 'happy', 'khush': 'happy', 'maza': 'fun', 'mazaa': 'fun',
    'shukriya': 'thank you', 'dhanyawad': 'thanks', 'thank': 'thank',
    'pasand': 'like', 'accha': 'good', 'bhadiya': 'excellent',
    # Negative words
    'bekar': 'useless', 'kharab': 'bad', 'ghatiya': 'terrible', 'bakwas': 'nonsense',
    'ganda': 'dirty', 'bura': 'bad', 'problem': 'problem', 'dikkat': 'problem',
    'pareshani': 'trouble', 'tension': 'tension', 'ghussa': 'anger', 'naraz': 'upset',
    'naraz': 'angry', 'fatigue': 'tired', 'thak': 'tired', 'bewakoof': 'foolish',
    'chor': 'thief', 'fraud': 'fraud', 'dhoka': 'cheat', 'cheating': 'cheating',
    'slow': 'slow', 'hang': 'hang', 'barbaad': 'ruined', 'lucha': 'rogue',
    'gussa': 'anger', 'dard': 'pain', 'takleef': 'difficulty', 'musibat': 'trouble'
}

# Urgency detection keywords for scoring
URGENCY_KEYWORDS = {
    # Critical - High severity (weight: 25-30 points each)
    'critical': 30, 'emergency': 30, 'urgent': 25, 'immediate': 25, 'life threatening': 30,
    'life-threatening': 30, 'danger': 25, 'hazardous': 25, 'severe': 25, 'extreme': 25,
    'heart attack': 30, 'stroke': 30, 'bleeding': 25, 'unconscious': 30, 'dying': 30,
    'death': 30, 'injury': 20, 'accident': 25, 'fire': 30, 'explosion': 30,
    'bomb': 30, 'terrorist': 30, 'gun': 25, 'weapon': 25, 'kill': 30,
    'murder': 30, 'suicide': 30, 'overdose': 30, 'poison': 25, 'electrocution': 30,
    'drowning': 30, 'choking': 30, 'asthma attack': 25, 'allergic reaction': 25,
    'anaphylaxis': 30, 'seizure': 25, 'convulsions': 25, 'coma': 30,
    
    # High - Serious issues (weight: 15-20 points each)
    'serious': 20, 'major': 18, 'significant': 15, 'widespread': 18, 'outage': 18,
    'blackout': 20, 'no power': 18, 'no water': 18, 'contaminated': 20, 'polluted': 18,
    'toxic': 20, 'gas leak': 25, 'chemical spill': 25, 'flood': 20, 'earthquake': 25,
    'storm': 18, 'hurricane': 25, 'tornado': 25, 'tsunami': 30, 'landslide': 25,
    'collapse': 25, 'structural damage': 20, 'evacuation': 22, 'trapped': 22,
    'stranded': 18, 'missing': 20, 'robbery': 20, 'theft': 15, 'assault': 20,
    'violence': 20, 'harassment': 18, 'fraud': 15, 'scam': 15, 'data breach': 18,
    'cyber attack': 20, 'hacking': 18, 'ransomware': 22, 'virus': 15, 'disease': 18,
    'epidemic': 22, 'pandemic': 25, 'infection': 18, 'contamination': 18,
    'broken pipe': 18, 'burst': 18, 'leak': 15, 'flooding': 20, 'short circuit': 18,
    'electrical fire': 25, 'sparking': 18, 'smoke': 18, 'burning': 20,
    
    # Medium - Important issues (weight: 8-14 points each)
    'important': 12, 'priority': 12, 'needed': 10, 'required': 10, 'essential': 12,
    'broken': 12, 'damaged': 10, 'not working': 12, 'failure': 12, 'error': 10,
    'defective': 12, 'malfunction': 12, 'out of order': 12, 'offline': 10,
    'disconnected': 10, 'blocked': 10, 'obstructed': 10, 'unsafe': 14, 'risk': 12,
    'hazard': 12, 'warning': 12, 'caution': 10, 'alert': 12, 'attention': 10,
    'delay': 10, 'late': 8, 'overdue': 12, 'deadline': 10, 'expired': 10,
    'cancelled': 10, 'postponed': 8, 'reschedule': 8, 'disruption': 10,
    'interruption': 10, 'inconvenience': 8, 'complaint': 10, 'grievance': 12,
    'dispute': 12, 'conflict': 12, 'argument': 10, 'fight': 12, 'threat': 14,
    'intimidation': 14, 'pressure': 10, 'stress': 10, 'anxiety': 10,
    'pain': 12, 'sick': 10, 'ill': 10, 'fever': 10, 'injury': 12, 'wound': 12,
    'bleeding': 15, 'cut': 8, 'bruise': 8, 'swelling': 10, 'infection': 12,
    'foul smell': 12, 'bad odor': 10, 'sewage': 14, 'waste': 10, 'garbage': 10,
    'noise': 8, 'loud': 8, 'disturbance': 10, 'vibration': 10, 'crack': 10,
    'leaking': 12, 'dripping': 10, 'overflow': 12, 'backup': 12, 'clogged': 12,
    'blocked drain': 12, 'sewer': 14, 'smell gas': 18, 'rotten egg': 15,
    
    # Low - Minor issues (weight: 3-7 points each)
    'minor': 5, 'small': 3, 'slight': 3, 'little': 3, 'tiny': 3, 'minimal': 3,
    'concern': 7, 'issue': 5, 'problem': 5, 'trouble': 6, 'difficulty': 5,
    'inconvenient': 5, 'annoying': 5, 'frustrating': 6, 'irritating': 5,
    'disappointing': 5, 'unsatisfactory': 6, 'poor': 5, 'bad': 5, 'worst': 7,
    'terrible': 7, 'awful': 7, 'horrible': 7, 'unacceptable': 7, 'inadequate': 5,
    'insufficient': 5, 'slow': 5, 'delayed': 5, 'waiting': 4, 'wait': 4,
    'pending': 4, 'processing': 3, 'queue': 3, 'backlog': 5, 'crowded': 5,
    'busy': 4, 'noisy': 5, 'dirty': 5, 'messy': 4, 'unclean': 5, 'dusty': 3,
    'smelly': 5, 'stinky': 5, 'uncomfortable': 5, 'cold': 4, 'hot': 4,
    'dark': 4, 'dim': 3, 'no light': 5, 'flickering': 4, 'weak': 4,
    'low pressure': 5, 'trickle': 4, 'discolored': 5, 'rusty': 4, 'stained': 4,
    'scratch': 3, 'dent': 3, 'chip': 3, 'crack': 5, 'loose': 4, 'wobbly': 4,
    'squeaky': 3, 'noisy': 5, 'loud': 5
}

# Issue type categories for urgency scoring
ISSUE_TYPE_WEIGHTS = {
    'health_medical': 25,    # Medical emergencies, health issues
    'safety_security': 25,   # Crime, safety hazards, security threats
    'utilities': 20,         # Power, water, gas outages
    'infrastructure': 18,    # Roads, bridges, buildings damage
    'environmental': 18,     # Pollution, contamination, natural disasters
    'technology': 15,        # System failures, cyber attacks
    'service_failure': 12,     # Service disruptions
    'financial_fraud': 15,   # Fraud, scams, financial crimes
    'general': 8            # General complaints
}

# Urgency level definitions
URGENCY_LEVELS = {
    'low': {'min': 0, 'max': 30, 'label': 'Low Priority', 'color': '#10b981'},
    'medium': {'min': 31, 'max': 60, 'label': 'Medium Priority', 'color': '#f59e0b'},
    'high': {'min': 61, 'max': 80, 'label': 'High Priority', 'color': '#ef4444'},
    'critical': {'min': 81, 'max': 100, 'label': 'Critical Priority', 'color': '#dc2626'}
}

# Department routing keywords
DEPARTMENT_KEYWORDS = {
    'water_supply': [
        'water', 'pani', 'pipe', 'leak', 'tap', 'supply', 'pressure', 'flow', 
        'burst', 'broken pipe', 'no water', 'dirty water', 'contaminated',
        'बूंद', 'पानी', 'नल', 'पाइप', 'लीक', 'टूटा हुआ', 'प्रेशर'
    ],
    'electricity': [
        'electricity', 'power', 'light', 'bill', 'meter', 'outage', 'blackout',
        'voltage', 'current', 'wire', 'short circuit', 'fuse', 'transformer',
        'बिजली', 'बत्ती', 'मीटर', 'वोल्टेज', 'तार', 'करंट'
    ],
    'roads': [
        'road', 'street', 'pathway', 'pothole', 'damage', 'broken', 'construction',
        'traffic', 'signal', 'sign', 'speed bump', 'divider', 'footpath',
        'सड़क', 'रास्ता', 'गड्ढा', 'टूटा हुआ', 'ट्रैफिक'
    ],
    'sanitation': [
        'garbage', 'waste', 'sewage', 'drain', 'cleaning', 'sweeper', 'dustbin',
        'litter', 'smell', 'odor', 'mosquito', 'hygiene', 'unclean',
        'कचरा', 'गंदगी', 'नाली', 'सफाई', 'गंदा', 'बदबू'
    ],
    'health': [
        'hospital', 'clinic', 'doctor', 'ambulance', 'medicine', 'emergency',
        'disease', 'fever', 'injury', 'patient', 'health', 'medical', 'covid',
        'अस्पताल', 'डॉक्टर', 'दवाई', 'बीमार', 'तबीयत', 'घायल'
    ],
    'police': [
        'police', 'crime', 'theft', 'robbery', 'assault', 'violence', 'fraud',
        'complaint', 'fir', 'missing', 'accident', 'safety', 'security',
        'पुलिस', 'चोरी', 'डकैती', 'अपराध', 'गुमशुदा', 'शिकायत'
    ],
    'education': [
        'school', 'college', 'university', 'teacher', 'student', 'education',
        'exam', 'admission', 'fee', 'scholarship', 'library', 'result',
        'स्कूल', 'कॉलेज', 'शिक्षक', 'छात्र', 'परीक्षा', 'फीस'
    ],
    'municipal': [
        'tax', 'property', 'building', 'license', 'permit', 'certificate',
        'birth', 'death', 'marriage', 'registration', 'zoning', 'illegal construction',
        'टैक्स', 'संपत्ति', 'इमारत', 'लाइसेंस', 'पंजीकरण', 'प्रमाण पत्र'
    ]
}

# Response templates by category and urgency
RESPONSE_TEMPLATES = {
    'critical': {
        'water_supply': "We acknowledge your urgent water supply complaint. Our emergency team has been dispatched and will arrive within 30 minutes. Reference ID: {ref_id}. For emergencies, call 1800-XXX-XXXX.",
        'electricity': "Critical power issue reported. Our rapid response team is en route. Expected resolution within 1 hour. Reference: {ref_id}. Emergency helpline: 1800-XXX-XXXX.",
        'health': "Emergency health complaint received. An ambulance/medical team has been alerted. Help is on the way! Reference: {ref_id}.",
        'police': "Your urgent safety complaint has been forwarded to the nearest police station. Officers will contact you within 15 minutes. Reference: {ref_id}. Emergency: 100",
        'default': "CRITICAL PRIORITY: Your complaint has been escalated to emergency response team. We will contact you within 15 minutes. Reference: {ref_id}"
    },
    'high': {
        'water_supply': "High priority water issue logged. Technician assigned and will visit within 4 hours. Reference ID: {ref_id}. Track status on our portal.",
        'electricity': "Power complaint registered with high priority. Maintenance crew scheduled for today. Reference: {ref_id}. Expected resolution: 4-6 hours.",
        'roads': "Road damage complaint noted. Inspection team will assess within 24 hours. Reference: {ref_id}. Thank you for helping improve our infrastructure.",
        'sanitation': "Sanitation issue marked urgent. Cleaning crew will address this within 4 hours. Reference: {ref_id}.",
        'default': "HIGH PRIORITY: Your complaint has been assigned to department experts. Resolution expected within 24 hours. Reference: {ref_id}"
    },
    'medium': {
        'water_supply': "Water supply complaint registered. Our team will investigate within 2 business days. Reference ID: {ref_id}.",
        'electricity': "Electrical issue logged. Technician will visit within 48 hours. Reference: {ref_id}.",
        'roads': "Road maintenance request received. Added to repair schedule. Reference: {ref_id}. ETA: 3-5 days.",
        'sanitation': "Sanitation concern noted. Will be addressed within 48 hours. Reference: {ref_id}.",
        'municipal': "Your municipal service request has been received. Processing time: 3-5 business days. Reference: {ref_id}.",
        'default': "MEDIUM PRIORITY: Complaint registered. Department will review and respond within 2-3 business days. Reference: {ref_id}"
    },
    'low': {
        'default': "Thank you for your feedback. Your suggestion has been recorded and will be reviewed during our next improvement cycle. Reference: {ref_id}"
    }
}

# Manual priority override storage (in-memory for now, could be persistent)
_manual_overrides = {}


def detect_duplicate_complaints(texts, similarity_threshold=0.75):
    """
    Detect duplicate complaints using TF-IDF cosine similarity
    
    Args:
        texts (list): List of complaint texts
        similarity_threshold (float): Threshold for considering complaints as duplicates (0-1)
        
    Returns:
        list: List of duplicate groups (each group contains indices of similar complaints)
    """
    if len(texts) < 2:
        return []
    
    # Preprocess texts
    processed_texts = []
    for text in texts:
        if text and not pd.isna(text):
            text_info = preprocess_multilingual_text(text)
            processed_text = text_info.get('hindi_version', text_info['processed_text'])
            processed_texts.append(processed_text if processed_text.strip() else "")
        else:
            processed_texts.append("")
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicate groups
        duplicates = []
        visited = set()
        
        for i in range(len(texts)):
            if i in visited:
                continue
            
            group = [i]
            for j in range(i + 1, len(texts)):
                if j not in visited and similarity_matrix[i, j] >= similarity_threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:
                duplicates.append(group)
            
            visited.add(i)
        
        return duplicates
    except Exception as e:
        print(f"Error in duplicate detection: {e}")
        return []


def generate_complaint_summary(text, max_length=150):
    """
    Generate a one-line summary of the complaint using keyword extraction
    
    Args:
        text (str): Complaint text
        max_length (int): Maximum length of summary
        
    Returns:
        str: One-line summary
    """
    if not text or pd.isna(text):
        return "No summary available"
    
    # Extract keywords
    keywords = extract_multilingual_keywords([text], top_n=5)
    keyword_list = [kw[0] for kw in keywords]
    
    # Detect urgency level
    urgency = calculate_urgency_score(text)
    urgency_prefix = ""
    if urgency['urgency_level'] == 'critical':
        urgency_prefix = "URGENT: "
    elif urgency['urgency_level'] == 'high':
        urgency_prefix = "HIGH PRIORITY: "
    
    # Detect department
    dept = suggest_department(text)
    
    # Create summary
    if keyword_list:
        summary = f"{urgency_prefix}Issue related to {', '.join(keyword_list[:3])}"
        if dept:
            summary += f" | Department: {dept}"
    else:
        summary = text[:max_length] + "..." if len(text) > max_length else text
    
    return summary[:max_length]


def suggest_department(text):
    """
    Suggest department based on complaint text
    
    Args:
        text (str): Complaint text
        
    Returns:
        str: Suggested department name
    """
    if not text or pd.isna(text):
        return 'municipal'
    
    text_lower = str(text).lower()
    
    # Count matches for each department
    dept_scores = {}
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        if score > 0:
            dept_scores[dept] = score
    
    if dept_scores:
        # Return department with highest score
        return max(dept_scores, key=dept_scores.get)
    
    return 'municipal'  # Default department


def suggest_response(text, urgency_level='medium', department='default'):
    """
    Suggest AI-generated response based on complaint
    
    Args:
        text (str): Complaint text
        urgency_level (str): Urgency level (critical/high/medium/low)
        department (str): Department handling the complaint
        
    Returns:
        str: Suggested response message
    """
    import random
    
    # Generate reference ID
    ref_id = f"CMP{random.randint(100000, 999999)}"
    
    # Get templates for urgency level
    templates = RESPONSE_TEMPLATES.get(urgency_level, RESPONSE_TEMPLATES['medium'])
    
    # Get template for department or use default
    template = templates.get(department, templates.get('default', RESPONSE_TEMPLATES['medium']['default']))
    
    # Fill in reference ID
    response = template.format(ref_id=ref_id)
    
    return response


def set_manual_priority_override(complaint_id, new_priority, reason=''):
    """
    Allow officers to manually override priority
    
    Args:
        complaint_id (str): Unique complaint identifier
        new_priority (str): New priority level (critical/high/medium/low)
        reason (str): Reason for override
        
    Returns:
        dict: Override confirmation
    """
    global _manual_overrides
    
    _manual_overrides[complaint_id] = {
        'priority': new_priority,
        'reason': reason,
        'timestamp': datetime.now().isoformat(),
        'original_priority': None  # Will be set when checking
    }
    
    return {
        'complaint_id': complaint_id,
        'new_priority': new_priority,
        'reason': reason,
        'status': 'overridden'
    }


def get_priority_with_override(complaint_id, auto_detected_priority):
    """
    Get final priority considering manual override
    
    Args:
        complaint_id (str): Unique complaint identifier
        auto_detected_priority (str): Auto-detected priority level
        
    Returns:
        dict: Final priority information
    """
    global _manual_overrides
    
    if complaint_id in _manual_overrides:
        override = _manual_overrides[complaint_id]
        override['original_priority'] = auto_detected_priority
        return {
            'priority': override['priority'],
            'is_override': True,
            'original_priority': auto_detected_priority,
            'reason': override['reason'],
            'timestamp': override['timestamp']
        }
    
    return {
        'priority': auto_detected_priority,
        'is_override': False,
        'original_priority': auto_detected_priority,
        'reason': '',
        'timestamp': None
    }


def batch_department_routing(texts):
    """
    Suggest department routing for multiple complaints
    
    Args:
        texts (list): List of complaint texts
        
    Returns:
        pd.DataFrame: DataFrame with routing suggestions
    """
    results = []
    
    for text in texts:
        dept = suggest_department(text)
        summary = generate_complaint_summary(text)
        response = suggest_response(text)
        
        results.append({
            'text': text,
            'suggested_department': dept,
            'department_display': dept.replace('_', ' ').title(),
            'summary': summary,
            'suggested_response': response
        })
    
    return pd.DataFrame(results)


def create_priority_queue(df, text_column, urgency_results):
    """
    Create a priority-based complaint queue sorted by urgency score
    
    Args:
        df (pd.DataFrame): Original dataframe with complaints
        text_column (str): Name of the text column
        urgency_results (pd.DataFrame): Urgency analysis results
        
    Returns:
        pd.DataFrame: Sorted priority queue with highest urgency first
    """
    # Merge original data with urgency results
    queue_df = df.copy()
    queue_df['urgency_score'] = urgency_results['urgency_score']
    queue_df['urgency_level'] = urgency_results['urgency_level']
    queue_df['priority_label'] = urgency_results['priority_label']
    queue_df['priority_color'] = urgency_results['priority_color']
    queue_df['matched_keywords'] = urgency_results['matched_keywords']
    
    # Sort by urgency score descending (highest urgency first)
    queue_df = queue_df.sort_values('urgency_score', ascending=False)
    
    # Add queue position
    queue_df = queue_df.reset_index(drop=True)
    queue_df['queue_position'] = range(1, len(queue_df) + 1)
    
    return queue_df


def get_priority_queue_stats(queue_df):
    """
    Get statistics for the priority queue
    
    Args:
        queue_df (pd.DataFrame): Priority queue dataframe
        
    Returns:
        dict: Queue statistics
    """
    total = len(queue_df)
    
    # Count by urgency level
    critical_count = len(queue_df[queue_df['urgency_level'] == 'critical'])
    high_count = len(queue_df[queue_df['urgency_level'] == 'high'])
    medium_count = len(queue_df[queue_df['urgency_level'] == 'medium'])
    low_count = len(queue_df[queue_df['urgency_level'] == 'low'])
    
    # Average urgency score
    avg_urgency = queue_df['urgency_score'].mean()
    
    # Top urgent complaints
    top_urgent = queue_df.head(5)[['queue_position', 'urgency_score', 'priority_label']].to_dict('records')
    
    return {
        'total_complaints': total,
        'critical_count': critical_count,
        'high_count': high_count,
        'medium_count': medium_count,
        'low_count': low_count,
        'average_urgency_score': round(avg_urgency, 2),
        'top_urgent_complaints': top_urgent
    }


# Multilingual sentiment model (lazy loading)
_multilingual_sentiment_model = None


def get_multilingual_sentiment_model():
    """Lazy load the multilingual sentiment model with caching"""
    global _multilingual_sentiment_model
    if _multilingual_sentiment_model is None:
        try:
            import os
            
            # Skip heavy model on cloud deployment (Render free tier can't handle 500MB model)
            if os.environ.get('SKIP_HEAVY_MODEL', 'false').lower() == 'true':
                print("Skipping heavy transformer model (cloud mode)")
                _multilingual_sentiment_model = False
                return _multilingual_sentiment_model
            
            # Import here to avoid slow startup
            from transformers import pipeline
            
            # Disable progress bars to avoid clutter
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            
            # Use a lightweight multilingual model that supports English and Indic languages
            model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
            
            # Use caching for model to avoid re-downloading
            _multilingual_sentiment_model = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=-1,  # CPU
                batch_size=1,  # Smaller batches for memory efficiency
                truncation=True,
                max_length=512  # Limit sequence length for speed
            )
        except Exception as e:
            print(f"Warning: Could not load multilingual model: {e}")
            _multilingual_sentiment_model = False
    return _multilingual_sentiment_model


def clear_model_cache():
    """Clear the model cache to free memory"""
    global _multilingual_sentiment_model
    _multilingual_sentiment_model = None
    import gc
    gc.collect()


def load_complaint_data(file_path):
    """
    Load complaint data from CSV or Excel file
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")
    
    return df


def detect_language(text):
    """
    Detect the language of the text (English, Hindi, Hinglish, or Mixed)
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Detected language code ('en', 'hi', 'hinglish', 'mixed')
    """
    if not text or pd.isna(text):
        return 'en'
    
    text = str(text).strip()
    if not text:
        return 'en'
    
    # Check cache first
    cache_key = hash(text[:100])  # Use first 100 chars for cache key
    if cache_key in _language_cache:
        return _language_cache[cache_key]
    
    # Check for Devanagari script (Hindi)
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    devanagari_count = len(devanagari_pattern.findall(text))
    
    # Check for Hinglish patterns (Roman Hindi)
    hinglish_indicators = [
        'hai', 'hain', 'tha', 'thi', 'the', 'kar', 'raha', 'rahi', 'rahe',
        'gaya', 'gya', 'gyi', 'gayi', 'gye', 'gaye', 'dia', 'diya', 'dya',
        'nahi', 'nh', 'nhi', 'kya', 'kaun', 'kaha', 'kyu', 'kaise', 'kaisi',
        'acha', 'accha', 'achha', 'bura', 'bekar', 'kharab', 'thik', 'theek',
        'bahut', 'bohot', 'jyada', 'zada', 'kam', 'sab', 'sabhi', 'kuch',
        'mera', 'tera', 'apna', 'iska', 'uska', 'iska', 'unska', 'apke',
        'mein', 'mai', 'mujhe', 'tujhe', 'usse', 'isse', 'unse', 'aap'
    ]
    
    # Count Hinglish indicators
    text_lower = text.lower()
    hinglish_count = sum(1 for word in hinglish_indicators if word in text_lower)
    
    # Check for English words
    english_words_pattern = re.compile(r'\b[a-zA-Z]+\b')
    english_words = english_words_pattern.findall(text)
    english_count = len(english_words)
    
    total_chars = len(text.replace(' ', ''))
    
    # Decision logic
    if devanagari_count > 0:
        # Contains Devanagari script
        if english_count > 2:
            result = 'mixed'  # Mixed Hindi-English
        else:
            result = 'hi'  # Pure Hindi
    elif hinglish_count >= 2 or hinglish_count > english_count * 0.3:
        # Contains significant Hinglish patterns
        result = 'hinglish'
    else:
        # Try langdetect for additional verification
        try:
            detected = detect(text)
            if detected == 'en':
                result = 'en'
            else:
                # If langdetect says non-English but no Hinglish patterns,
                # it might be pure Hindi or other language
                result = 'en' if english_count > len(text.split()) * 0.5 else 'hi'
        except LangDetectException:
            result = 'en' if english_count > len(text.split()) * 0.5 else 'hinglish'
    
    _language_cache[cache_key] = result
    return result


def transliterate_hinglish_to_hindi(text):
    """
    Simple transliteration from Hinglish (Roman Hindi) to Hindi script
    This is a basic implementation - for production, consider using a proper
    transliteration library like indic-trans
    
    Args:
        text (str): Hinglish text
        
    Returns:
        str: Hindi transliterated text (or original if transliteration fails)
    """
    # Common Hinglish to Hindi mappings
    hinglish_hindi_map = {
        'hai': 'है', 'hain': 'हैं', 'tha': 'था', 'thi': 'थी', 'the': 'थे',
        'kar': 'कर', 'karo': 'करो', 'kiya': 'किया', 'kiye': 'किए',
        'raha': 'रहा', 'rahi': 'रही', 'rahe': 'रहे', 'raho': 'रहो',
        'gaya': 'गया', 'gayi': 'गई', 'gya': 'गया', 'gyi': 'गई', 'gye': 'गए',
        'dia': 'दिया', 'diya': 'दिया', 'dya': 'दिया', 'diye': 'दिए',
        'nahi': 'नहीं', 'nhi': 'नहीं', 'nh': 'नहीं',
        'kya': 'क्या', 'kaun': 'कौन', 'kaha': 'कहाँ', 'kahan': 'कहाँ',
        'kyu': 'क्यों', 'kyon': 'क्यों', 'kaise': 'कैसे', 'kaisi': 'कैसी',
        'acha': 'अच्छा', 'accha': 'अच्छा', 'achha': 'अच्छा',
        'achi': 'अच्छी', 'ache': 'अच्छे', 'bura': 'बुरा', 'buri': 'बुरी',
        'bekar': 'बेकार', 'kharab': 'खराब', 'thik': 'ठीक', 'theek': 'ठीक',
        'bahut': 'बहुत', 'bohot': 'बहुत', 'jyada': 'ज्यादा', 'zada': 'ज़्यादा',
        'kam': 'कम', 'sab': 'सब', 'sabhi': 'सभी', 'kuch': 'कुछ',
        'mera': 'मेरा', 'meri': 'मेरी', 'mere': 'मेरे',
        'tera': 'तेरा', 'teri': 'तेरी', 'tere': 'तेरे',
        'apna': 'अपना', 'apni': 'अपनी', 'apne': 'अपने',
        'iska': 'इसका', 'iski': 'इसकी', 'iske': 'इसके',
        'uska': 'उसका', 'uski': 'उसकी', 'uske': 'उसके',
        'mein': 'में', 'mai': 'मैं', 'main': 'मैं',
        'mujhe': 'मुझे', 'tujhe': 'तुझे', 'use': 'उसे', 'ise': 'इसे',
        'hum': 'हम', 'tum': 'तुम', 'aap': 'आप',
        'sahi': 'सही', 'galat': 'गलत', 'haan': 'हाँ', 'hn': 'हाँ',
        'paisa': 'पैसा', 'paise': 'पैसे', 'rupee': 'रुपये', 'rupaye': 'रुपये',
        'samay': 'समय', 'waqt': 'वक्त', 'din': 'दिन', 'raat': 'रात',
        'kaam': 'काम', 'kaaj': 'काज', 'seva': 'सेवा',
        'shikayat': 'शिकायत', 'shikayat': 'शिकायत', 'problem': 'समस्या',
        'dikkat': 'दिक्कत', 'pareshani': 'परेशानी', 'tension': 'टेंशन'
    }
    
    words = text.lower().split()
    hindi_words = []
    
    for word in words:
        # Remove punctuation for lookup
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in hinglish_hindi_map:
            hindi_words.append(hinglish_hindi_map[clean_word])
        else:
            # Keep original word if no mapping found
            hindi_words.append(word)
    
    return ' '.join(hindi_words)


def preprocess_multilingual_text(text, language=None):
    """
    Clean and preprocess text data for multilingual support
    
    Args:
        text (str): Raw text to preprocess
        language (str): Known language code ('en', 'hi', 'hinglish', 'mixed')
        
    Returns:
        dict: Dictionary with 'processed_text', 'original_text', 'language', 'hindi_version'
    """
    if pd.isna(text):
        return {
            'processed_text': "",
            'original_text': "",
            'language': 'en',
            'hindi_version': ""
        }
    
    original_text = str(text).strip()
    if not original_text:
        return {
            'processed_text': "",
            'original_text': "",
            'language': 'en',
            'hindi_version': ""
        }
    
    # Detect language if not provided
    if language is None:
        language = detect_language(original_text)
    
    processed_text = original_text
    hindi_version = ""
    
    if language == 'en':
        # English preprocessing (existing logic)
        processed_text = re.sub(r'[^a-zA-Z\s]', '', original_text.lower())
        processed_text = ' '.join(processed_text.split())
        hindi_version = processed_text  # For sentiment analysis
        
    elif language == 'hi':
        # Hindi preprocessing - keep Devanagari, remove Latin and special chars
        # Remove URLs, emails, and special characters but keep Hindi script
        processed_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', original_text)
        processed_text = re.sub(r'\S+@\S+', '', processed_text)
        # Keep Devanagari and basic punctuation
        processed_text = re.sub(r'[^\u0900-\u097F\s।,!?.]', ' ', processed_text)
        processed_text = ' '.join(processed_text.split())
        hindi_version = processed_text
        
    elif language in ['hinglish', 'mixed']:
        # Hinglish preprocessing - transliterate to Hindi for better analysis
        processed_text = original_text.lower()
        # Remove special chars but keep letters and spaces
        processed_text = re.sub(r'[^a-zA-Z\u0900-\u097F\s]', ' ', processed_text)
        processed_text = ' '.join(processed_text.split())
        
        # Create Hindi version for sentiment analysis
        try:
            hindi_version = transliterate_hinglish_to_hindi(processed_text)
        except:
            hindi_version = processed_text
    
    return {
        'processed_text': processed_text,
        'original_text': original_text,
        'language': language,
        'hindi_version': hindi_version if hindi_version else processed_text
    }


def analyze_multilingual_sentiment(text_info):
    """
    Perform sentiment analysis supporting English, Hindi, and Hinglish
    Optimized for cloud deployment with fallback
    
    Args:
        text_info (dict or str): Either a dict from preprocess_multilingual_text or raw text
        
    Returns:
        dict: Sentiment results with polarity, subjectivity, sentiment_label, and language
    """
    if isinstance(text_info, str):
        text_info = preprocess_multilingual_text(text_info)
    
    original_text = text_info['original_text']
    language = text_info['language']
    
    if not original_text:
        return {
            'polarity': 0,
            'subjectivity': 0,
            'sentiment': 'neutral',
            'language': language,
            'confidence': 0
        }
    
    # Get multilingual model (with timeout protection)
    try:
        model = get_multilingual_sentiment_model()
    except Exception as e:
        print(f"Model loading failed, using fallback: {e}")
        model = False
    
    if model and model != False:
        try:
            # Use multilingual BERT model for sentiment
            # For Hinglish and Hindi, use the Hindi version
            text_to_analyze = text_info.get('hindi_version', original_text)
            
            # Truncate if too long
            if len(text_to_analyze) > 512:
                text_to_analyze = text_to_analyze[:512]
            
            result = model(text_to_analyze)[0]
            
            label = result['label'].lower()
            confidence = result['score']
            
            # Map to our format
            if 'positive' in label:
                sentiment_label = 'positive'
                polarity = confidence
            elif 'negative' in label:
                sentiment_label = 'negative'
                polarity = -confidence
            else:
                sentiment_label = 'neutral'
                polarity = 0
            
            return {
                'polarity': polarity,
                'subjectivity': confidence,
                'sentiment': sentiment_label,
                'language': language,
                'confidence': confidence,
                'model_label': label
            }
        except Exception as e:
            # Fall back to rule-based approach
            pass
    
    # Fallback: Rule-based sentiment analysis for Hinglish
    if language in ['hinglish', 'mixed']:
        text_lower = original_text.lower()
        positive_score = 0
        negative_score = 0
        
        # Check Hinglish sentiment words
        for hinglish_word, english_meaning in HINGLISH_SENTIMENT_WORDS.items():
            count = text_lower.count(hinglish_word)
            if count > 0:
                if english_meaning in ['good', 'excellent', 'correct', 'awesome', 'amazing', 'wonderful', 
                                      'beautiful', 'lovely', 'best', 'nice', 'happy', 'fun', 'thank you',
                                      'thanks', 'thank', 'like']:
                    positive_score += count
                elif english_meaning in ['useless', 'bad', 'terrible', 'nonsense', 'dirty', 'problem',
                                        'trouble', 'tension', 'anger', 'upset', 'angry', 'tired', 'foolish',
                                        'thief', 'fraud', 'cheat', 'cheating', 'slow', 'hang', 'ruined',
                                        'rogue', 'pain', 'difficulty']:
                    negative_score += count
        
        # Also check for English sentiment words
        try:
            blob = TextBlob(original_text)
            english_polarity = blob.sentiment.polarity
            if english_polarity > 0.1:
                positive_score += english_polarity * 2
            elif english_polarity < -0.1:
                negative_score += abs(english_polarity) * 2
        except:
            pass
        
        total_score = positive_score + negative_score
        if total_score == 0:
            polarity = 0
            sentiment_label = 'neutral'
        else:
            polarity = (positive_score - negative_score) / total_score
            if polarity > 0.1:
                sentiment_label = 'positive'
            elif polarity < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': min(total_score / 5, 1.0),
            'sentiment': sentiment_label,
            'language': language,
            'confidence': abs(polarity)
        }
    
    # For pure English or Hindi, use TextBlob as fallback
    try:
        # For Hindi, transliterated text might work better with TextBlob
        text_for_blob = text_info.get('hindi_version', original_text)
        blob = TextBlob(text_for_blob)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment_label = 'positive'
        elif polarity < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment_label,
            'language': language,
            'confidence': abs(polarity)
        }
    except:
        return {
            'polarity': 0,
            'subjectivity': 0,
            'sentiment': 'neutral',
            'language': language,
            'confidence': 0
        }


def calculate_urgency_score(text, sentiment_info=None):
    """
    Calculate urgency score (0-100) for a complaint based on NLP analysis
    
    Args:
        text (str): Complaint text
        sentiment_info (dict): Pre-computed sentiment information (optional)
        
    Returns:
        dict: Urgency score and level information
    """
    if not text or pd.isna(text):
        return {
            'urgency_score': 0,
            'urgency_level': 'low',
            'priority_label': 'Low Priority',
            'priority_color': '#10b981',
            'factors': {}
        }
    
    text_lower = str(text).lower()
    
    # Initialize score components
    keyword_score = 0
    matched_keywords = []
    
    # Calculate keyword-based urgency score
    for keyword, weight in URGENCY_KEYWORDS.items():
        if keyword in text_lower:
            keyword_score += weight
            matched_keywords.append((keyword, weight))
    
    # Cap keyword score at 70 to leave room for sentiment adjustment
    keyword_score = min(keyword_score, 70)
    
    # Get sentiment information if not provided
    if sentiment_info is None:
        text_info = preprocess_multilingual_text(text)
        sentiment_info = analyze_multilingual_sentiment(text_info)
    
    # Sentiment-based adjustment
    sentiment_score = 0
    sentiment_factor = sentiment_info.get('sentiment', 'neutral')
    polarity = abs(sentiment_info.get('polarity', 0))
    
    # Negative sentiment increases urgency
    if sentiment_factor == 'negative':
        sentiment_score = 15 + (polarity * 10)  # 15-25 points based on negativity
    elif sentiment_factor == 'neutral':
        sentiment_score = 5
    else:  # positive sentiment reduces urgency
        sentiment_score = 0
    
    # Language factor - some languages might indicate local urgency patterns
    language = sentiment_info.get('language', 'en')
    language_factor = 0  # Could be customized based on region-specific patterns
    
    # Calculate total urgency score
    total_score = keyword_score + sentiment_score + language_factor
    
    # Cap at 100
    total_score = min(total_score, 100)
    
    # Determine urgency level
    urgency_level = 'low'
    priority_label = 'Low Priority'
    priority_color = '#10b981'
    
    for level, config in URGENCY_LEVELS.items():
        if config['min'] <= total_score <= config['max']:
            urgency_level = level
            priority_label = config['label']
            priority_color = config['color']
            break
    
    return {
        'urgency_score': round(total_score, 1),
        'urgency_level': urgency_level,
        'priority_label': priority_label,
        'priority_color': priority_color,
        'factors': {
            'keyword_score': keyword_score,
            'sentiment_score': sentiment_score,
            'matched_keywords': matched_keywords[:5],  # Top 5 matched keywords
            'sentiment': sentiment_factor,
            'polarity': polarity
        }
    }


def batch_urgency_analysis(texts, sentiment_results=None):
    """
    Perform urgency analysis on a list of texts
    
    Args:
        texts (list): List of complaint texts
        sentiment_results (pd.DataFrame): Pre-computed sentiment results (optional)
        
    Returns:
        pd.DataFrame: DataFrame with urgency scores for each text
    """
    results = []
    
    for i, text in enumerate(texts):
        # Use pre-computed sentiment if available
        sentiment_info = None
        if sentiment_results is not None and i < len(sentiment_results):
            sentiment_info = sentiment_results.iloc[i].to_dict()
        
        urgency = calculate_urgency_score(text, sentiment_info)
        results.append({
            'text': text,
            'urgency_score': urgency['urgency_score'],
            'urgency_level': urgency['urgency_level'],
            'priority_label': urgency['priority_label'],
            'priority_color': urgency['priority_color'],
            'matched_keywords': urgency['factors']['matched_keywords'],
            'urgency_keyword_score': urgency['factors']['keyword_score'],
            'urgency_sentiment_score': urgency['factors']['sentiment_score']
        })
    
    return pd.DataFrame(results)


def get_urgency_distribution(urgency_df):
    """
    Get distribution of urgency levels
    
    Args:
        urgency_df (pd.DataFrame): DataFrame with urgency analysis results
        
    Returns:
        dict: Distribution of urgency levels
    """
    distribution = urgency_df['urgency_level'].value_counts().to_dict()
    
    # Ensure all levels are represented
    for level in ['critical', 'high', 'medium', 'low']:
        if level not in distribution:
            distribution[level] = 0
    
    return distribution


def batch_multilingual_sentiment_analysis(texts):
    """
    Perform multilingual sentiment analysis on a list of texts
    
    Args:
        texts (list): List of texts to analyze
        
    Returns:
        pd.DataFrame: DataFrame with sentiment results including language detection
    """
    results = []
    for text in texts:
        text_info = preprocess_multilingual_text(text)
        sentiment = analyze_multilingual_sentiment(text_info)
        results.append({
            'text': text,
            'language': sentiment['language'],
            'polarity': sentiment['polarity'],
            'subjectivity': sentiment['subjectivity'],
            'sentiment': sentiment['sentiment'],
            'confidence': sentiment['confidence']
        })
    return pd.DataFrame(results)


def categorize_complaints(texts, n_categories=5):
    """
    Categorize complaints using K-means clustering on TF-IDF vectors
    Supports multilingual texts (English, Hindi, Hinglish)
    
    Args:
        texts (list): List of complaint texts
        n_categories (int): Number of categories to create
        
    Returns:
        tuple: (clusters, cluster_labels, vectorizer, kmeans_model)
    """
    # Preprocess texts with multilingual support
    processed_texts = []
    for text in texts:
        text_info = preprocess_multilingual_text(text)
        # Use processed text - for Hinglish/Hindi use Hindi version for better clustering
        processed_text = text_info.get('hindi_version', text_info['processed_text'])
        if processed_text.strip():
            processed_texts.append(processed_text)
    
    # Remove empty texts
    processed_texts = [text for text in processed_texts if text.strip()]
    
    if len(processed_texts) < n_categories:
        n_categories = max(1, len(processed_texts))
    
    if len(processed_texts) == 0:
        # Return empty results if no valid texts
        return np.array([]), [], None, None
    
    # Create TF-IDF vectors with multilingual support
    # Use a larger max_features to capture multilingual vocabulary
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words='english',  # English stopwords
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_categories, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Get top terms for each cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_labels = []
        
        for i in range(n_categories):
            top_indices = kmeans.cluster_centers_[i].argsort()[-3:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            cluster_labels.append(' '.join(top_terms))
        
        return clusters, cluster_labels, vectorizer, kmeans
    except Exception as e:
        print(f"Error in categorization: {e}")
        return np.array([0] * len(texts)), ['General Category'], None, None


def extract_multilingual_keywords(texts, top_n=20):
    """
    Extract top keywords using TF-IDF with multilingual support
    
    Args:
        texts (list): List of texts
        top_n (int): Number of top keywords to return
        
    Returns:
        list: List of (keyword, score) tuples - backward compatible format
    """
    # Preprocess texts with multilingual support
    processed_texts = []
    
    for text in texts:
        if text and not pd.isna(text):
            text_info = preprocess_multilingual_text(text)
            # For Hinglish, use original processed text (English characters work better with TF-IDF)
            # For Hindi, use the Devanagari script directly
            if text_info['language'] == 'hinglish':
                processed_text = text_info['processed_text']
            else:
                processed_text = text_info.get('hindi_version', text_info['processed_text'])
            if processed_text.strip():
                processed_texts.append(processed_text)
    
    if not processed_texts:
        return []
    
    # Create TF-IDF vectors with better parameters for short texts
    vectorizer = TfidfVectorizer(
        max_features=top_n * 2,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True  # Use sublinear tf scaling for better results on short texts
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        
        # Create keyword-score pairs
        keyword_scores = list(zip(feature_names, tfidf_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores[:top_n]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        # Return empty list on error
        return []


def generate_multilingual_wordcloud(texts, width=800, height=400):
    """
    Generate word cloud from complaint texts with multilingual support
    
    Args:
        texts (list): List of texts
        width (int): Word cloud width
        height (int): Word cloud height
        
    Returns:
        WordCloud: Generated word cloud object
    """
    # Combine all texts with multilingual preprocessing
    processed_texts = []
    for text in texts:
        if text and not pd.isna(text):
            text_info = preprocess_multilingual_text(text)
            # Use Hindi version for better word cloud if available
            processed_text = text_info.get('hindi_version', text_info['processed_text'])
            if processed_text.strip():
                processed_texts.append(processed_text)
    
    combined_text = ' '.join(processed_texts)
    
    if not combined_text.strip():
        return None
    
    # Custom stopwords for multilingual support
    multilingual_stopwords = set([
        'complaint', 'issue', 'problem', 'service', 'customer',
        'का', 'की', 'के', 'को', 'में', 'से', 'है', 'हैं', 'और', 'यह', 'वह',
        'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'mein', 'me', 'se', 'aur', 'ye', 'vo'
    ])
    
    # Create word cloud with multilingual font support
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            stopwords=multilingual_stopwords,
            max_words=100,
            colormap='viridis',
            font_path=None,  # Use default font (may not render Hindi perfectly)
            random_state=42
        ).generate(combined_text)
        
        return wordcloud
    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return None


def complete_multilingual_nlp_analysis(df, text_column):
    """
    Perform complete NLP analysis on complaint data with multilingual support
    Includes urgency scoring (0-100), duplicate detection, summarization, routing
    Optimized for cloud deployment with error handling
    
    Args:
        df (pd.DataFrame): DataFrame containing complaint data
        text_column (str): Name of the column containing complaint text
        
    Returns:
        dict: Dictionary containing all analysis results
    """
    results = {}
    
    # Get text data
    texts = df[text_column].fillna('').tolist()
    
    # 1. Multilingual Sentiment Analysis (with fallback)
    try:
        sentiment_results = batch_multilingual_sentiment_analysis(texts)
    except Exception as e:
        print(f"Sentiment analysis error: {e}. Using fallback.")
        # Create basic sentiment results
        sentiment_results = pd.DataFrame({
            'polarity': [0] * len(texts),
            'subjectivity': [0] * len(texts),
            'sentiment': ['neutral'] * len(texts),
            'language': ['en'] * len(texts),
            'confidence': [0] * len(texts)
        })
    results['sentiment'] = sentiment_results
    
    # 2. Language Distribution
    try:
        language_counts = sentiment_results['language'].value_counts().to_dict()
        results['language_distribution'] = language_counts
    except:
        results['language_distribution'] = {'en': len(texts)}
    
    # 3. Urgency Analysis
    try:
        urgency_results = batch_urgency_analysis(texts, sentiment_results)
        results['urgency'] = urgency_results
    except Exception as e:
        print(f"Urgency analysis error: {e}")
        # Create basic urgency results
        urgency_results = pd.DataFrame({
            'urgency_score': [0] * len(texts),
            'urgency_level': ['low'] * len(texts),
            'priority_label': ['Low Priority'] * len(texts),
            'priority_color': ['#10b981'] * len(texts),
            'matched_keywords': [[] for _ in range(len(texts))]
        })
        results['urgency'] = urgency_results
    
    # 4. Urgency Distribution
    try:
        urgency_distribution = get_urgency_distribution(urgency_results)
        results['urgency_distribution'] = urgency_distribution
    except:
        results['urgency_distribution'] = {'low': len(texts), 'medium': 0, 'high': 0, 'critical': 0}
    
    # 5. Duplicate Detection (skip for large datasets in cloud)
    try:
        if len(texts) <= 500:  # Only for smaller datasets
            duplicates = detect_duplicate_complaints(texts, similarity_threshold=0.75)
            results['duplicates'] = duplicates
        else:
            results['duplicates'] = []
    except Exception as e:
        print(f"Duplicate detection error: {e}")
        results['duplicates'] = []
    
    # 6. Complaint Summaries & Department Routing
    try:
        routing_results = batch_department_routing(texts)
        results['routing'] = routing_results
    except Exception as e:
        print(f"Routing error: {e}")
        # Create basic routing results
        routing_results = pd.DataFrame({
            'text': texts,
            'suggested_department': ['general'] * len(texts),
            'department_display': ['General'] * len(texts),
            'summary': texts[:100] if texts else [''] * len(texts),
            'suggested_response': ['We have received your complaint and will look into it.'] * len(texts)
        })
        results['routing'] = routing_results
    
    # 7. Complaint Categorization
    try:
        clusters, cluster_labels, vectorizer, kmeans = categorize_complaints(texts)
        results['categories'] = {
            'clusters': clusters,
            'labels': cluster_labels,
            'vectorizer': vectorizer,
            'model': kmeans
        }
    except Exception as e:
        print(f"Categorization error: {e}")
        # Create basic categories
        results['categories'] = {
            'clusters': [0] * len(texts),
            'labels': ['General'],
            'vectorizer': None,
            'model': None
        }
    
    # 8. Multilingual Keyword Extraction
    try:
        keywords = extract_multilingual_keywords(texts)
        results['keywords'] = keywords if keywords else [('complaint', 1.0)]
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        results['keywords'] = [('complaint', 1.0)]
    
    # 9. Multilingual Word Cloud
    try:
        wordcloud = generate_multilingual_wordcloud(texts)
        results['wordcloud'] = wordcloud
    except Exception as e:
        print(f"Wordcloud error: {e}")
        results['wordcloud'] = None
    
    return results


# Legacy function aliases for backward compatibility
def preprocess_text(text):
    """Legacy function - use preprocess_multilingual_text instead"""
    result = preprocess_multilingual_text(text)
    return result['processed_text']

def analyze_sentiment(text):
    """Legacy function - use analyze_multilingual_sentiment instead"""
    result = analyze_multilingual_sentiment(text)
    return result['polarity'], result['subjectivity'], result['sentiment']

def batch_sentiment_analysis(texts):
    """Legacy function - use batch_multilingual_sentiment_analysis instead"""
    return batch_multilingual_sentiment_analysis(texts)

def extract_keywords(texts, top_n=20):
    """Legacy function - use extract_multilingual_keywords instead"""
    return extract_multilingual_keywords(texts, top_n)

def generate_wordcloud(texts, width=800, height=400):
    """Legacy function - use generate_multilingual_wordcloud instead"""
    return generate_multilingual_wordcloud(texts, width, height)

def complete_nlp_analysis(df, text_column):
    """Legacy function - use complete_multilingual_nlp_analysis instead"""
    return complete_multilingual_nlp_analysis(df, text_column)


def create_sentiment_chart(sentiment_data):
    """
    Create sentiment distribution chart
    
    Args:
        sentiment_data (pd.DataFrame): Sentiment analysis results
        
    Returns:
        matplotlib.figure.Figure: Sentiment chart
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sentiment distribution pie chart
    sentiment_counts = sentiment_data['sentiment'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    ax1.set_title('Sentiment Distribution')
    
    # Polarity histogram
    ax2.hist(sentiment_data['polarity'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Sentiment Polarity Distribution')
    ax2.set_xlabel('Polarity Score')
    ax2.set_ylabel('Frequency')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def create_category_chart(categories_data):
    """
    Create complaint categories chart
    
    Args:
        categories_data (dict): Categories analysis results
        
    Returns:
        matplotlib.figure.Figure: Categories chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    category_counts = pd.Series(categories_data['clusters']).value_counts().sort_index()
    labels = [f"Cat {i+1}: {categories_data['labels'][i][:20]}..." 
              for i in range(len(category_counts))]
    
    bars = ax.bar(range(len(category_counts)), category_counts.values, color='lightcoral')
    ax.set_title('Complaint Categories Distribution')
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Complaints')
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def create_keywords_chart(keywords_data, top_n=10):
    """
    Create top keywords chart
    
    Args:
        keywords_data (list): Keywords analysis results
        top_n (int): Number of top keywords to display
        
    Returns:
        matplotlib.figure.Figure: Keywords chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_keywords = keywords_data[:top_n]
    keywords, scores = zip(*top_keywords)
    
    bars = ax.barh(range(len(keywords)), scores, color='lightgreen')
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords)
    ax.set_title(f'Top {top_n} Keywords by TF-IDF Score')
    ax.set_xlabel('TF-IDF Score')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    return fig


def wordcloud_to_base64(wordcloud):
    """
    Convert WordCloud to base64 string for web display
    
    Args:
        wordcloud (WordCloud): WordCloud object
        
    Returns:
        str: Base64 encoded image string
    """
    if wordcloud is None:
        return None
    
    img_buffer = io.BytesIO()
    wordcloud.to_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    return img_str


def get_summary_stats(df, text_column, results):
    """
    Generate summary statistics for the analysis
    
    Args:
        df (pd.DataFrame): Original dataframe
        text_column (str): Text column name
        results (dict): Analysis results
        
    Returns:
        dict: Summary statistics
    """
    # Handle multilingual keywords format
    top_keywords = []
    for kw in results['keywords'][:5]:
        if isinstance(kw, tuple) and len(kw) >= 1:
            top_keywords.append(kw[0])  # Get keyword text
        else:
            top_keywords.append(str(kw))
    
    stats = {
        'total_complaints': len(df),
        'avg_text_length': df[text_column].str.len().mean(),
        'sentiment_distribution': results['sentiment']['sentiment'].value_counts().to_dict(),
        'most_positive': results['sentiment']['polarity'].max(),
        'most_negative': results['sentiment']['polarity'].min(),
        'avg_polarity': results['sentiment']['polarity'].mean(),
        'num_categories': len(results['categories']['labels']),
        'top_keywords': top_keywords,
        'language_distribution': results.get('language_distribution', {})
    }
    
    return stats


def generate_pdf_report(results, stats):
    """
    Generate a PDF report of the NLP analysis results
    Supports multilingual content (English, Hindi, Hinglish)
    
    Args:
        results (dict): Analysis results from complete_nlp_analysis
        stats (dict): Summary statistics from get_summary_stats
        
    Returns:
        bytes: PDF file as bytes buffer
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 12, 'NLP Analysis Report - Public Complaints', 0, 1, 'C')
    pdf.ln(3)
    
    # Date
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(8)
    
    # Summary Statistics Section
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1, 'L', fill=True)
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 7, f'Total Complaints: {stats["total_complaints"]}', 0, 1, 'L')
    pdf.cell(0, 7, f'Average Text Length: {stats["avg_text_length"]:.0f} characters', 0, 1, 'L')
    pdf.cell(0, 7, f'Average Polarity: {stats["avg_polarity"]:.3f}', 0, 1, 'L')
    pdf.cell(0, 7, f'Number of Categories: {stats["num_categories"]}', 0, 1, 'L')
    
    # Language Distribution
    if 'language_distribution' in stats and stats['language_distribution']:
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 7, 'Language Distribution:', 0, 1, 'L')
        pdf.set_font('Arial', '', 11)
        for lang, count in stats['language_distribution'].items():
            percentage = (count / stats['total_complaints']) * 100
            lang_display = {'en': 'English', 'hi': 'Hindi', 'hinglish': 'Hinglish', 'mixed': 'Mixed'}.get(lang, lang)
            pdf.cell(0, 6, f'  {lang_display}: {count} ({percentage:.1f}%)', 0, 1, 'L')
    
    pdf.ln(8)
    
    # Sentiment Analysis Section
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, 'Sentiment Analysis', 0, 1, 'L', fill=True)
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 11)
    sentiment_counts = results['sentiment']['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results['sentiment'])) * 100
        pdf.cell(0, 7, f'{sentiment.title()}: {count} ({percentage:.1f}%)', 0, 1, 'L')
    pdf.ln(8)
    
    # Top Keywords Section
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, 'Top Keywords', 0, 1, 'L', fill=True)
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 10)
    for i, keyword_data in enumerate(results['keywords'][:15], 1):
        # Handle both 2-tuple and 3-tuple formats
        if isinstance(keyword_data, tuple):
            keyword = str(keyword_data[0])
            score = keyword_data[1] if len(keyword_data) > 1 else 0
        else:
            keyword = str(keyword_data)
            score = 0
        
        # Sanitize keyword for PDF (remove non-latin characters that FPDF can't handle)
        # Keep only ASCII printable characters
        safe_keyword = ''.join(c for c in keyword if ord(c) < 128 and c.isprintable())
        if not safe_keyword:
            safe_keyword = f'[Keyword_{i}]'
        
        pdf.cell(0, 6, f'{i}. {safe_keyword}: {score:.3f}', 0, 1, 'L')
    pdf.ln(8)
    
    # Categories Section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, 'Complaint Categories', 0, 1, 'L', fill=True)
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 11)
    for i, label in enumerate(results['categories']['labels'], 1):
        count = sum(1 for c in results['categories']['clusters'] if c == i-1)
        # Sanitize label for PDF
        safe_label = ''.join(c for c in str(label) if ord(c) < 128 and c.isprintable())
        if not safe_label:
            safe_label = f'Category_{i}'
        pdf.cell(0, 7, f'Category {i}: {safe_label} ({count} complaints)', 0, 1, 'L')
    
    # Language-specific Analysis
    if 'language' in results['sentiment'].columns:
        pdf.ln(8)
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, 'Language-Specific Analysis', 0, 1, 'L', fill=True)
        pdf.ln(3)
        
        pdf.set_font('Arial', '', 11)
        lang_stats = results['sentiment'].groupby('language')['sentiment'].value_counts().unstack(fill_value=0)
        for lang in lang_stats.index:
            lang_display = {'en': 'English', 'hi': 'Hindi', 'hinglish': 'Hinglish', 'mixed': 'Mixed'}.get(lang, lang)
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 7, f'{lang_display}:', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            for sentiment, count in lang_stats.loc[lang].items():
                if count > 0:
                    pdf.cell(0, 6, f'  {sentiment.title()}: {count}', 0, 1, 'L')
            pdf.ln(2)
    
    # Save PDF to bytes buffer
    buffer = io.BytesIO()
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin-1', errors='ignore')
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer.getvalue()
