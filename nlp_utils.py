"""
NLP Utilities for Public Complaints Analysis
This module contains functions for sentiment analysis, complaint categorization,
keyword extraction, and word cloud generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import re
from collections import Counter
import warnings
import io
import base64
from fpdf import FPDF
from datetime import datetime

warnings.filterwarnings('ignore')

# Download required NLTK data
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


def preprocess_text(text):
    """
    Clean and preprocess text data
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def analyze_sentiment(text):
    """
    Perform sentiment analysis using TextBlob
    
    Args:
        text (str): Text to analyze
        
    Returns:
        tuple: (polarity, subjectivity, sentiment_label)
    """
    if not text or pd.isna(text):
        return 0, 0, 'neutral'
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Classify sentiment
    if polarity > 0.1:
        sentiment_label = 'positive'
    elif polarity < -0.1:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    
    return polarity, subjectivity, sentiment_label


def batch_sentiment_analysis(texts):
    """
    Perform sentiment analysis on a list of texts
    
    Args:
        texts (list): List of texts to analyze
        
    Returns:
        pd.DataFrame: DataFrame with sentiment results
    """
    results = []
    for text in texts:
        polarity, subjectivity, label = analyze_sentiment(text)
        results.append({
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': label
        })
    return pd.DataFrame(results)


def categorize_complaints(texts, n_categories=5):
    """
    Categorize complaints using K-means clustering on TF-IDF vectors
    
    Args:
        texts (list): List of complaint texts
        n_categories (int): Number of categories to create
        
    Returns:
        tuple: (clusters, cluster_labels, vectorizer, kmeans_model)
    """
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Remove empty texts
    processed_texts = [text for text in processed_texts if text.strip()]
    
    if len(processed_texts) < n_categories:
        n_categories = max(1, len(processed_texts))
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
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


def extract_keywords(texts, top_n=20):
    """
    Extract top keywords using TF-IDF
    
    Args:
        texts (list): List of texts
        top_n (int): Number of top keywords to return
        
    Returns:
        list: List of (keyword, score) tuples
    """
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts if text and not pd.isna(text)]
    
    if not processed_texts:
        return []
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=top_n * 2,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Get feature names and scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    
    # Create keyword-score pairs
    keyword_scores = list(zip(feature_names, tfidf_scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_scores[:top_n]


def generate_wordcloud(texts, width=800, height=400):
    """
    Generate word cloud from complaint texts
    
    Args:
        texts (list): List of texts
        width (int): Word cloud width
        height (int): Word cloud height
        
    Returns:
        WordCloud: Generated word cloud object
    """
    # Combine all texts
    combined_text = ' '.join([preprocess_text(text) for text in texts if text and not pd.isna(text)])
    
    if not combined_text.strip():
        return None
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        stopwords=set(['complaint', 'issue', 'problem', 'service', 'customer']),
        max_words=100,
        colormap='viridis'
    ).generate(combined_text)
    
    return wordcloud


def complete_nlp_analysis(df, text_column):
    """
    Perform complete NLP analysis on complaint data
    
    Args:
        df (pd.DataFrame): DataFrame containing complaint data
        text_column (str): Name of the column containing complaint text
        
    Returns:
        dict: Dictionary containing all analysis results
    """
    results = {}
    
    # Get text data
    texts = df[text_column].fillna('').tolist()
    
    # 1. Sentiment Analysis
    sentiment_results = batch_sentiment_analysis(texts)
    results['sentiment'] = sentiment_results
    
    # 2. Complaint Categorization
    clusters, cluster_labels, vectorizer, kmeans = categorize_complaints(texts)
    results['categories'] = {
        'clusters': clusters,
        'labels': cluster_labels,
        'vectorizer': vectorizer,
        'model': kmeans
    }
    
    # 3. Keyword Extraction
    keywords = extract_keywords(texts)
    results['keywords'] = keywords
    
    # 4. Word Cloud
    wordcloud = generate_wordcloud(texts)
    results['wordcloud'] = wordcloud
    
    return results


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
    stats = {
        'total_complaints': len(df),
        'avg_text_length': df[text_column].str.len().mean(),
        'sentiment_distribution': results['sentiment']['sentiment'].value_counts().to_dict(),
        'most_positive': results['sentiment']['polarity'].max(),
        'most_negative': results['sentiment']['polarity'].min(),
        'avg_polarity': results['sentiment']['polarity'].mean(),
        'num_categories': len(results['categories']['labels']),
        'top_keywords': [kw[0] for kw in results['keywords'][:5]]
    }
    
    return stats


def generate_pdf_report(results, stats):
    """
    Generate a PDF report of the NLP analysis results
    
    Args:
        results (dict): Analysis results from complete_nlp_analysis
        stats (dict): Summary statistics from get_summary_stats
        
    Returns:
        bytes: PDF file as bytes buffer
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # Title
    pdf.cell(0, 10, 'NLP Analysis Report - Public Complaints', 0, 1, 'C')
    pdf.ln(5)
    
    # Date
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(10)
    
    # Summary Statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f'Total Complaints: {stats["total_complaints"]}', 0, 1, 'L')
    pdf.cell(0, 8, f'Average Text Length: {stats["avg_text_length"]:.0f} characters', 0, 1, 'L')
    pdf.cell(0, 8, f'Average Polarity: {stats["avg_polarity"]:.3f}', 0, 1, 'L')
    pdf.cell(0, 8, f'Number of Categories: {stats["num_categories"]}', 0, 1, 'L')
    pdf.ln(10)
    
    # Sentiment Analysis
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Sentiment Analysis', 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    sentiment_counts = results['sentiment']['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results['sentiment'])) * 100
        pdf.cell(0, 8, f'{sentiment.title()}: {count} ({percentage:.1f}%)', 0, 1, 'L')
    pdf.ln(10)
    
    # Top Keywords
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Top Keywords', 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    for i, (keyword, score) in enumerate(results['keywords'][:10], 1):
        pdf.cell(0, 8, f'{i}. {keyword}: {score:.3f}', 0, 1, 'L')
    pdf.ln(10)
    
    # Categories
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Complaint Categories', 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    for i, label in enumerate(results['categories']['labels'], 1):
        count = sum(1 for c in results['categories']['clusters'] if c == i-1)
        pdf.cell(0, 8, f'Category {i}: {label} ({count} complaints)', 0, 1, 'L')
    
    # Convert to bytes
    return pdf.output(dest='S').encode('latin-1')
