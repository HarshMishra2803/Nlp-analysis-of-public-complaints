import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from datetime import datetime
import base64
from io import BytesIO
from fpdf import FPDF
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Detect if running on cloud (Render) vs local
# On Render, skip the heavy 500MB transformer model
is_cloud = os.environ.get('RENDER', 'false').lower() == 'true' or os.environ.get('PORT') is not None
if is_cloud:
    os.environ['SKIP_HEAVY_MODEL'] = 'true'
    print("☁️ Cloud deployment detected - using lightweight rule-based analysis")

# Import custom NLP utilities
import nlp_utils

# Page configuration
st.set_page_config(
    page_title="NLP Analysis of Public Complaints",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark professional theme base */
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero section - Clean Blue Gradient */
    .hero-section {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #60a5fa;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Section containers - Professional Gray-Blue */
    .section-container {
        background: linear-gradient(135deg, #475569 0%, #64748b 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 1px solid #94a3b8;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.15);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #60a5fa;
    }
    
    /* Metric cards - Clean Blue Accent */
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        text-align: center;
        border: 1px solid #60a5fa;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #dbeafe;
        font-weight: 600;
    }
    
    /* Category and keyword cards - Subtle Gray */
    .category-card, .keyword-item {
        background: rgba(148, 163, 184, 0.2);
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #60a5fa;
        color: #f1f5f9;
        transition: transform 0.2s ease;
    }
    
    .category-card:hover, .keyword-item:hover {
        transform: translateX(5px);
        background: rgba(148, 163, 184, 0.3);
    }
    
    .keyword-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Results counter - Professional Accent */
    .results-counter {
        background: rgba(59, 130, 246, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        color: #dbeafe;
        margin-bottom: 1rem;
        text-align: center;
        border: 1px solid #60a5fa;
    }
    
    /* Filter container - Consistent Theme */
    .filter-container {
        background: linear-gradient(135deg, #475569 0%, #64748b 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 1px solid #94a3b8;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Streamlit component overrides for professional theme */
    .stSelectbox > div > div {
        background-color: #475569 !important;
        color: #f1f5f9 !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 0.5rem !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #475569 !important;
        color: #f1f5f9 !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 0.5rem !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #475569 !important;
        color: #f1f5f9 !important;
        border: 1px solid #94a3b8 !important;
    }
    
    .stDataFrame {
        background-color: #475569 !important;
        color: #f1f5f9 !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 0.75rem !important;
    }
    
    .stDataFrame table {
        background-color: #475569 !important;
        color: #f1f5f9 !important;
    }
    
    .stDataFrame th {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .stDataFrame td {
        background-color: #475569 !important;
        color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }
    
    .stDataFrame tr:hover td {
        background-color: rgba(59, 130, 246, 0.2) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: 1px solid #60a5fa !important;
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    }
    
    .stExpander {
        background-color: #475569 !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 0.75rem !important;
    }
    
    .stExpander > div {
        background-color: #475569 !important;
        color: #f1f5f9 !important;
    }
    
    .stFileUploader > div {
        background-color: #475569 !important;
        border: 2px dashed #60a5fa !important;
        border-radius: 1rem !important;
    }
    
    .stFileUploader label {
        color: #dbeafe !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'df_analyzed' not in st.session_state:
        st.session_state.df_analyzed = None
    if 'selected_column' not in st.session_state:
        st.session_state.selected_column = None
    
    # Header with hero section
    st.markdown("""
    <div class="hero-section">
        <h1 class="main-header">📊 NLP Analysis of Public Complaints</h1>
        <p style="font-size: 1rem; color: #dbeafe; margin: 0; font-weight: 500;">
            Transform complaint data into actionable insights with advanced Natural Language Processing
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Complaint Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file containing complaint data"
        )
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        n_categories = st.slider("Number of Categories", min_value=2, max_value=10, value=5)
        n_keywords = st.slider("Number of Keywords", min_value=5, max_value=30, value=15)
        
        # Display options
        st.subheader("Display Options")
        show_wordcloud = st.checkbox("Show Word Cloud", value=True)
        show_raw_data = st.checkbox("Show Raw Data", value=False)
        
        # Clear analysis button
        if st.button("🔄 Clear Analysis", help="Reset analysis and start over"):
            st.session_state.analysis_results = None
            st.session_state.df_analyzed = None
            st.session_state.selected_column = None
            st.rerun()
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Loading data..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            
            # Column selection
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if not text_columns:
                st.error("No text columns found in the uploaded file.")
                return
            
            selected_column = st.selectbox(
                "Select the complaint text column:",
                text_columns,
                help="Choose the column containing the complaint text to analyze"
            )
            st.session_state.selected_column = selected_column
            
            # Data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Perform NLP analysis
            if st.button("🚀 Run NLP Analysis", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📥 Loading data...")
                    progress_bar.progress(10)
                    
                    # Check data size and limit if too large for cloud
                    data_size = len(df)
                    if data_size > 1000:
                        st.warning(f"⚠️ Large dataset detected ({data_size} rows). Processing first 1000 rows for cloud deployment.")
                        df = df.head(1000)
                    
                    # Check if using cloud mode (lightweight)
                    if is_cloud:
                        status_text.text("☁️ Cloud mode: Using lightweight rule-based analysis...")
                        st.info("ℹ️ Running in cloud mode - using fast rule-based sentiment analysis (no 500MB model download)")
                    else:
                        status_text.text("🧠 Loading sentiment model (one-time)...")
                    progress_bar.progress(20)
                    
                    # Pre-load model to avoid timeout
                    model = nlp_utils.get_multilingual_sentiment_model()
                    if model is False:
                        st.info("ℹ️ Using rule-based sentiment analysis")
                    
                    status_text.text("📊 Analyzing sentiments...")
                    progress_bar.progress(40)
                    
                    # Run analysis with progress tracking
                    results = nlp_utils.complete_nlp_analysis(df, selected_column)
                    
                    status_text.text("🎯 Calculating urgency scores...")
                    progress_bar.progress(70)
                    
                    # Add results to dataframe
                    df_analyzed = df.copy()
                    df_analyzed['sentiment'] = results['sentiment']['sentiment']
                    df_analyzed['polarity'] = results['sentiment']['polarity']
                    df_analyzed['subjectivity'] = results['sentiment']['subjectivity']
                    df_analyzed['language'] = results['sentiment']['language']
                    df_analyzed['confidence'] = results['sentiment']['confidence']
                    
                    # Add urgency columns
                    df_analyzed['urgency_score'] = results['urgency']['urgency_score']
                    df_analyzed['urgency_level'] = results['urgency']['urgency_level']
                    df_analyzed['priority_label'] = results['urgency']['priority_label']
                    
                    status_text.text("📂 Categorizing complaints...")
                    progress_bar.progress(90)
                    
                    # Add categories if available
                    if len(results['categories']['clusters']) == len(df):
                        df_analyzed['category'] = results['categories']['clusters']
                        df_analyzed['category_label'] = [
                            results['categories']['labels'][cat] for cat in results['categories']['clusters']
                        ]
                    
                    # Store in session state
                    st.session_state.analysis_results = results
                    st.session_state.df_analyzed = df_analyzed
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Multilingual analysis with urgency scoring completed! Supports English, Hindi, and Hinglish.")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Tip: If the issue persists, try uploading a smaller file (under 500 rows) for cloud deployment.")
            
            # Display results if analysis has been run
            if st.session_state.analysis_results is not None and st.session_state.df_analyzed is not None:
                results = st.session_state.analysis_results
                df_analyzed = st.session_state.df_analyzed
                
                # Summary Statistics with modern cards
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📈 Summary Statistics</div>', unsafe_allow_html=True)
                stats = nlp_utils.get_summary_stats(df, st.session_state.selected_column, results)
                
                # Language distribution
                language_dist = results.get('language_distribution', {})
                urgency_dist = results.get('urgency_distribution', {})
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{stats['total_complaints']}</div>
                        <div class="metric-label">Total Complaints</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{stats['avg_text_length']:.0f}</div>
                        <div class="metric-label">Avg Text Length</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{stats['avg_polarity']:.3f}</div>
                        <div class="metric-label">Avg Polarity</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{stats['num_categories']}</div>
                        <div class="metric-label">Categories Found</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col5:
                    num_languages = len(language_dist)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{num_languages}</div>
                        <div class="metric-label">Languages Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col6:
                    # Count critical/high urgency complaints
                    critical_count = urgency_dist.get('critical', 0) + urgency_dist.get('high', 0)
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);">
                        <div class="metric-value">{critical_count}</div>
                        <div class="metric-label">Critical/High Priority</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Language Distribution Section
                if language_dist:
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">🌐 Language Distribution</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Language distribution pie chart
                        lang_names = {
                            'en': 'English 🇬🇧',
                            'hi': 'Hindi 🇮🇳',
                            'hinglish': 'Hinglish 🔄',
                            'mixed': 'Mixed Hindi-English 🌐'
                        }
                        
                        lang_labels = [lang_names.get(lang, lang) for lang in language_dist.keys()]
                        lang_values = list(language_dist.values())
                        
                        fig_lang = px.pie(
                            values=lang_values,
                            names=lang_labels,
                            title="Language Distribution in Complaints",
                            color_discrete_sequence=['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']
                        )
                        fig_lang.update_layout(
                            font_size=14, 
                            title_font_size=16, 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_lang, use_container_width=True)
                    
                    with col2:
                        # Language breakdown table
                        st.markdown('<div class="section-header" style="font-size: 1.2rem;">Language Breakdown</div>', unsafe_allow_html=True)
                        
                        total = sum(language_dist.values())
                        for lang, count in language_dist.items():
                            percentage = (count / total) * 100
                            lang_label = lang_names.get(lang, lang)
                            
                            # Color coding for different languages
                            lang_colors = {
                                'en': '#3b82f6',
                                'hi': '#f59e0b',
                                'hinglish': '#10b981',
                                'mixed': '#8b5cf6'
                            }
                            color = lang_colors.get(lang, '#64748b')
                            
                            st.markdown(f"""
                            <div class="category-card" style="border-left-color: {color};">
                                <strong>{lang_label}</strong><br>
                                <span style="color: #94a3b8;">{count} complaints ({percentage:.1f}%)</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Urgency Analysis Section (NEW)
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🚨 Urgency & Priority Analysis</div>', unsafe_allow_html=True)
                
                urgency_dist = results.get('urgency_distribution', {})
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Urgency distribution pie chart
                    urgency_colors = {
                        'critical': '#dc2626',
                        'high': '#ef4444',
                        'medium': '#f59e0b',
                        'low': '#10b981'
                    }
                    
                    urgency_labels = {
                        'critical': 'Critical (81-100)',
                        'high': 'High (61-80)',
                        'medium': 'Medium (31-60)',
                        'low': 'Low (0-30)'
                    }
                    
                    # Only show levels that have complaints
                    urgency_data = {k: v for k, v in urgency_dist.items() if v > 0}
                    
                    if urgency_data:
                        labels = [urgency_labels.get(k, k) for k in urgency_data.keys()]
                        values = list(urgency_data.values())
                        colors = [urgency_colors.get(k, '#64748b') for k in urgency_data.keys()]
                        
                        fig_urgency = px.pie(
                            values=values,
                            names=labels,
                            title="Urgency Level Distribution",
                            color_discrete_sequence=colors
                        )
                        fig_urgency.update_layout(
                            font_size=14,
                            title_font_size=16,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_urgency, use_container_width=True)
                    else:
                        st.info("No urgency data available")
                
                with col2:
                    st.markdown('<div class="section-header" style="font-size: 1.2rem;">Priority Breakdown</div>', unsafe_allow_html=True)
                    
                    total = sum(urgency_dist.values()) if urgency_dist else 0
                    
                    for level in ['critical', 'high', 'medium', 'low']:
                        count = urgency_dist.get(level, 0)
                        percentage = (count / total * 100) if total > 0 else 0
                        
                        config = {
                            'critical': {'label': '🔴 Critical (81-100)', 'color': '#dc2626'},
                            'high': {'label': '🟠 High (61-80)', 'color': '#ef4444'},
                            'medium': {'label': '🟡 Medium (31-60)', 'color': '#f59e0b'},
                            'low': {'label': '🟢 Low (0-30)', 'color': '#10b981'}
                        }.get(level)
                        
                        st.markdown(f"""
                        <div class="category-card" style="border-left-color: {config['color']};">
                            <strong>{config['label']}</strong><br>
                            <span style="color: #94a3b8;">{count} complaints ({percentage:.1f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show average urgency score
                    if 'urgency' in results and 'urgency_score' in results['urgency'].columns:
                        avg_urgency = results['urgency']['urgency_score'].mean()
                        st.markdown(f"""
                        <div style="margin-top: 1rem; padding: 1rem; background: rgba(59, 130, 246, 0.15); border-radius: 0.5rem; text-align: center;">
                            <strong style="color: #dbeafe;">Average Urgency Score</strong><br>
                            <span style="font-size: 1.5rem; color: #f1f5f9;">{avg_urgency:.1f}/100</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Sentiment Analysis
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">😊 Sentiment Analysis</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment distribution pie chart
                    sentiment_counts = results['sentiment']['sentiment'].value_counts()
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={
                            'positive': '#2E8B57',
                            'neutral': '#4682B4',
                            'negative': '#DC143C'
                        }
                    )
                    fig_pie.update_layout(font_size=14, title_font_size=16, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Polarity histogram
                    fig_hist = px.histogram(
                        results['sentiment'],
                        x='polarity',
                        nbins=20,
                        title="Sentiment Polarity Distribution",
                        labels={'polarity': 'Polarity Score', 'count': 'Frequency'},
                        color_discrete_sequence=['#667eea']
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Neutral")
                    fig_hist.update_layout(font_size=14, title_font_size=16, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Complaint Categories
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📂 Complaint Categories</div>', unsafe_allow_html=True)
                
                if 'category' in df_analyzed.columns:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Category distribution
                        category_counts = df_analyzed['category'].value_counts().sort_index()
                        category_labels = [results['categories']['labels'][i] for i in category_counts.index]
                        
                        fig_cat = px.bar(
                            x=category_labels,
                            y=category_counts.values,
                            title="Complaint Categories Distribution",
                            labels={'x': 'Category', 'y': 'Number of Complaints'},
                            color_discrete_sequence=['#764ba2']
                        )
                        fig_cat.update_xaxes(tickangle=45)
                        fig_cat.update_layout(font_size=14, title_font_size=16, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_cat, use_container_width=True)
                    
                    with col2:
                        st.markdown('<div class="section-header" style="font-size: 1.2rem;">Category Details</div>', unsafe_allow_html=True)
                        for i, label in enumerate(results['categories']['labels']):
                            count = sum(1 for c in results['categories']['clusters'] if c == i)
                            st.markdown(f"""
                            <div class="category-card">
                                <strong>Category {i+1}:</strong> {label}<br>
                                <strong>Count:</strong> {count}
                            </div>
                            """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Keywords Analysis
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🔍 Keywords Analysis</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Top keywords chart
                    top_keywords = results['keywords'][:n_keywords]
                    keywords, scores = zip(*top_keywords)
                    
                    fig_kw = px.bar(
                        x=list(scores),
                        y=list(keywords),
                        orientation='h',
                        title=f"Top {n_keywords} Keywords by TF-IDF Score",
                        labels={'x': 'TF-IDF Score', 'y': 'Keywords'},
                        color_discrete_sequence=['#667eea']
                    )
                    fig_kw.update_layout(height=600, font_size=14, title_font_size=16, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_kw, use_container_width=True)
                
                with col2:
                    st.markdown('<div class="section-header" style="font-size: 1.2rem;">Top Keywords</div>', unsafe_allow_html=True)
                    
                    # Language badges for keywords
                    lang_badge_colors = {
                        'en': '#3b82f6',
                        'hi': '#f59e0b',
                        'hinglish': '#10b981',
                        'mixed': '#8b5cf6'
                    }
                    
                    for i, keyword_data in enumerate(results['keywords'][:10], 1):
                        # Handle both old format (keyword, score) and new format (keyword, score, language)
                        if len(keyword_data) >= 3:
                            keyword, score, lang = keyword_data[:3]
                        else:
                            keyword, score = keyword_data[:2]
                            lang = nlp_utils.detect_language(keyword)
                        
                        lang_label = {'en': 'EN', 'hi': 'HI', 'hinglish': 'HIN', 'mixed': 'MIX'}.get(lang, lang.upper())
                        badge_color = lang_badge_colors.get(lang, '#64748b')
                        
                        st.markdown(f"""
                        <div class="keyword-item">
                            <span><strong>{i}.</strong> {keyword}</span>
                            <div>
                                <span style="background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-right: 8px;">{lang_label}</span>
                                <span style="color: #667eea; font-weight: 600;">{score:.3f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Word Cloud
                if show_wordcloud and results['wordcloud']:
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">☁️ Word Cloud</div>', unsafe_allow_html=True)
                    
                    # Convert wordcloud to image
                    fig_wc, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(results['wordcloud'], interpolation='bilinear')
                    ax.axis('off')
                    fig_wc.patch.set_facecolor('none')
                    st.pyplot(fig_wc)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Duplicate Detection Section
                if results.get('duplicates'):
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">🔄 Duplicate Complaint Detection</div>', unsafe_allow_html=True)
                    
                    duplicates = results['duplicates']
                    st.warning(f"Found {len(duplicates)} group(s) of potentially duplicate complaints")
                    
                    for i, group in enumerate(duplicates[:5], 1):  # Show first 5 groups
                        with st.expander(f"Duplicate Group #{i} ({len(group)} complaints)"):
                            for idx in group:
                                st.markdown(f"""
                                <div style="padding: 0.5rem; background: rgba(245, 158, 11, 0.1); border-radius: 0.25rem; margin: 0.25rem 0;">
                                    <strong>Complaint #{idx + 1}:</strong> {df_analyzed.iloc[idx][st.session_state.selected_column][:100]}...
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Department Routing & Summarization Section
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🏢 Smart Department Routing & Auto-Summarization</div>', unsafe_allow_html=True)
                
                if 'routing' in results:
                    routing_df = results['routing']
                    
                    # Department distribution
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        dept_counts = routing_df['suggested_department'].value_counts()
                        fig_dept = px.pie(
                            values=dept_counts.values,
                            names=[d.replace('_', ' ').title() for d in dept_counts.index],
                            title="Suggested Department Distribution",
                            color_discrete_sequence=['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#ec4899']
                        )
                        fig_dept.update_layout(font_size=14, title_font_size=16, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_dept, use_container_width=True)
                    
                    with col2:
                        st.markdown('<div class="section-header" style="font-size: 1.2rem;">Department Breakdown</div>', unsafe_allow_html=True)
                        for dept, count in dept_counts.items():
                            dept_display = dept.replace('_', ' ').title()
                            st.markdown(f"""
                            <div class="category-card">
                                <strong>{dept_display}</strong><br>
                                <span style="color: #94a3b8;">{count} complaints</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Sample complaints with summaries and responses
                    st.markdown('<div class="section-header" style="font-size: 1.2rem; margin-top: 1rem;">Sample Auto-Generated Summaries & Responses</div>', unsafe_allow_html=True)
                    
                    # Show top 3 urgent complaints with summaries
                    urgent_indices = df_analyzed.nlargest(3, 'urgency_score').index
                    
                    for idx in urgent_indices[:3]:
                        urgency_level = df_analyzed.loc[idx, 'urgency_level']
                        urgency_color = {'critical': '#dc2626', 'high': '#ef4444', 'medium': '#f59e0b', 'low': '#10b981'}.get(urgency_level, '#64748b')
                        
                        with st.expander(f"Complaint #{idx + 1} - Urgency: {urgency_level.upper()}"):
                            st.markdown(f"""
                            <div style="padding: 1rem; background: rgba(30, 41, 59, 0.8); border-radius: 0.5rem; border-left: 4px solid {urgency_color};">
                                <strong>📝 Original:</strong><br>
                                {df_analyzed.loc[idx, st.session_state.selected_column]}<br><br>
                                <strong>📋 Summary:</strong> {routing_df.loc[idx, 'summary']}<br><br>
                                <strong>🏢 Suggested Department:</strong> {routing_df.loc[idx, 'department_display']}<br><br>
                                <strong>💬 Suggested Response:</strong><br>
                                <div style="padding: 0.75rem; background: rgba(59, 130, 246, 0.15); border-radius: 0.25rem;">
                                    {routing_df.loc[idx, 'suggested_response']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Manual Override Button
                            if st.button(f"⚙️ Override Priority for Complaint #{idx + 1}", key=f"override_{idx}"):
                                st.session_state[f"show_override_{idx}"] = True
                            
                            if st.session_state.get(f"show_override_{idx}", False):
                                new_priority = st.selectbox(
                                    "Select New Priority:",
                                    options=['critical', 'high', 'medium', 'low'],
                                    key=f"priority_select_{idx}"
                                )
                                override_reason = st.text_input(
                                    "Reason for override (optional):",
                                    key=f"reason_{idx}"
                                )
                                if st.button("✅ Apply Override", key=f"apply_{idx}"):
                                    # Apply the override
                                    nlp_utils.set_manual_priority_override(f"complaint_{idx}", new_priority, override_reason)
                                    st.success(f"Priority overridden to {new_priority.upper()}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Priority-based Complaint Queue
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🚨 Priority-Based Complaint Queue</div>', unsafe_allow_html=True)
                
                # Create priority queue
                queue_df = nlp_utils.create_priority_queue(df_analyzed, st.session_state.selected_column, results['urgency'])
                queue_stats = nlp_utils.get_priority_queue_stats(queue_df)
                
                # Queue statistics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);">
                        <div class="metric-value">{queue_stats['critical_count']}</div>
                        <div class="metric-label">🔴 Critical</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);">
                        <div class="metric-value">{queue_stats['high_count']}</div>
                        <div class="metric-label">🟠 High</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #ca8a04 0%, #a16207 100%);">
                        <div class="metric-value">{queue_stats['medium_count']}</div>
                        <div class="metric-label">🟡 Medium</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);">
                        <div class="metric-value">{queue_stats['low_count']}</div>
                        <div class="metric-label">🟢 Low</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col5:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);">
                        <div class="metric-value">{queue_stats['average_urgency_score']:.1f}</div>
                        <div class="metric-label">Avg Urgency</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<br>', unsafe_allow_html=True)
                
                # Display priority queue table
                st.markdown('<div class="section-header" style="font-size: 1.2rem;">📋 Priority Queue (Highest Urgency First)</div>', unsafe_allow_html=True)
                
                # Prepare queue display columns
                queue_display_cols = ['queue_position', st.session_state.selected_column, 'urgency_score', 'urgency_level', 'priority_label']
                if 'suggested_department' in queue_df.columns:
                    queue_display_cols.append('suggested_department')
                if 'category_label' in queue_df.columns:
                    queue_display_cols.append('category_label')
                
                # Show top 20 urgent complaints
                st.dataframe(
                    queue_df[queue_display_cols].head(20),
                    use_container_width=True,
                    height=400
                )
                
                # Export priority queue button
                queue_csv = queue_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Priority Queue (CSV)",
                    data=queue_csv,
                    file_name=f"priority_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Filtering and Search
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🔍 Filter & Search Results</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sentiment_filter = st.multiselect(
                        "Filter by Sentiment:",
                        options=['positive', 'neutral', 'negative'],
                        default=['positive', 'neutral', 'negative']
                    )
                
                with col2:
                    if 'category' in df_analyzed.columns:
                        category_options = list(range(len(results['categories']['labels'])))
                        category_filter = st.multiselect(
                            "Filter by Category:",
                            options=category_options,
                            default=category_options,
                            format_func=lambda x: f"Category {x+1}: {results['categories']['labels'][x]}"
                        )
                    else:
                        category_filter = []
                
                with col3:
                    # Urgency filter
                    urgency_filter = st.multiselect(
                        "Filter by Urgency:",
                        options=['critical', 'high', 'medium', 'low'],
                        default=['critical', 'high', 'medium', 'low'],
                        format_func=lambda x: {
                            'critical': '🔴 Critical',
                            'high': '🟠 High',
                            'medium': '🟡 Medium',
                            'low': '🟢 Low'
                        }.get(x, x)
                    )
                
                with col4:
                    search_term = st.text_input(
                        "Search in complaints:",
                        placeholder="Enter keywords to search..."
                    )
                
                # Apply filters
                filtered_df = df_analyzed[df_analyzed['sentiment'].isin(sentiment_filter)]
                
                if category_filter and 'category' in df_analyzed.columns:
                    filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
                
                if urgency_filter and 'urgency_level' in df_analyzed.columns:
                    filtered_df = filtered_df[filtered_df['urgency_level'].isin(urgency_filter)]
                
                if search_term:
                    filtered_df = filtered_df[
                        filtered_df[st.session_state.selected_column].str.contains(search_term, case=False, na=False)
                    ]
                
                st.markdown(f"""
                <div class="results-counter">
                    Showing {len(filtered_df)} of {len(df_analyzed)} complaints
                </div>
                """, unsafe_allow_html=True)
                
                # Display filtered data with enhanced styling
                display_columns = [st.session_state.selected_column, 'sentiment', 'urgency_score', 'priority_label', 'language']
                if 'category_label' in filtered_df.columns:
                    display_columns.append('category_label')
                if 'suggested_department' in df_analyzed.columns:
                    display_columns.append('suggested_department')
                
                st.dataframe(
                    filtered_df[display_columns].head(100),
                    use_container_width=True,
                    height=400
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download Options
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV download
                    csv_data = df_analyzed.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Complete Analysis as CSV",
                        data=csv_data,
                        file_name=f"complaint_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # PDF report download
                    if st.button("📑 Generate PDF Report", use_container_width=True):
                        try:
                            with st.spinner("Generating PDF report..."):
                                try:
                                    pdf_data = nlp_utils.generate_pdf_report(results, stats)
                                    st.success("PDF report generated successfully!")
                                    st.download_button(
                                        label="📑 Download PDF Report",
                                        data=pdf_data,
                                        file_name=f"complaint_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.error(f"Error creating PDF download: {str(e)}")
                                    st.error("Please try again or contact support if the issue persists.")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                            st.error("Please ensure all analysis data is available.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Raw data display
                if show_raw_data:
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">📊 Raw Analysis Data</div>', unsafe_allow_html=True)
                    
                    with st.expander("📈 Sentiment Analysis Data", expanded=False):
                        st.dataframe(results['sentiment'], use_container_width=True)
                    
                    with st.expander("🔍 Keywords Analysis Data", expanded=False):
                        keywords_df = pd.DataFrame(results['keywords'], columns=['Keyword', 'TF-IDF Score'])
                        st.dataframe(keywords_df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your data format and try again.")
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div class="section-container">
            <div class="section-header">🚀 Welcome to Multilingual NLP Complaint Analysis</div>
            <p style="font-size: 1rem; line-height: 1.4; color: #f1f5f9; margin-bottom: 1rem;">
                Upload your complaint data using the sidebar to get started with advanced NLP analysis.
                <strong style="color: #60a5fa;">Now supports English, Hindi, and Hinglish!</strong>
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 0.75rem; margin-top: 1rem;">
                <div style="background: rgba(59, 130, 246, 0.15); padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #60a5fa;">
                    <strong style="color: #dbeafe;">🌐 Multilingual Support</strong><br>
                    <span style="font-size: 0.9rem; color: #f1f5f9;">English, Hindi & Hinglish</span>
                </div>
                <div style="background: rgba(59, 130, 246, 0.15); padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #60a5fa;">
                    <strong style="color: #dbeafe;">📊 Sentiment Analysis</strong><br>
                    <span style="font-size: 0.9rem; color: #f1f5f9;">Understand emotional tone</span>
                </div>
                <div style="background: rgba(59, 130, 246, 0.15); padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #60a5fa;">
                    <strong style="color: #dbeafe;">📂 Auto Categorization</strong><br>
                    <span style="font-size: 0.9rem; color: #f1f5f9;">Group similar complaints</span>
                </div>
                <div style="background: rgba(59, 130, 246, 0.15); padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #60a5fa;">
                    <strong style="color: #dbeafe;">🔍 Keyword Extraction</strong><br>
                    <span style="font-size: 0.9rem; color: #f1f5f9;">Identify important terms</span>
                </div>
                <div style="background: rgba(59, 130, 246, 0.15); padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #60a5fa;">
                    <strong style="color: #dbeafe;">📈 Interactive Charts</strong><br>
                    <span style="font-size: 0.9rem; color: #f1f5f9;">Explore with visualizations</span>
                </div>
                <div style="background: rgba(59, 130, 246, 0.15); padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #60a5fa;">
                    <strong style="color: #dbeafe;">� Export Options</strong><br>
                    <span style="font-size: 0.9rem; color: #f1f5f9;">Download CSV/PDF reports</span>
                </div>
            </div>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(16, 185, 129, 0.15); border-radius: 0.5rem; border-left: 3px solid #10b981;">
                <strong style="color: #6ee7b7;">� Sample Data Format:</strong>
                <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #f1f5f9;">
                    <strong>English:</strong> "The service is very slow and frustrating"<br>
                    <strong>Hindi:</strong> "यह सेवा बहुत खराब है, कोई मदद नहीं मिल रही"<br>
                    <strong>Hinglish:</strong> "Bohot badiya service hai, acha kaam kiya"
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
