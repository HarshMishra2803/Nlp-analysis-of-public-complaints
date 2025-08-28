"""
NLP Analysis of Public Complaints - Streamlit Dashboard
Interactive dashboard for analyzing public complaint data using NLP techniques.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64
import io
from datetime import datetime
import nlp_utils

# Page configuration
st.set_page_config(
    page_title="NLP Analysis of Public Complaints",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark purple theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme base */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
        border-radius: 1.5rem;
        border: 2px solid #6366f1;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }
    
    .main-header {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0 0 1rem 0;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
    }
    
    /* Section containers */
    .section-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 1.5rem;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(139, 92, 246, 0.1);
        border: 1px solid #6366f1;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #c084fc;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #6366f1;
        text-shadow: 0 0 10px rgba(192, 132, 252, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
        margin: 1rem 0;
        border: 1px solid #a855f7;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(139, 92, 246, 0.6);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Category cards */
    .category-card {
        background: linear-gradient(135deg, #312e81 0%, #1e1b4b 100%);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #8b5cf6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #6366f1;
    }
    
    /* Keyword items */
    .keyword-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 0.75rem 0;
        border: 1px solid #6366f1;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Results counter */
    .results-counter {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        font-weight: 700;
        margin: 2rem 0;
        font-size: 1.2rem;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
        border: 1px solid #a855f7;
    }
        gap: 0.5rem;
    }
    
    /* Filter container */
    .filter-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        padding: 2rem;
        border-radius: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid #6366f1;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Streamlit component overrides for dark theme */
    .stSelectbox > div > div {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
        border: 1px solid #6366f1 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
        border: 1px solid #6366f1 !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
        border: 1px solid #6366f1 !important;
    }
    
    .stDataFrame {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
        border: 1px solid #6366f1 !important;
        border-radius: 0.75rem !important;
    }
    
    .stDataFrame table {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
    }
    
    .stDataFrame th {
        background-color: #312e81 !important;
        color: #c084fc !important;
        font-weight: 700 !important;
    }
    
    .stDataFrame td {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
        border-color: #6366f1 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: 1px solid #a855f7 !important;
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
    }
    
    .stExpander {
        background-color: #1e1b4b !important;
        border: 1px solid #6366f1 !important;
        border-radius: 0.75rem !important;
    }
    
    .stExpander > div {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
    }
    
    /* Sidebar styling */
    .css-1lcbmhc {
        background-color: #0f0f23 !important;
    }
    
    .css-17eq0hr {
        background-color: #1a1a2e !important;
        color: #e2e8f0 !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: #1e1b4b !important;
        border: 2px dashed #6366f1 !important;
        border-radius: 1rem !important;
    }
    
    .stFileUploader label {
        color: #c084fc !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .keyword-item {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .keyword-item:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }
    
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        overflow: hidden;
    }
    
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f0f4ff;
    }
    
    .stAlert {
        border-radius: 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stExpander {
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.75rem;
        padding: 0.5rem 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        <p style="font-size: 1.2rem; color: #64748b; margin: 0; font-weight: 400;">
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
            
            st.success(f"✅ Data loaded successfully! {len(df)} records found.")
            
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
                with st.spinner("Performing NLP analysis... This may take a moment."):
                    # Run complete analysis
                    results = nlp_utils.complete_nlp_analysis(df, selected_column)
                    
                    # Add results to dataframe
                    df_analyzed = df.copy()
                    df_analyzed['sentiment'] = results['sentiment']['sentiment']
                    df_analyzed['polarity'] = results['sentiment']['polarity']
                    df_analyzed['subjectivity'] = results['sentiment']['subjectivity']
                    
                    # Add categories if available
                    if len(results['categories']['clusters']) == len(df):
                        df_analyzed['category'] = results['categories']['clusters']
                        df_analyzed['category_label'] = [
                            results['categories']['labels'][cat] for cat in results['categories']['clusters']
                        ]
                    
                    # Store in session state
                    st.session_state.analysis_results = results
                    st.session_state.df_analyzed = df_analyzed
                
                st.success("✅ Analysis completed!")
            
            # Display results if analysis has been run
            if st.session_state.analysis_results is not None and st.session_state.df_analyzed is not None:
                results = st.session_state.analysis_results
                df_analyzed = st.session_state.df_analyzed
                
                # Summary Statistics with modern cards
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📈 Summary Statistics</div>', unsafe_allow_html=True)
                stats = nlp_utils.get_summary_stats(df, st.session_state.selected_column, results)
                
                col1, col2, col3, col4 = st.columns(4)
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
                    fig_pie.update_layout(font_size=14, title_font_size=16)
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
                    fig_hist.update_layout(font_size=14, title_font_size=16)
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
                        fig_cat.update_layout(font_size=14, title_font_size=16)
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
                    fig_kw.update_layout(height=600, font_size=14, title_font_size=16)
                    st.plotly_chart(fig_kw, use_container_width=True)
                
                with col2:
                    st.markdown('<div class="section-header" style="font-size: 1.2rem;">Top Keywords</div>', unsafe_allow_html=True)
                    for i, (keyword, score) in enumerate(results['keywords'][:10], 1):
                        st.markdown(f"""
                        <div class="keyword-item">
                            <span><strong>{i}.</strong> {keyword}</span>
                            <span style="color: #667eea; font-weight: 600;">{score:.3f}</span>
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
                    st.pyplot(fig_wc)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Filtering and Search - MOVED OUTSIDE THE BUTTON BLOCK
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                st.header("🔎 Data Exploration & Filtering")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_filter = st.multiselect(
                        "🎭 Filter by Sentiment",
                        options=df_analyzed['sentiment'].unique(),
                        default=df_analyzed['sentiment'].unique(),
                        key="sentiment_filter"
                    )
                
                with col2:
                    if 'category' in df_analyzed.columns:
                        category_filter = st.multiselect(
                            "📂 Filter by Category",
                            options=sorted(df_analyzed['category'].unique()),
                            default=sorted(df_analyzed['category'].unique()),
                            key="category_filter"
                        )
                    else:
                        category_filter = None
                
                with col3:
                    search_term = st.text_input("🔍 Search in complaints", "", key="search_input")
                
                # Apply filters
                filtered_df = df_analyzed[df_analyzed['sentiment'].isin(sentiment_filter)]
                
                if category_filter and 'category' in df_analyzed.columns:
                    filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
                
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
                display_columns = [st.session_state.selected_column, 'sentiment', 'polarity']
                if 'category_label' in filtered_df.columns:
                    display_columns.append('category_label')
                
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
                    # Generate PDF report
                    if st.button("📑 Generate PDF Report", use_container_width=True):
                        pdf_buffer = generate_pdf_report(stats, results, df_analyzed)
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"complaint_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
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
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your file contains text data and is properly formatted.")
    
    else:
        # Welcome message and instructions
        st.info("👆 Please upload a CSV or Excel file containing complaint data to begin analysis.")
        
        # Sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = pd.DataFrame({
            'complaint_id': [1, 2, 3],
            'complaint_text': [
                "The internet service is very slow and unreliable",
                "Billing department charged me incorrectly",
                "Customer service was helpful and resolved my issue quickly"
            ],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown("""
        **Requirements:**
        - At least one column containing complaint text
        - CSV or Excel format (.csv, .xlsx, .xls)
        - UTF-8 encoding recommended for CSV files
        
        **Features:**
        - 😊 **Sentiment Analysis**: Analyze positive, negative, and neutral sentiments
        - 📂 **Complaint Categorization**: Automatically group similar complaints
        - 🔍 **Keyword Extraction**: Identify most important terms and phrases
        - ☁️ **Word Cloud**: Visual representation of frequent words
        - 🔎 **Interactive Filtering**: Search and filter results
        - 💾 **Export Options**: Download results as CSV or PDF report
        """)


def generate_pdf_report(stats, results, df_analyzed):
    """Generate PDF report of the analysis"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # Title
    pdf.cell(0, 10, 'NLP Analysis of Public Complaints - Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Summary Statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    pdf.cell(0, 8, f'Total Complaints: {stats["total_complaints"]}', 0, 1)
    pdf.cell(0, 8, f'Average Text Length: {stats["avg_text_length"]:.0f} characters', 0, 1)
    pdf.cell(0, 8, f'Average Polarity: {stats["avg_polarity"]:.3f}', 0, 1)
    pdf.cell(0, 8, f'Number of Categories: {stats["num_categories"]}', 0, 1)
    pdf.ln(5)
    
    # Sentiment Distribution
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Sentiment Distribution', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    for sentiment, count in stats['sentiment_distribution'].items():
        percentage = (count / stats['total_complaints']) * 100
        pdf.cell(0, 8, f'{sentiment.capitalize()}: {count} ({percentage:.1f}%)', 0, 1)
    pdf.ln(5)
    
    # Top Keywords
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Top Keywords', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    for i, keyword in enumerate(stats['top_keywords'], 1):
        pdf.cell(0, 8, f'{i}. {keyword}', 0, 1)
    
    # Save to buffer
    pdf_buffer = io.BytesIO()
    pdf_string = pdf.output(dest='S').encode('latin-1')
    pdf_buffer.write(pdf_string)
    pdf_buffer.seek(0)
    
    return pdf_buffer.getvalue()


if __name__ == "__main__":
    main()
