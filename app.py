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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 NLP Analysis of Public Complaints</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
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
                
                st.success("✅ Analysis completed!")
                
                # Summary Statistics
                st.header("📈 Summary Statistics")
                stats = nlp_utils.get_summary_stats(df, selected_column, results)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Complaints", stats['total_complaints'])
                with col2:
                    st.metric("Avg Text Length", f"{stats['avg_text_length']:.0f} chars")
                with col3:
                    st.metric("Avg Polarity", f"{stats['avg_polarity']:.3f}")
                with col4:
                    st.metric("Categories Found", stats['num_categories'])
                
                # Sentiment Analysis
                st.header("😊 Sentiment Analysis")
                
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
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Polarity histogram
                    fig_hist = px.histogram(
                        results['sentiment'],
                        x='polarity',
                        nbins=20,
                        title="Sentiment Polarity Distribution",
                        labels={'polarity': 'Polarity Score', 'count': 'Frequency'}
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Complaint Categories
                st.header("📂 Complaint Categories")
                
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
                            labels={'x': 'Category', 'y': 'Number of Complaints'}
                        )
                        fig_cat.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_cat, use_container_width=True)
                    
                    with col2:
                        st.subheader("Category Details")
                        for i, label in enumerate(results['categories']['labels']):
                            count = sum(1 for c in results['categories']['clusters'] if c == i)
                            st.write(f"**Category {i+1}:** {label}")
                            st.write(f"Count: {count}")
                            st.write("---")
                
                # Keywords Analysis
                st.header("🔍 Keywords Analysis")
                
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
                        labels={'x': 'TF-IDF Score', 'y': 'Keywords'}
                    )
                    fig_kw.update_layout(height=600)
                    st.plotly_chart(fig_kw, use_container_width=True)
                
                with col2:
                    st.subheader("Top Keywords")
                    for i, (keyword, score) in enumerate(results['keywords'][:10], 1):
                        st.write(f"{i}. **{keyword}** ({score:.3f})")
                
                # Word Cloud
                if show_wordcloud and results['wordcloud']:
                    st.header("☁️ Word Cloud")
                    
                    # Convert wordcloud to image
                    fig_wc, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(results['wordcloud'], interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig_wc)
                
                # Filtering and Search
                st.header("🔎 Data Exploration")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_filter = st.multiselect(
                        "Filter by Sentiment",
                        options=df_analyzed['sentiment'].unique(),
                        default=df_analyzed['sentiment'].unique()
                    )
                
                with col2:
                    if 'category' in df_analyzed.columns:
                        category_filter = st.multiselect(
                            "Filter by Category",
                            options=sorted(df_analyzed['category'].unique()),
                            default=sorted(df_analyzed['category'].unique())
                        )
                    else:
                        category_filter = None
                
                with col3:
                    search_term = st.text_input("Search in complaints", "")
                
                # Apply filters
                filtered_df = df_analyzed[df_analyzed['sentiment'].isin(sentiment_filter)]
                
                if category_filter and 'category' in df_analyzed.columns:
                    filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
                
                if search_term:
                    filtered_df = filtered_df[
                        filtered_df[selected_column].str.contains(search_term, case=False, na=False)
                    ]
                
                st.write(f"**Showing {len(filtered_df)} of {len(df_analyzed)} complaints**")
                
                # Display filtered data
                display_columns = [selected_column, 'sentiment', 'polarity']
                if 'category_label' in filtered_df.columns:
                    display_columns.append('category_label')
                
                st.dataframe(
                    filtered_df[display_columns].head(100),
                    use_container_width=True
                )
                
                # Download Options
                st.header("💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV download
                    csv_data = df_analyzed.to_csv(index=False)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv_data,
                        file_name=f"complaint_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Generate PDF report
                    if st.button("📑 Generate PDF Report"):
                        pdf_buffer = generate_pdf_report(stats, results, df_analyzed)
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"complaint_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                
                # Raw data display
                if show_raw_data:
                    st.header("📊 Raw Analysis Data")
                    
                    with st.expander("Sentiment Data"):
                        st.dataframe(results['sentiment'])
                    
                    with st.expander("Keywords Data"):
                        keywords_df = pd.DataFrame(results['keywords'], columns=['Keyword', 'TF-IDF Score'])
                        st.dataframe(keywords_df)
        
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
