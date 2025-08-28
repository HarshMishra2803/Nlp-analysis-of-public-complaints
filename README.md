# 📊 NLP Analysis of Public Complaints

A professional web application for analyzing public complaints using advanced Natural Language Processing techniques. Features a modern, presentation-ready dashboard with comprehensive sentiment analysis, automatic categorization, and interactive visualizations.

## 🚀 Features

- **📊 Professional Dashboard**: Modern blue-themed UI perfect for presentations
- **😊 Sentiment Analysis**: Analyze positive, negative, and neutral sentiments using TextBlob
- **📂 Complaint Categorization**: Automatic grouping using K-means clustering on TF-IDF vectors
- **🔍 Keyword Extraction**: Identify most important terms using TF-IDF scoring
- **☁️ Word Cloud Generation**: Visual representation of frequent words
- **🔎 Advanced Filtering**: Search and filter complaints by sentiment, category, and keywords
- **📈 Interactive Visualizations**: Charts and graphs using Plotly and Matplotlib
- **💾 Export Options**: Download results as CSV or comprehensive PDF reports
- **🎨 Professional Theme**: Clean, modern design suitable for academic presentations

## 📋 Project Structure

```
minor project nlp/
├── venv/                           # Virtual environment
├── app.py                          # Main Streamlit dashboard
├── nlp_utils.py                    # NLP processing utilities
├── nlp_pipeline_development.ipynb  # Jupyter notebook for development
├── requirements.txt                # Python dependencies
├── sample_complaints.csv           # Sample data for testing
└── README.md                       # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Virtual environment (already created as `venv/`)

### Installation Steps

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Usage

### Data Requirements
- **File formats**: CSV or Excel (.csv, .xlsx, .xls)
- **Required**: At least one column containing complaint text
- **Recommended**: UTF-8 encoding for CSV files

### Sample Data Format
```csv
complaint_id,complaint_text,date,department
1,"The internet service is extremely slow...",2024-01-15,Internet Services
2,"I was charged twice for the same bill...",2024-01-16,Billing
```

### Dashboard Features

1. **Upload Data**: Use the sidebar to upload your complaint data file
2. **Configure Analysis**: Set number of categories and keywords to extract
3. **Run Analysis**: Click "Run NLP Analysis" to process the data
4. **Explore Results**: View interactive charts, word clouds, and statistics
5. **Filter & Search**: Use filters to explore specific subsets of data
6. **Export Results**: Download processed data as CSV or PDF report

## 🔧 Technical Details

### NLP Pipeline Components

1. **Text Preprocessing**
   - Lowercase conversion
   - Special character removal
   - Whitespace normalization

2. **Sentiment Analysis**
   - Uses TextBlob for polarity and subjectivity scoring
   - Classifies sentiments as positive, negative, or neutral
   - Polarity range: -1 (negative) to +1 (positive)

3. **Complaint Categorization**
   - TF-IDF vectorization with n-grams (1-2)
   - K-means clustering for automatic categorization
   - Configurable number of categories (2-10)

4. **Keyword Extraction**
   - TF-IDF scoring for importance ranking
   - Configurable number of top keywords (5-30)
   - Filters common stop words

5. **Word Cloud Generation**
   - Visual representation of word frequencies
   - Custom stop words filtering
   - Configurable dimensions and color schemes

### Dependencies
- **streamlit**: Web dashboard framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static plotting
- **textblob**: Natural language processing
- **scikit-learn**: Machine learning algorithms
- **wordcloud**: Word cloud generation
- **nltk**: Natural language toolkit
- **fpdf2**: PDF report generation

## 📈 Analysis Outputs

### Summary Statistics
- Total number of complaints
- Average text length
- Overall sentiment polarity
- Number of identified categories

### Visualizations
- Sentiment distribution pie chart
- Polarity histogram
- Category distribution bar chart
- Top keywords horizontal bar chart
- Interactive word cloud

### Export Options
- **CSV**: Complete analyzed dataset with sentiment scores and categories
- **PDF**: Summary report with key statistics and findings

## 🧪 Development

### Jupyter Notebook
The `nlp_pipeline_development.ipynb` notebook contains:
- Step-by-step NLP pipeline development
- Function testing and validation
- Sample data analysis
- Visualization prototypes

### Extending the Project
- Add more sophisticated sentiment analysis models
- Implement topic modeling (LDA)
- Include named entity recognition
- Add multilingual support
- Integrate with databases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is created for educational and portfolio purposes.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed in the virtual environment
2. **File Upload Issues**: Check file format and encoding (UTF-8 recommended)
3. **Memory Issues**: For large datasets, consider sampling or processing in chunks
4. **NLTK Data**: Required NLTK data is downloaded automatically on first run

### Performance Tips
- Use smaller datasets for initial testing
- Reduce number of categories for faster clustering
- Limit keyword extraction for better performance

## 📞 Support

For questions or issues, please check the troubleshooting section or review the Jupyter notebook for detailed implementation examples.

---

**Built with ❤️ for NLP analysis and data visualization**
