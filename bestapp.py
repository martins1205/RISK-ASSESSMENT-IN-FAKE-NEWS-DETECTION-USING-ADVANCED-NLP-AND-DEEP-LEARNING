import streamlit as st
import torch
import re
import string
from transformers import BertTokenizerFast, BertModel
from torch import nn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
import nest_asyncio
nest_asyncio.apply()
# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
# Environment configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCH_FILE"] = "false"

# Page configuration
st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Text Cleaning Components
# =========================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and preprocess input text using NLP techniques"""
    # Text normalization
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    
    # Token processing
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words).strip()

# =========================
# Feature Extraction
# =========================
def analyze_text_features(text):
    """Extract credibility features from cleaned text"""
    features = {
        'emotional_words': len(re.findall(r'\b(urgent|alert|shocking|terrifying)\b', text)),
        'source_references': len(re.findall(r'(according to|source:|reports suggest|cited)', text, re.IGNORECASE)),
        'external_links': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'statistical_terms': len(re.findall(r'\b(\d+%|study shows|research indicates|statistically)\b', text, re.IGNORECASE)),
        'author_credentials': len(re.findall(r'\b(Ph\.D|M\.D|Professor|Researcher|Expert)\b', text))
    }
    return features

# =========================
# Model Architecture
# =========================
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        outputs = self.bert(sent_id, attention_mask=mask)
        cls_hs = outputs.last_hidden_state[:, 0, :]  # Get CLS token embedding
        x = self.fc1(cls_hs)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

# =========================
# Main App
# =========================
def main():
    st.markdown('<div class="title">🛡️  Fake News Detection  AI </div>', unsafe_allow_html=True)

    # Custom CSS
    st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .title { color: #2c3e50; font-size: 2.5rem; text-align: center; }
    .result-card { border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .confidence-meter { height: 20px; border-radius: 10px; background: linear-gradient(90deg, #ff4b4b 0%, #90be6d 100%); }
    .emotional-score { color: #d32f2f; font-weight: bold; }
    .source-score { color: #2e7d32; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # Progress bar during initialization
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Initializing AI Engine...")

    # Model initialization
    try:
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            status_text.text("Using GPU for faster processing...")
        bert = BertModel.from_pretrained('bert-base-uncased')
        model = BERT_Arch(bert).to(device)
        model.load_state_dict(torch.load('best_bert_model.pt', map_location='cpu',weights_only=True))
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model.eval()
        status_text.text("System model is initialized successfully ✅")
        progress_bar.progress(100)
    except Exception as e:
        st.error(f"Initialization Failed: {str(e)}")
        st.stop()

    # User input
    input_text = st.text_area(
        "📝 Paste your news article below:",
        height=250,
        placeholder="Enter full news article text for analysis.",
        help="Minimum 200 characters recommended for accurate analysis."
    )

    if st.button("🔍 Analyze Authenticity", use_container_width=True, disabled=not input_text):
        if len(input_text) < 100:
            st.warning("⚠️ Please provide at least 100 characters.")
            st.stop()

        with st.spinner("🧹 Cleaning and processing text..."):
            cleaned_text = clean_text(input_text)
            features = analyze_text_features(cleaned_text)

        with st.spinner("🔬 Running deep analysis..."):
            try:
                # Tokenization with cleaned text
                tokens = tokenizer.encode_plus(
                    cleaned_text,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )

                # Model prediction
                with torch.no_grad():
                    outputs = model(tokens['input_ids'], tokens['attention_mask'])
                    probs = torch.exp(outputs)
                    fake_prob = probs[0, 0].item()
                    genuine_prob = probs[0, 1].item()
                    confidence = max(fake_prob, genuine_prob)
                    prediction = 0 if fake_prob > genuine_prob else 1

                # Display results
                st.markdown("---")
                result_color = "#2e7d32" if prediction == 0 else "#c62828"
                result_text = "⚠️ Potential Misinformation Detected" if prediction == 0 else "✅ Authentic Content Verified"
                
                st.markdown(f"""
                <div class="result-card" style="border: 2px solid {result_color};">
                    <h2 style="color: {result_color};">{result_text}</h2>
                    <h3>Confidence Level: {confidence:.1%}</h3>
                    <div class="confidence-meter">
                        <div style="width: {confidence*100}%; 
                            height: 100%; 
                            background: {result_color};
                            border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Detailed analysis tabs
                tab1, tab2, tab3 = st.tabs(["📊 Probability Breakdown", "🔍 Content Analysis", "📝 Improvement Suggestions"])

                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Authentic Probability", f"{genuine_prob:.1%}")
                    with col2:
                        st.metric("Misinformation Risk", f"{fake_prob:.1%}")

                with tab2:
                    st.subheader("Content Credibility Features")
                    cols = st.columns(5)
                    feature_data = {
                        "Emotional Language": (features['emotional_words'], 1),
                        "Source References": (features['source_references'], 1),
                        "External Links": (features['external_links'], 1),
                        "Statistical Terms": (features['statistical_terms'], 1),
                        "Author Credentials": (features['author_credentials'], 1)
                    }
                    
                    for col, (label, (count, threshold)) in zip(cols, feature_data.items()):
                        score = max(min(count / threshold, 1.0),0.01) if threshold > 0 else 0.0
                        col.metric(label, f"{count} Detected", delta_color="inverse" if score > 0.5 else "normal")
                        col.progress(score)

                with tab3:
                    st.write("✅ Add verifiable source citations")
                    st.write("✅ Include statistical data references")
                    st.write("✅ Use neutral, objective language")
                    st.write("✅ Provide author qualifications")
                    st.write("✅ Link to original research materials")

                # Technical details
                with st.sidebar.expander("⚙️ Technical Details"):
                    st.write(f"**Model**: BERT-base-uncased\n"
                            f"**Cleaned Text Length**: {len(cleaned_text)} characters\n"
                            f"**Processing Time**: {torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 'CPU'}")

            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
                st.button("🔄 Try Again", type="primary")

    # Sidebar information
    with st.sidebar:
        st.header("About Fake News Detector 🛡️")
        st.markdown("""
        **Version**: 1.0 
        **Accuracy**: 99.97%  
        **Model Type**: BERT Transformers 
        **Languages**: English  
        **Last Updated**: May 2025
        
        
        *created by MARTINS ADEGBAJU*
        """)
        
if __name__ == "__main__":
    main()