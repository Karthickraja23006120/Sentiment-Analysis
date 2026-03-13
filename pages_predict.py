"""
AI Echo - Predict Sentiment Page
Live prediction using trained ML model
"""
import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

MODEL_DIR = "models"
SENTIMENT_COLORS = {'Positive': '#10B981', 'Neutral': '#F59E0B', 'Negative': '#EF4444'}
SENTIMENT_EMOJIS = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😠'}

def download_nltk():
    for r in ['stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk()

lemmatizer = WordNetLemmatizer()
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = set()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

@st.cache_resource
def load_model():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        return model, vectorizer, le
    except Exception as e:
        return None, None, None

def predict(text, model, vectorizer, le):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred_idx = model.predict(X)[0]
    label = le.inverse_transform([pred_idx])[0]
    # Confidence
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        confidence = {le.inverse_transform([i])[0]: round(float(p) * 100, 1)
                      for i, p in enumerate(proba)}
    elif hasattr(model, 'decision_function'):
        df_scores = model.decision_function(X)[0]
        import numpy as np
        proba = np.exp(df_scores - df_scores.max())
        proba = proba / proba.sum()
        confidence = {le.inverse_transform([i])[0]: round(float(p) * 100, 1)
                      for i, p in enumerate(proba)}
    else:
        confidence = {label: 100.0}
    return label, confidence, cleaned

def show():
    st.markdown('<h2 style="background:linear-gradient(135deg,#f093fb,#f5576c);-webkit-background-clip:text;-webkit-text-fill-color:transparent">🔮 Predict Sentiment</h2>', unsafe_allow_html=True)
    st.markdown("Enter any review text and our AI model will classify its sentiment in real time.")

    model, vectorizer, le = load_model()
    models_ready = model is not None

    if not models_ready:
        st.error("⚠️ Trained models not found. Please run the pipeline first:")
        st.code("python preprocess.py\npython train_model.py", language='bash')
        return

    # Example reviews
    st.markdown("##### 💡 Try an example or type your own:")
    examples = {
        "😊 Positive example": "This app is absolutely amazing! The interface is super clean and it works exactly as expected. Highly recommended!",
        "😐 Neutral example": "The app is okay. It does its job but there are some areas where it could be improved.",
        "😠 Negative example": "Very disappointing experience. The app crashes often and has many bugs. Not worth it at all.",
    }
    col1, col2, col3 = st.columns(3)
    for col, (label, text) in zip([col1, col2, col3], examples.items()):
        if col.button(label, use_container_width=True):
            st.session_state['predict_input'] = text

    # Text area
    user_input = st.text_area(
        "✍️ Enter Review Text:",
        value=st.session_state.get('predict_input', ''),
        height=150,
        placeholder="Type or paste a ChatGPT review here...",
        key="predict_input"
    )

    col_btn, col_clear = st.columns([3, 1])
    predict_clicked = col_btn.button("🔮 Analyze Sentiment", type='primary', use_container_width=True)
    if col_clear.button("🗑️ Clear", use_container_width=True):
        st.session_state['predict_input'] = ''
        st.rerun()

    if predict_clicked and user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            label, confidence, cleaned = predict(user_input, model, vectorizer, le)

        color = SENTIMENT_COLORS.get(label, '#888')
        emoji = SENTIMENT_EMOJIS.get(label, '')

        # Result banner
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}11);
                    border:2px solid {color};border-radius:16px;
                    padding:1.5rem 2rem;text-align:center;margin:1rem 0;">
            <div style="font-size:3rem">{emoji}</div>
            <div style="font-size:2rem;font-weight:800;color:{color}">{label} Sentiment</div>
            <div style="color:#94a3b8;margin-top:0.3rem">
                Confidence: <b style="color:{color}">{confidence.get(label, 0):.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars for all classes
        st.markdown("#### 📊 Model Confidence Breakdown")
        for sent in ['Positive', 'Neutral', 'Negative']:
            pct = confidence.get(sent, 0)
            c = SENTIMENT_COLORS[sent]
            st.markdown(f"""
            <div style="margin-bottom:0.6rem">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span>{SENTIMENT_EMOJIS[sent]} {sent}</span>
                <span style="color:{c};font-weight:700">{pct:.1f}%</span>
            </div>
            <div style="background:#1e293b;border-radius:999px;height:10px">
                <div style="width:{pct}%;background:{c};height:10px;border-radius:999px;
                             transition:width 1s ease"></div>
            </div>
            </div>
            """, unsafe_allow_html=True)

        # NLP processing details
        st.markdown("#### 🧹 NLP Processing Details")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Text:**")
            st.info(user_input[:300] + ("..." if len(user_input) > 300 else ""))
        with col2:
            st.markdown("**After NLP Cleaning:**")
            st.info(cleaned[:300] + ("..." if len(cleaned) > 300 else "") if cleaned else "*(empty after cleaning)*")

    elif predict_clicked:
        st.warning("Please enter some text to analyze.")
