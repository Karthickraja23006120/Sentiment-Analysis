"""
AI Echo: Your Smartest Conversational Partner
Main Streamlit Application — Entry Point
"""
import streamlit as st
import sys
import os

# Make sure we can import page modules from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Echo — Sentiment Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import premium font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Dark premium background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.3);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Sidebar nav items */
.sidebar-nav-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.7rem 1rem; border-radius: 10px;
    cursor: pointer; margin-bottom: 0.3rem;
    transition: all 0.2s ease;
    border: 1px solid transparent;
    font-size: 0.95rem; font-weight: 500;
}
.sidebar-nav-item:hover {
    background: rgba(139,92,246,0.2);
    border-color: rgba(139,92,246,0.4);
}
.sidebar-nav-item.active {
    background: linear-gradient(135deg, rgba(139,92,246,0.3), rgba(168,85,247,0.2));
    border-color: #7c3aed;
    color: #a78bfa !important;
}

/* Main content */
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Cards */
div.stMetric { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 1rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white !important; border: none; border-radius: 10px;
    font-weight: 600; transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(139,92,246,0.4);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10b981, #059669);
}

/* Text input & textarea */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
}

/* Info/Warning/Error boxes */
.stInfo { background: rgba(6,182,212,0.1) !important; border-color: #06b6d4 !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border-color: #f59e0b !important; }
.stError { background: rgba(239,68,68,0.1) !important; border-color: #ef4444 !important; }

/* Slider */
.stSlider > div > div { color: #a78bfa !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #a78bfa !important; }

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Plotly charts */
.js-plotly-plot .plotly { border-radius: 12px; }

/* Divider */
hr { border-color: rgba(139,92,246,0.2) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = {
    "🏠 Home": "home",
    "📊 EDA Dashboard": "eda",
    "💬 Sentiment Insights": "sentiment",
    "🔮 Predict Sentiment": "predict",
    "📈 Model Performance": "model",
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
        <div style="font-size:2.5rem">🤖</div>
        <div style="font-size:1.2rem;font-weight:800;
             background:linear-gradient(135deg,#a78bfa,#f093fb);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            AI Echo
        </div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:0.2rem">
            Sentiment Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 Navigation")
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'

    for label, key in PAGES.items():
        active = "active" if st.session_state['page'] == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state['page'] = key
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;font-size:0.8rem;color:#94a3b8">
    <b style="color:#a78bfa">ChatGPT Reviews Dataset</b><br>
    500 reviews · 12 features<br>
    Platforms: Web, Mobile, App Store<br>
    Models: NB, LR, RF, SVM
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1rem;font-size:0.75rem;color:#4b5563;text-align:center">
    🧠 Powered by NLTK + Scikit-learn<br>
    Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Page Routing ──────────────────────────────────────────────────────────────
current = st.session_state.get('page', 'home')

if current == 'home':
    import pages_home
    pages_home.show()

elif current == 'eda':
    import pages_eda
    pages_eda.show()

elif current == 'sentiment':
    import pages_sentiment
    pages_sentiment.show()

elif current == 'predict':
    import pages_predict
    pages_predict.show()

elif current == 'model':
    import pages_model
    pages_model.show()
