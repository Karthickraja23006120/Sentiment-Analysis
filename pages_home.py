"""
AI Echo - Home Page
"""
import streamlit as st
import pandas as pd
import os

def show():
    st.markdown("""
    <style>
    .hero-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1e1b4b, #2d1b69);
        border: 1px solid #4c1d95; border-radius: 16px;
        padding: 1.5rem; text-align: center;
    }
    .stat-num { font-size: 2.2rem; font-weight: 800; color: #a78bfa; }
    .stat-lbl { font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem; }
    .feature-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 1.2rem; margin-bottom: 0.8rem;
    }
    .badge {
        display:inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero-title">🤖 AI Echo</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Your Smartest Conversational Partner — Sentiment Intelligence Platform</div>', unsafe_allow_html=True)

    # Load data for stats
    stats = {}
    if os.path.exists("cleaned_data.csv"):
        df = pd.read_csv("cleaned_data.csv")
        stats['total'] = len(df)
        dist = df['sentiment'].value_counts()
        stats['pos'] = dist.get('Positive', 0)
        stats['neu'] = dist.get('Neutral', 0)
        stats['neg'] = dist.get('Negative', 0)
        stats['avg_rating'] = df['rating'].mean()
        stats['platforms'] = df['platform'].nunique() if 'platform' in df.columns else 0
    else:
        stats = {'total': 500, 'pos': '—', 'neu': '—', 'neg': '—', 'avg_rating': '—', 'platforms': 6}

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{stats["total"]}</div><div class="stat-lbl">Total Reviews</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:#10b981">{stats["pos"]}</div><div class="stat-lbl">😊 Positive</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:#f59e0b">{stats["neu"]}</div><div class="stat-lbl">😐 Neutral</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:#ef4444">{stats["neg"]}</div><div class="stat-lbl">😠 Negative</div></div>', unsafe_allow_html=True)
    with c5:
        avg = f"{stats['avg_rating']:.2f}★" if isinstance(stats['avg_rating'], float) else stats['avg_rating']
        st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:#f59e0b">{avg}</div><div class="stat-lbl">Avg Rating</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # About Section
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("📋 About This Project")
        st.markdown("""
        **AI Echo** is a full-stack NLP Sentiment Analysis system built on ChatGPT-style user reviews.
        It combines classical machine learning with interactive visualizations to deliver deep insights
        into customer experience and satisfaction.

        **Key Capabilities:**
        """)
        features = [
            ("🧹", "Data Preprocessing", "NLP cleaning: stopword removal, lemmatization, tokenization"),
            ("📊", "Exploratory Analysis", "10 interactive charts revealing trends and patterns"),
            ("💬", "Sentiment Insights", "10 key business questions answered with data"),
            ("🔮", "Live Prediction", "Classify any review as Positive / Neutral / Negative"),
            ("📈", "Model Benchmarking", "Compare 4 ML models: NB, LR, Random Forest, SVM"),
        ]
        for icon, title, desc in features:
            st.markdown(f'<div class="feature-card"><b>{icon} {title}</b><br><span style="color:#94a3b8;font-size:0.9rem">{desc}</span></div>', unsafe_allow_html=True)

    with col2:
        st.subheader("📦 Dataset Overview")
        dataset_info = {
            "Feature": ["date", "title", "review", "rating", "username", "helpful_votes",
                        "review_length", "platform", "language", "location", "version", "verified_purchase"],
            "Description": [
                "Review submission date",
                "Short review headline",
                "Full review text (main input)",
                "Score 1–5 (1=worst, 5=best)",
                "Reviewer identifier",
                "No. of users who found helpful",
                "Character count of review",
                "Web / Mobile / App Store etc.",
                "ISO language code (en, es…)",
                "Country of reviewer",
                "ChatGPT version reviewed",
                "Verified subscriber or not"
            ]
        }
        st.dataframe(pd.DataFrame(dataset_info), use_container_width=True, height=400)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔧 Tech Stack")
    badges = [
        ("#667eea", "Python 3.10+"), ("#10b981", "NLTK"), ("#f59e0b", "Scikit-learn"),
        ("#ef4444", "Streamlit"), ("#8b5cf6", "Plotly"), ("#06b6d4", "Pandas"),
        ("#ec4899", "WordCloud"), ("#14b8a6", "TF-IDF"), ("#f97316", "NLP"),
    ]
    badge_html = "".join([
        f'<span class="badge" style="background:rgba(255,255,255,0.08);border:1px solid {c};color:{c}">{t}</span>'
        for c, t in badges
    ])
    st.markdown(badge_html, unsafe_allow_html=True)

    if os.path.exists("cleaned_data.csv"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🗂️ Sample Cleaned Data")
        df = pd.read_csv("cleaned_data.csv")
        st.dataframe(df[['date','review','rating','sentiment','platform','location','version']].head(10),
                     use_container_width=True)
