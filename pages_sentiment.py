"""
AI Echo - Sentiment Insights Page
Answers 10 key business questions with interactive charts
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt

SENTIMENT_COLORS = {'Positive': '#10B981', 'Neutral': '#F59E0B', 'Negative': '#EF4444'}
PLOTLY_TEMPLATE = "plotly_dark"

@st.cache_data
def load_data():
    if not os.path.exists("cleaned_data.csv"):
        return None
    return pd.read_csv("cleaned_data.csv", parse_dates=['date'])

def wc_image(text, cmap='coolwarm'):
    if not text.strip():
        return None
    wc = WordCloud(width=600, height=300, background_color='#0f172a',
                   colormap=cmap, max_words=60, collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#0f172a')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return buf

def show():
    st.markdown('<h2 style="background:linear-gradient(135deg,#10b981,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent">💬 Sentiment Insights — 10 Key Questions</h2>', unsafe_allow_html=True)
    st.markdown("Business intelligence powered by NLP — every question answered with data.")

    df = load_data()
    if df is None:
        st.error("⚠️ `cleaned_data.csv` not found. Please run `preprocess.py` first.")
        return

    # Q1 — Overall Sentiment Distribution
    st.markdown("---")
    st.subheader("Q1 — What is the overall sentiment of user reviews?")
    sent_counts = df['sentiment'].value_counts().reset_index()
    sent_counts.columns = ['Sentiment', 'Count']
    sent_counts['Pct'] = (sent_counts['Count'] / len(df) * 100).round(1)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(data=[go.Pie(
            labels=sent_counts['Sentiment'], values=sent_counts['Count'], hole=0.55,
            marker=dict(colors=[SENTIMENT_COLORS.get(s, '#888') for s in sent_counts['Sentiment']]),
            textinfo='label+percent'
        )])
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        for _, row in sent_counts.iterrows():
            color = SENTIMENT_COLORS.get(row['Sentiment'], '#888')
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border-left:4px solid {color};
                        border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.6rem;">
            <b style="color:{color}">{row['Sentiment']}</b>
            <span style="float:right;color:#94a3b8">{row['Count']} reviews ({row['Pct']}%)</span>
            </div>""", unsafe_allow_html=True)
        dominant = sent_counts.iloc[0]['Sentiment']
        st.info(f"💡 Dominant sentiment: **{dominant}** with {sent_counts.iloc[0]['Pct']}% of reviews.")

    # Q2 — Sentiment vs Rating
    st.markdown("---")
    st.subheader("Q2 — How does sentiment vary by rating?")
    cross = df.groupby(['rating', 'sentiment']).size().reset_index(name='count')
    fig2 = px.bar(cross, x='rating', y='count', color='sentiment',
                  color_discrete_map=SENTIMENT_COLORS, barmode='stack',
                  template=PLOTLY_TEMPLATE, labels={'rating':'Star Rating','count':'Reviews'})
    fig2.update_layout(title="Sentiment Distribution Across Ratings", xaxis=dict(tickvals=[1,2,3,4,5]))
    st.plotly_chart(fig2, use_container_width=True)
    st.info("💡 3★ reviews contain all three sentiments, revealing that some users are ambivalent even at neutral ratings.")

    # Q3 — Keywords per Sentiment
    st.markdown("---")
    st.subheader("Q3 — Which keywords are most associated with each sentiment?")
    if 'clean_review' in df.columns:
        tabs = st.tabs(["😊 Positive", "😐 Neutral", "😠 Negative"])
        for tab, sent, cmap in zip(tabs, ['Positive','Neutral','Negative'], ['YlGn','Blues','Reds']):
            with tab:
                text = ' '.join(df[df['sentiment']==sent]['clean_review'].dropna())
                img = wc_image(text, cmap)
                if img:
                    st.image(img, use_column_width=True)

    # Q4 — Sentiment Over Time
    st.markdown("---")
    st.subheader("Q4 — How has sentiment changed over time?")
    df_dated = df.dropna(subset=['date'])
    if len(df_dated) > 5:
        df_dated = df_dated.copy()
        df_dated['month'] = df_dated['date'].dt.to_period('M').astype(str)
        time_sent = df_dated.groupby(['month','sentiment']).size().reset_index(name='count')
        fig4 = px.line(time_sent, x='month', y='count', color='sentiment',
                       color_discrete_map=SENTIMENT_COLORS, markers=True,
                       template=PLOTLY_TEMPLATE)
        fig4.update_layout(title="Monthly Sentiment Trend", xaxis_title="Month", yaxis_title="Reviews")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Not enough dated records for trend analysis.")

    # Q5 — Verified vs Non-Verified
    st.markdown("---")
    st.subheader("Q5 — Do verified users leave more positive or negative reviews?")
    ver_sent = df.groupby(['verified_purchase','sentiment']).size().reset_index(name='count')
    fig5 = px.bar(ver_sent, x='verified_purchase', y='count', color='sentiment',
                  color_discrete_map=SENTIMENT_COLORS, barmode='group',
                  template=PLOTLY_TEMPLATE, labels={'verified_purchase':'Verified Purchase','count':'Reviews'})
    fig5.update_layout(title="Sentiment by Verification Status")
    st.plotly_chart(fig5, use_container_width=True)

    # Q6 — Review Length vs Sentiment
    st.markdown("---")
    st.subheader("Q6 — Are longer reviews more likely to be negative or positive?")
    len_col = 'char_length' if 'char_length' in df.columns else 'review_length'
    fig6 = px.violin(df, x='sentiment', y=len_col, color='sentiment',
                     color_discrete_map=SENTIMENT_COLORS, box=True, points='outliers',
                     template=PLOTLY_TEMPLATE, labels={len_col:'Review Length','sentiment':'Sentiment'})
    fig6.update_layout(title="Review Length Distribution by Sentiment", showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

    # Q7 — Location-wise sentiment
    st.markdown("---")
    st.subheader("Q7 — Which locations show the most positive or negative sentiment?")
    loc_sent = df.groupby(['location','sentiment']).size().reset_index(name='count')
    top_locs = df['location'].value_counts().head(8).index.tolist()
    loc_sent = loc_sent[loc_sent['location'].isin(top_locs)]
    fig7 = px.bar(loc_sent, x='count', y='location', color='sentiment',
                  color_discrete_map=SENTIMENT_COLORS, orientation='h', barmode='stack',
                  template=PLOTLY_TEMPLATE)
    fig7.update_layout(title="Sentiment by Location (Top 8)", yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig7, use_container_width=True)

    # Q8 — Platform vs Sentiment
    st.markdown("---")
    st.subheader("Q8 — Is there a difference in sentiment across platforms?")
    plat_sent = df.groupby(['platform','sentiment']).size().reset_index(name='count')
    fig8 = px.bar(plat_sent, x='platform', y='count', color='sentiment',
                  color_discrete_map=SENTIMENT_COLORS, barmode='group',
                  template=PLOTLY_TEMPLATE)
    fig8.update_layout(title="Platform vs Sentiment — Where is UX Best?")
    st.plotly_chart(fig8, use_container_width=True)

    # Q9 — Version vs Sentiment
    st.markdown("---")
    st.subheader("Q9 — Which ChatGPT versions are associated with higher/lower sentiment?")
    ver_sent = df.groupby(['version','sentiment']).size().reset_index(name='count')
    fig9 = px.bar(ver_sent, x='version', y='count', color='sentiment',
                  color_discrete_map=SENTIMENT_COLORS, barmode='stack',
                  template=PLOTLY_TEMPLATE)
    fig9.update_layout(title="Sentiment by ChatGPT Version")
    st.plotly_chart(fig9, use_container_width=True)

    # Q10 — Negative Feedback Themes
    st.markdown("---")
    st.subheader("Q10 — What are the most common negative feedback themes?")
    if 'clean_review' in df.columns:
        from collections import Counter
        neg_text = ' '.join(df[df['sentiment']=='Negative']['clean_review'].dropna())
        words = [w for w in neg_text.split() if len(w) > 3]
        top15 = Counter(words).most_common(15)
        if top15:
            wdf = pd.DataFrame(top15, columns=['Theme', 'Frequency'])
            fig10 = px.bar(wdf, x='Frequency', y='Theme', orientation='h',
                           color='Frequency', color_continuous_scale='OrRd',
                           template=PLOTLY_TEMPLATE)
            fig10.update_layout(title="Top 15 Negative Feedback Keywords",
                                coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig10, use_container_width=True)
            st.info(f"💡 Top complaint keywords: **{', '.join([w for w,_ in top15[:5]])}** — target these for product improvement.")
