"""
AI Echo - EDA Dashboard Page
10 Exploratory Data Analysis visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

SENTIMENT_COLORS = {'Positive': '#10B981', 'Neutral': '#F59E0B', 'Negative': '#EF4444'}
PLOTLY_TEMPLATE = "plotly_dark"

@st.cache_data
def load_data():
    if not os.path.exists("cleaned_data.csv"):
        return None
    df = pd.read_csv("cleaned_data.csv", parse_dates=['date'])
    return df

def wordcloud_figure(text, title, colormap='RdYlGn'):
    if not text.strip():
        return None
    wc = WordCloud(width=700, height=350, background_color='#0f172a',
                   colormap=colormap, max_words=80, collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0f172a')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=13, pad=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return buf

def show():
    st.markdown('<h2 style="background:linear-gradient(135deg,#667eea,#f093fb);-webkit-background-clip:text;-webkit-text-fill-color:transparent">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    st.markdown("Deep-dive into 10 diverse visualizations uncovering patterns in ChatGPT user reviews.")

    df = load_data()
    if df is None:
        st.error("⚠️ `cleaned_data.csv` not found. Please run `preprocess.py` first.")
        st.code("python preprocess.py", language='bash')
        return

    # ── 1. Rating Distribution
    st.markdown("---")
    st.subheader("⭐ 1. Distribution of Review Ratings")
    rating_counts = df['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    rating_counts['Color'] = rating_counts['Rating'].map({
        1: '#EF4444', 2: '#F97316', 3: '#F59E0B', 4: '#84CC16', 5: '#10B981'})
    fig1 = px.bar(rating_counts, x='Rating', y='Count',
                  color='Color', color_discrete_map="identity",
                  text='Count', template=PLOTLY_TEMPLATE,
                  labels={'Rating': 'Star Rating (1–5)', 'Count': 'Number of Reviews'})
    fig1.update_traces(textposition='outside', marker_line_width=0)
    fig1.update_layout(showlegend=False, title="Rating Distribution — Are Users Happy?",
                       xaxis=dict(tickmode='array', tickvals=[1,2,3,4,5]))
    st.plotly_chart(fig1, use_container_width=True)
    pos_pct = (df['rating'] >= 4).mean() * 100
    st.info(f"💡 **Insight**: {pos_pct:.1f}% of reviews are 4★ or above, indicating overall positive sentiment.")

    # ── 2. Helpful Votes Analysis
    st.markdown("---")
    st.subheader("👍 2. Helpful Reviews Analysis")
    threshold = st.slider("Helpful votes threshold:", 1, 50, 10)
    helpful = (df['helpful_votes'] > threshold).sum()
    not_helpful = len(df) - helpful
    fig2 = go.Figure(data=[go.Pie(
        labels=[f'> {threshold} helpful votes', f'≤ {threshold} helpful votes'],
        values=[helpful, not_helpful],
        hole=0.5,
        marker=dict(colors=['#10B981', '#475569']),
    )])
    fig2.update_layout(template=PLOTLY_TEMPLATE,
                       title=f"Reviews with > {threshold} Helpful Votes")
    st.plotly_chart(fig2, use_container_width=True)
    st.info(f"💡 **Insight**: {helpful} reviews ({helpful/len(df)*100:.1f}%) are considered highly helpful with >{threshold} votes.")

    # ── 3. Word Clouds
    st.markdown("---")
    st.subheader("☁️ 3. Most Common Keywords — Positive vs Negative Reviews")
    if 'clean_review' in df.columns:
        col1, col2 = st.columns(2)
        pos_text = ' '.join(df[df['rating'] >= 4]['clean_review'].dropna())
        neg_text = ' '.join(df[df['rating'] <= 2]['clean_review'].dropna())
        with col1:
            wc_buf = wordcloud_figure(pos_text, "Positive Reviews (4-5★)", "YlGn")
            if wc_buf:
                st.image(wc_buf, use_column_width=True)
        with col2:
            wc_buf = wordcloud_figure(neg_text, "Negative Reviews (1-2★)", "OrRd")
            if wc_buf:
                st.image(wc_buf, use_column_width=True)
        st.info("💡 **Insight**: Positive reviews emphasize 'quality', 'easy', 'helpful'. Negative reviews highlight 'bugs', 'crash', 'poor'.")

    # ── 4. Rating Over Time
    st.markdown("---")
    st.subheader("📆 4. Average Rating Over Time")
    df_dated = df.dropna(subset=['date'])
    if len(df_dated) > 5:
        df_dated = df_dated.copy()
        df_dated['month'] = df_dated['date'].dt.to_period('M').astype(str)
        time_df = df_dated.groupby('month')['rating'].agg(['mean','count']).reset_index()
        time_df.columns = ['Month', 'Avg Rating', 'Count']
        time_df = time_df.sort_values('Month')
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=time_df['Month'], y=time_df['Avg Rating'],
                                   mode='lines+markers', line=dict(color='#8b5cf6', width=3),
                                   marker=dict(size=8, color='#a78bfa'),
                                   fill='tozeroy', fillcolor='rgba(139,92,246,0.12)',
                                   name='Avg Rating'))
        fig4.add_hline(y=3, line_dash='dash', line_color='#f59e0b', annotation_text="Neutral Line (3★)")
        fig4.update_layout(template=PLOTLY_TEMPLATE, title="Monthly Average Rating Trend",
                           xaxis_title="Month", yaxis_title="Average Rating",
                           yaxis=dict(range=[1, 5]))
        st.plotly_chart(fig4, use_container_width=True)
        st.info(f"💡 **Insight**: Average rating over {len(time_df)} months shows satisfaction evolution with ChatGPT updates.")
    else:
        st.warning("Not enough dated records for time-series analysis.")

    # ── 5. Rating by Location
    st.markdown("---")
    st.subheader("🌍 5. Average Rating by Location")
    loc_df = df.groupby('location')['rating'].agg(['mean','count']).reset_index()
    loc_df.columns = ['Location', 'Avg Rating', 'Reviews']
    loc_df = loc_df.sort_values('Avg Rating', ascending=True).tail(10)
    fig5 = px.bar(loc_df, x='Avg Rating', y='Location', orientation='h',
                  color='Avg Rating', color_continuous_scale='RdYlGn',
                  text='Reviews', template=PLOTLY_TEMPLATE, range_color=[1,5])
    fig5.update_traces(texttemplate='%{text} reviews', textposition='outside')
    fig5.update_layout(title="Top Locations by Average Rating", coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)
    best_loc = loc_df.iloc[-1]
    st.info(f"💡 **Insight**: {best_loc['Location']} has the highest average rating of {best_loc['Avg Rating']:.2f}★.")

    # ── 6. Platform Comparison
    st.markdown("---")
    st.subheader("🧑‍💻 6. Platform-wise Average Rating")
    plat_df = df.groupby('platform')['rating'].agg(['mean','count']).reset_index()
    plat_df.columns = ['Platform', 'Avg Rating', 'Count']
    plat_df = plat_df.sort_values('Avg Rating', ascending=False)
    fig6 = px.bar(plat_df, x='Platform', y='Avg Rating', color='Avg Rating',
                  color_continuous_scale='Bluered_r', text=plat_df['Avg Rating'].round(2),
                  template=PLOTLY_TEMPLATE)
    fig6.update_traces(textposition='outside')
    fig6.update_layout(title="Which Platform Gets Better Reviews?",
                       yaxis=dict(range=[0, 5.5]), coloraxis_showscale=False)
    st.plotly_chart(fig6, use_container_width=True)

    # ── 7. Verified vs Non-Verified
    st.markdown("---")
    st.subheader("✅ 7. Verified vs Non-Verified User Ratings")
    ver_df = df.groupby('verified_purchase')['rating'].agg(['mean','count']).reset_index()
    ver_df.columns = ['Verified', 'Avg Rating', 'Count']
    fig7 = px.bar(ver_df, x='Verified', y='Avg Rating',
                  color='Verified', color_discrete_map={'Yes': '#10B981', 'No': '#EF4444'},
                  text=ver_df['Avg Rating'].round(2), template=PLOTLY_TEMPLATE)
    fig7.update_traces(textposition='outside')
    fig7.update_layout(title="Are Verified Users Happier?", yaxis=dict(range=[0, 5.5]),
                       showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

    # ── 8. Review Length by Rating
    st.markdown("---")
    st.subheader("🔠 8. Review Length by Rating Category")
    fig8 = px.box(df, x='rating', y='char_length' if 'char_length' in df.columns else 'review_length',
                  color='rating',
                  color_discrete_map={1:'#EF4444',2:'#F97316',3:'#F59E0B',4:'#84CC16',5:'#10B981'},
                  template=PLOTLY_TEMPLATE, labels={'rating':'Star Rating','char_length':'Review Length (chars)'})
    fig8.update_layout(title="Do Unhappy or Happy Users Write Longer Reviews?", showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)
    st.info("💡 **Insight**: Extreme ratings (1★ & 5★) often produce longer reviews as users feel strongly motivated to share.")

    # ── 9. 1-Star Review Keywords
    st.markdown("---")
    st.subheader("💬 9. Most Mentioned Words in 1-Star Reviews")
    if 'clean_review' in df.columns:
        one_star = ' '.join(df[df['rating'] == 1]['clean_review'].dropna())
        col1, col2 = st.columns(2)
        with col1:
            wc_buf = wordcloud_figure(one_star, "1★ Review Keywords", "Reds")
            if wc_buf:
                st.image(wc_buf, use_column_width=True)
        with col2:
            from collections import Counter
            words = [w for w in one_star.split() if len(w) > 3]
            top_words = Counter(words).most_common(12)
            if top_words:
                wdf = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                fig9b = px.bar(wdf, x='Frequency', y='Word', orientation='h',
                               color='Frequency', color_continuous_scale='Reds',
                               template=PLOTLY_TEMPLATE)
                fig9b.update_layout(title="Top 12 Terms in 1★ Reviews",
                                    coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
                st.plotly_chart(fig9b, use_container_width=True)

    # ── 10. Version Analysis
    st.markdown("---")
    st.subheader("📱 10. ChatGPT Version vs Average Rating")
    ver_rating = df.groupby('version')['rating'].agg(['mean','count']).reset_index()
    ver_rating.columns = ['Version', 'Avg Rating', 'Reviews']
    ver_rating = ver_rating.sort_values('Avg Rating', ascending=False)
    fig10 = px.bar(ver_rating, x='Version', y='Avg Rating',
                   color='Avg Rating', color_continuous_scale='RdYlGn',
                   text=ver_rating['Avg Rating'].round(2),
                   hover_data={'Reviews': True},
                   template=PLOTLY_TEMPLATE, range_color=[1, 5])
    fig10.update_traces(textposition='outside')
    fig10.update_layout(title="Which ChatGPT Version Has Best Reviews?",
                        yaxis=dict(range=[0, 5.5]), coloraxis_showscale=False)
    st.plotly_chart(fig10, use_container_width=True)
    best_v = ver_rating.iloc[0]
    st.info(f"💡 **Insight**: Version **{best_v['Version']}** leads with avg rating of **{best_v['Avg Rating']:.2f}★** across {best_v['Reviews']} reviews.")
