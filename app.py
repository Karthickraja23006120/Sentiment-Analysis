import streamlit as st
import pandas as pd
import matplotlib.pyplot as st_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import re

# Load models and data
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_data
def load_data():
    df = pd.read_csv('processed_reviews.csv')
    return df

try:
    model, vectorizer = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading models or data. Ensure train.py has been run. {e}")
    st.stop()

st.title("AI Echo: ChatGPT Sentiment Analysis Dashboard")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA & Insights", "Predict Sentiment"])

if page == "EDA & Insights":
    st.header("Exploratory Data Analysis & Insights")

    # 1. Distribution of review ratings
    st.subheader("1. Distribution of Review Ratings")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='rating', palette='viridis', ax=ax)
    ax.set_title('Review Ratings Count')
    st.pyplot(fig)

    # 2. Helpful votes
    st.subheader("2. Helpful Votes Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['helpful_votes'], bins=20, kde=True, ax=ax)
    ax.set_title('Helpful Votes Distribution')
    st.pyplot(fig)

    # 3. Word clouds
    st.subheader("3. Most Common Keywords")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Positive Reviews (4-5 Stars)")
        positive_text = " ".join(df[df['rating'] >= 4]['clean_review'].dropna())
        if positive_text:
            wordcloud_pos = WordCloud(width=400, height=300, background_color='white').generate(positive_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_pos, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
    with col2:
        st.write("Negative Reviews (1-2 Stars)")
        negative_text = " ".join(df[df['rating'] <= 2]['clean_review'].dropna())
        if negative_text:
            wordcloud_neg = WordCloud(width=400, height=300, background_color='black').generate(negative_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neg, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

    # 4. Average rating over time
    st.subheader("4. Average Rating Over Time")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    time_df = df.groupby(df['date'].dt.date)['rating'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=time_df, x='date', y='rating', marker='o', ax=ax)
    ax.set_title('Average Rating Over Time')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 5. Rating by User Location
    st.subheader("5. Ratings by User Location")
    loc_df = df.groupby('location')['rating'].mean().reset_index().sort_values('rating', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=loc_df, x='location', y='rating', palette='coolwarm', ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Average Rating by Location')
    st.pyplot(fig)

    # 6. Platform Comparison
    st.subheader("6. Platform Comparison (Web vs Mobile)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x='platform', y='rating', errorbar=None, palette='Set2', ax=ax)
    ax.set_title('Average Rating by Platform')
    st.pyplot(fig)

    # 7. Verified Users vs Non-Verified
    st.subheader("7. Verified Purchase Satisfaction")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x='verified_purchase', y='rating', errorbar=None, palette='pastel', ax=ax)
    ax.set_title('Average Rating: Verified vs Non-Verified')
    st.pyplot(fig)

    # 8. Review length vs rating
    st.subheader("8. Review Length vs Rating")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x='rating', y='review_length', palette='muted', ax=ax)
    ax.set_title('Review Length Distribution by Rating')
    st.pyplot(fig)
    
    # 9. Version vs Average Rating
    st.subheader("9. Version vs Average Rating")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df, x='version', y='rating', errorbar=None, palette='magma', ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Average Rating by ChatGPT Version')
    st.pyplot(fig)

elif page == "Predict Sentiment":
    st.header("Predict Sentiment of a Review")
    user_input = st.text_area("Enter your review of ChatGPT here:")
    
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            # Preprocess
            text = user_input.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Vectorize
            text_vectorized = vectorizer.transform([text])
            
            # Predict
            prediction = model.predict(text_vectorized)[0]
            
            st.success(f"Predicted Sentiment: **{prediction}**")
