"""
AI Echo - Data Preprocessing Module
Cleans and preprocesses ChatGPT-style reviews dataset
"""

import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_FILE = "chatgpt_style_reviews_dataset.xlsx - Sheet1.csv"
OUTPUT_FILE = "cleaned_data.csv"

# ─── Helpers ──────────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

def get_stopwords():
    try:
        sw = set(stopwords.words('english'))
    except Exception:
        sw = set()
    return sw

STOPWORDS = get_stopwords()

EXTRA_STOPWORDS = {'app', 'use', 'used', 'using', 'one', 'also', 'get', 'got',
                   'would', 'could', 'much', 'many', 'really', 'well', 'even',
                   'way', 'need', 'make', 'made', 'good', 'great', 'bad'}
STOPWORDS.update(EXTRA_STOPWORDS)


def clean_text(text: str) -> str:
    """Full NLP cleaning pipeline: lowercase → remove special chars → remove stopwords → lemmatize."""
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def assign_sentiment(rating: float) -> str:
    """Map numeric rating to sentiment label."""
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'


def parse_date(val):
    """Handle '########' invalid dates and proper date strings."""
    if isinstance(val, str) and val.strip() == '########':
        return pd.NaT
    try:
        return pd.to_datetime(val, format='%m/%d/%Y', errors='coerce')
    except Exception:
        return pd.NaT


# ─── Main Preprocessing ───────────────────────────────────────────────────────
def preprocess():
    print("=" * 60)
    print("  AI Echo — Data Preprocessing")
    print("=" * 60)

    # 1. Load
    print(f"\n[1/7] Loading dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"      Rows: {len(df)}, Columns: {df.shape[1]}")

    # 2. Parse dates
    print("[2/7] Parsing dates (handling '########' entries)...")
    df['date'] = df['date'].apply(parse_date)
    invalid_dates = df['date'].isna().sum()
    print(f"      Invalid/missing dates: {invalid_dates}")

    # 3. Drop rows with missing review text
    print("[3/7] Handling missing values...")
    before = len(df)
    df = df.dropna(subset=['review'])
    df['title'] = df['title'].fillna('')
    df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0).astype(int)
    df['review_length'] = pd.to_numeric(df['review_length'], errors='coerce').fillna(0).astype(int)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    df['rating'] = df['rating'].astype(int)
    print(f"      Dropped {before - len(df)} rows with missing critical fields.")

    # 4. Assign sentiment labels
    print("[4/7] Assigning sentiment labels from ratings...")
    df['sentiment'] = df['rating'].apply(assign_sentiment)
    dist = df['sentiment'].value_counts()
    print(f"      Positive: {dist.get('Positive', 0)} | Neutral: {dist.get('Neutral', 0)} | Negative: {dist.get('Negative', 0)}")

    # 5. Clean text
    print("[5/7] Cleaning review text (NLP pipeline)...")
    df['clean_review'] = df['review'].apply(clean_text)
    empty_after = (df['clean_review'].str.strip() == '').sum()
    print(f"      Reviews with empty text after cleaning: {empty_after}")

    # 6. Feature: Compute actual review length from raw review
    print("[6/7] Computing review character length...")
    df['char_length'] = df['review'].apply(lambda x: len(str(x)))

    # 7. Save
    print(f"[7/7] Saving cleaned dataset to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Done! Cleaned dataset saved with {len(df)} rows.\n")
    print(f"   Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    preprocess()
