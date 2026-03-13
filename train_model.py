"""
AI Echo - Model Training Module
Trains multiple ML models for sentiment classification using TF-IDF features
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score)
from sklearn.preprocessing import label_binarize

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_FILE = "cleaned_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS = ['Negative', 'Neutral', 'Positive']
LABEL_COLORS = {'Negative': '#EF4444', 'Neutral': '#F59E0B', 'Positive': '#10B981'}


def load_data():
    print(f"[1/5] Loading cleaned dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df = df[df['clean_review'].notna() & (df['clean_review'].str.strip() != '')]
    print(f"      Samples: {len(df)} | Sentiment distribution:")
    print(df['sentiment'].value_counts().to_string())
    return df


def build_features(df):
    print("[2/5] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df['clean_review'])
    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(df['sentiment'])
    print(f"      Feature matrix: {X.shape}")
    return X, y, vectorizer, le


def evaluate_model(name, model, X_test, y_test, le, class_names):
    y_pred = model.predict(X_test)

    # For AUC-ROC, we need probability estimates
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, 'decision_function'):
        y_score_raw = model.decision_function(X_test)
        # Softmax normalization
        y_score = np.exp(y_score_raw - y_score_raw.max(axis=1, keepdims=True))
        y_score = y_score / y_score.sum(axis=1, keepdims=True)
    else:
        y_score = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # AUC-ROC (one-vs-rest)
    if y_score is not None:
        try:
            y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
            auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='weighted')
        except Exception:
            auc = None
    else:
        auc = None

    # Per-class metrics
    report = classification_report(y_test, y_pred,
                                    target_names=class_names,
                                    output_dict=True, zero_division=0)

    return {
        'name': name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4) if auc is not None else None,
        'confusion_matrix': cm,
        'classification_report': report,
    }


def train_models(X_train, X_test, y_train, y_test, le):
    print("[3/5] Training models...")
    class_names = le.classes_.tolist()

    models = {
        'Naive Bayes': MultinomialNB(alpha=0.5),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'SVM (LinearSVC)': LinearSVC(
            C=1.0, class_weight='balanced', random_state=42, max_iter=2000),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"      Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test, le, class_names)
        results.append(metrics)
        trained_models[name] = model
        print(f"        Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc_roc']}")

    return results, trained_models


def save_artifacts(results, trained_models, vectorizer, le):
    print("[4/5] Saving model artifacts...")

    # Find best model by F1-score
    best = max(results, key=lambda x: x['f1_score'])
    best_model = trained_models[best['name']]

    # Save vectorizer and label encoder
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    # Save all trained models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(model, os.path.join(MODEL_DIR, f'{safe_name}.pkl'))

    # Save best model reference
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))

    # Save metrics
    metrics_payload = {
        'models': results,
        'best_model': best['name'],
        'labels': le.classes_.tolist(),
    }
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"      Best Model: {best['name']} (F1={best['f1_score']:.4f})")
    print(f"      Saved to: {MODEL_DIR}/")


def main():
    print("=" * 60)
    print("  AI Echo — Model Training")
    print("=" * 60)

    df = load_data()
    X, y, vectorizer, le = build_features(df)

    print("[2.5] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    results, trained_models = train_models(X_train, X_test, y_train, y_test, le)
    save_artifacts(results, trained_models, vectorizer, le)

    print("\n[5/5] Summary:")
    print("-" * 50)
    for r in sorted(results, key=lambda x: -x['f1_score']):
        auc_str = f"{r['auc_roc']:.4f}" if r['auc_roc'] else "N/A"
        print(f"  {r['name']:<22} Acc={r['accuracy']:.4f}  F1={r['f1_score']:.4f}  AUC={auc_str}")
    print("\n✅ Training complete!\n")


if __name__ == "__main__":
    main()
