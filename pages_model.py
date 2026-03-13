"""
AI Echo - Model Performance Page
Compare all 4 ML models with metrics, confusion matrices, and ROC curves
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

MODEL_DIR = "models"
PLOTLY_TEMPLATE = "plotly_dark"
SENTIMENT_COLORS = {'Positive': '#10B981', 'Neutral': '#F59E0B', 'Negative': '#EF4444'}
MODEL_COLORS = ['#667eea', '#f093fb', '#10b981', '#f59e0b']

@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, 'metrics.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def show():
    st.markdown('<h2 style="background:linear-gradient(135deg,#f59e0b,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent">📈 Model Performance Report</h2>', unsafe_allow_html=True)
    st.markdown("Benchmarking 4 ML classifiers: **Naive Bayes**, **Logistic Regression**, **Random Forest**, **SVM**.")

    data = load_metrics()
    if data is None:
        st.error("⚠️ `models/metrics.json` not found. Please run the pipeline first:")
        st.code("python preprocess.py\npython train_model.py", language='bash')
        return

    models = data['models']
    best_name = data.get('best_model', '')
    labels = data.get('labels', ['Negative', 'Neutral', 'Positive'])

    # ── Best Model Banner
    best = next((m for m in models if m['name'] == best_name), models[0])
    best_color = '#10b981'
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{best_color}22,{best_color}11);
                border:2px solid {best_color};border-radius:16px;
                padding:1.2rem 2rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:2rem">
        <div style="font-size:2.5rem">🏆</div>
        <div>
            <div style="font-size:1.4rem;font-weight:800;color:{best_color}">Best Model: {best_name}</div>
            <div style="color:#94a3b8;margin-top:0.3rem">
                Accuracy: <b style="color:white">{best['accuracy']*100:.2f}%</b> &nbsp;|&nbsp;
                F1-Score: <b style="color:white">{best['f1_score']*100:.2f}%</b> &nbsp;|&nbsp;
                AUC-ROC: <b style="color:white">{f"{best['auc_roc']*100:.2f}%" if best.get('auc_roc') else "N/A"}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Table
    st.subheader("📊 Model Comparison Table")
    metrics_data = []
    for m in models:
        metrics_data.append({
            'Model': ('🏆 ' if m['name'] == best_name else '') + m['name'],
            'Accuracy': f"{m['accuracy']*100:.2f}%",
            'Precision': f"{m['precision']*100:.2f}%",
            'Recall': f"{m['recall']*100:.2f}%",
            'F1-Score': f"{m['f1_score']*100:.2f}%",
            'AUC-ROC': f"{m['auc_roc']*100:.2f}%" if m.get('auc_roc') else 'N/A',
        })
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # ── Side-by-side metric bars
    st.subheader("📉 Visual Metrics Comparison")
    metrics_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig_metrics = go.Figure()
    for i, m in enumerate(models):
        vals = [m[k] * 100 for k in metrics_plot]
        fig_metrics.add_trace(go.Bar(
            name=m['name'], x=metric_labels, y=vals,
            marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            text=[f"{v:.1f}%" for v in vals], textposition='outside'
        ))
    fig_metrics.update_layout(
        barmode='group', template=PLOTLY_TEMPLATE,
        yaxis=dict(range=[0, 110], title="Score (%)"),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # ── Confusion Matrices
    st.subheader("🔢 Confusion Matrices")
    cols = st.columns(2)
    for i, m in enumerate(models):
        with cols[i % 2]:
            cm = np.array(m['confusion_matrix'])
            fig_cm = px.imshow(cm, x=labels, y=labels,
                               color_continuous_scale='Purples',
                               text_auto=True, aspect='auto',
                               template=PLOTLY_TEMPLATE)
            fig_cm.update_layout(
                title=f"{'🏆 ' if m['name']==best_name else ''}{m['name']}",
                xaxis_title="Predicted", yaxis_title="Actual",
                coloraxis_showscale=False, height=280
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Per-Class Metrics
    st.subheader("📋 Per-Class Classification Report")
    selected = st.selectbox("Select Model:", [m['name'] for m in models])
    sel_model = next(m for m in models if m['name'] == selected)
    report = sel_model.get('classification_report', {})
    rows = []
    for cls in labels:
        if cls in report:
            r = report[cls]
            rows.append({
                'Class': cls,
                'Precision': f"{r['precision']*100:.2f}%",
                'Recall': f"{r['recall']*100:.2f}%",
                'F1-Score': f"{r['f1-score']*100:.2f}%",
                'Support': int(r['support'])
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── AUC-ROC Radar
    st.subheader("🕸️ Model Performance Radar")
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    fig_radar = go.Figure()
    for i, m in enumerate(models):
        vals = [
            m['accuracy'] * 100,
            m['precision'] * 100,
            m['recall'] * 100,
            m['f1_score'] * 100,
            (m['auc_roc'] * 100) if m.get('auc_roc') else 0
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill='toself', name=m['name'],
            line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2),
            fillcolor=MODEL_COLORS[i % len(MODEL_COLORS)].replace('#', 'rgba(') + ',0.1)'
            if '#' in MODEL_COLORS[i % len(MODEL_COLORS)] else MODEL_COLORS[i % len(MODEL_COLORS)]
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, template=PLOTLY_TEMPLATE, height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)
