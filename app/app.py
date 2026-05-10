import os
import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import tensorflow as tf
import numpy as np
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="MindScan AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }
.hero { text-align: center; padding: 3rem 1rem 2rem; position: relative; }
.hero-badge {
    display: inline-block; background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4); color: #a5b4fc;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; padding: 0.35rem 1rem;
    border-radius: 100px; margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif; font-size: 3.2rem;
    line-height: 1.1; color: #ffffff; margin: 0 0 0.6rem; letter-spacing: -0.02em;
}
.hero-title span { color: #818cf8; font-style: italic; }
.hero-sub { font-size: 1rem; color: #6b7280; font-weight: 300; max-width: 480px; margin: 0 auto; line-height: 1.6; }
.fancy-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent); margin: 2rem 0; }
.stTextArea textarea {
    background: #13131f !important; border: 1px solid #2a2a3d !important;
    border-radius: 14px !important; color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important;
    line-height: 1.7 !important; padding: 1.2rem !important; transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important; }
.stTextArea textarea::placeholder { color: #3d3d56 !important; }
.stTextArea label { color: #9ca3af !important; font-size: 0.85rem !important; font-weight: 500 !important; letter-spacing: 0.05em !important; }
.stButton > button {
    width: 100%; background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.85rem 2rem !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important; font-weight: 600 !important; letter-spacing: 0.03em !important;
    cursor: pointer !important; transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 32px rgba(99, 102, 241, 0.45) !important; }
.result-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }
.result-card {
    background: #13131f; border: 1px solid #2a2a3d; border-radius: 16px;
    padding: 1.4rem 1rem; text-align: center; position: relative; overflow: hidden;
}
.result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.card-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7280; margin-bottom: 0.6rem; }
.card-value { font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: #ffffff; line-height: 1.2; }
.card-value.big { font-size: 2rem; }
.risk-minimal { color: #34d399; }
.risk-low     { color: #60a5fa; }
.risk-medium  { color: #fbbf24; }
.risk-high    { color: #f87171; }
.section-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: #6366f1; margin: 2rem 0 1rem; }
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
.insight-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08));
    border: 1px solid rgba(99,102,241,0.25); border-radius: 14px;
    padding: 1.2rem 1.4rem; margin: 1rem 0; font-size: 0.92rem; color: #c4c4d4; line-height: 1.6;
}
.insight-box strong { color: #a5b4fc; }
.warning-box {
    background: linear-gradient(135deg, rgba(248,113,113,0.08), rgba(239,68,68,0.08));
    border: 1px solid rgba(248,113,113,0.3); border-radius: 14px;
    padding: 1rem 1.4rem; margin: 0.5rem 0; font-size: 0.88rem; color: #fca5a5; line-height: 1.6;
}
.app-footer { text-align: center; padding: 2rem 0 1rem; font-size: 0.75rem; color: #3d3d56; letter-spacing: 0.05em; }
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD LSTM MODEL ──
@st.cache_resource
def load_dl_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "model", "best_lstm_model_v2.h5")
    tokenizer_path = os.path.join(BASE_DIR, "model", "tokenizer_v2.pkl")
    label_path = os.path.join(BASE_DIR, "model", "label_encoder_v2.pkl")
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, le = load_dl_model()
max_len = 100
stop_words = set(stopwords.words('english'))

# ── TEXT PROCESSING ──
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def normalize_text(text):
    text = text.replace("rn", "right now")
    text = text.replace("lol", "laughing")
    text = re.sub(r"(ha)+", "haha", text)
    return text

def map_risk(emotion):
    if emotion in ['Sad', 'Fear']:
        return "High Risk"
    elif emotion == 'Angry':
        return "Medium Risk"
    elif emotion == 'Neutral':
        return "Low Risk"
    else:
        return "Minimal Risk"

# ── KEYWORD SAFETY LAYER ──
high_risk_keywords = [
    'dead', 'death', 'died', 'suicide', 'suicidal', 'kill myself',
    'end my life', 'hopeless', 'worthless', 'extremely sad', 'overwhelmed',
    'cant go on', "can't go on", 'give up', 'no point', 'devastated',
    'no reason to live', 'want to die', 'self harm'
]

def check_keyword_risk(text):
    text_lower = text.lower()
    triggered = [kw for kw in high_risk_keywords if kw in text_lower]
    return triggered

# ── PREDICTION ──
def predict_emotion_and_risk(text):
    cleaned = clean_text(text)
    cleaned = normalize_text(cleaned)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded, verbose=0)
    idx = pred.argmax(axis=1)[0]
    emotion = le.inverse_transform([idx])[0].capitalize()
    confidence = float(pred.max())

    # Keyword safety override
    triggered_keywords = check_keyword_risk(text)
    keyword_triggered = len(triggered_keywords) > 0

    if keyword_triggered:
        emotion = 'Sad'
        confidence = max(confidence, 0.7)

    return emotion, round(confidence * 100, 1), pred[0], keyword_triggered

# ── EMOTION CONFIG ──
emotion_config = {
    'Happy':   {'risk': 'Minimal Risk', 'risk_class': 'risk-minimal', 'emoji': '😊',
                'insight': 'You appear to be in a <strong>positive emotional state</strong>. Keep nurturing the things that bring you joy.'},
    'Neutral': {'risk': 'Low Risk',     'risk_class': 'risk-low',     'emoji': '😐',
                'insight': 'Your emotional tone appears <strong>balanced and stable</strong>. A neutral state often reflects calm and groundedness.'},
    'Angry':   {'risk': 'Medium Risk',  'risk_class': 'risk-medium',  'emoji': '😠',
                'insight': 'Signs of <strong>frustration or anger</strong> detected. Consider stepping away briefly and practising a calming technique.'},
    'Fear':    {'risk': 'High Risk',    'risk_class': 'risk-high',    'emoji': '😨',
                'insight': '<strong>Anxiety or fear</strong> indicators present. If this persists, speaking to someone you trust can help significantly.'},
    'Sad':     {'risk': 'High Risk',    'risk_class': 'risk-high',    'emoji': '😢',
                'insight': 'Emotional distress signals detected. <strong>You are not alone</strong> — reaching out to a friend or counsellor is a sign of strength.'},
}

# ── SESSION STATE ──
if 'history' not in st.session_state:
    st.session_state.history = []

# ── HERO ──
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧠 AI · NLP · Mental Health</div>
    <div class="hero-title">Mind<span>Scan</span> AI</div>
    <div class="hero-sub">Write anything — a journal entry, how your day went, or simply how you feel. Our model reads between the lines.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── INPUT ──
st.markdown('<div class="section-label">Your Entry</div>', unsafe_allow_html=True)
user_input = st.text_area(
    "What's on your mind?",
    height=160,
    placeholder="Today was really overwhelming. I couldn't focus on anything and kept thinking about everything that could go wrong...",
    label_visibility="collapsed"
)

analyse = st.button("Analyse My Text →")

# ── RESULTS ──
if analyse:
    if not user_input.strip():
        st.warning("Please write something first.")
    else:
        emotion, confidence, proba, keyword_triggered = predict_emotion_and_risk(user_input)
        cfg = emotion_config[emotion]
        cfg['risk'] = map_risk(emotion)

        if confidence < 60:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Low confidence prediction.</strong> The model is uncertain about this text — result may not be accurate.
            </div>
            """, unsafe_allow_html=True)

        # Save to history
        st.session_state.history.append({
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Emotion': f"{cfg['emoji']} {emotion}",
            'Confidence': f"{confidence}%",
            'Risk': cfg['risk'],
            'Text Preview': user_input[:60] + "..." if len(user_input) > 60 else user_input
        })

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)

        # Keyword warning
        if keyword_triggered:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Distress signals detected</strong> in your text. 
                If you are going through a difficult time, please consider reaching out to 
                a trusted person or a mental health professional.
            </div>
            """, unsafe_allow_html=True)

        # Result cards
        st.markdown(f"""
        <div class="result-grid">
            <div class="result-card">
                <div class="card-label">Detected Emotion</div>
                <div class="card-value big">{cfg['emoji']}</div>
                <div class="card-value" style="font-size:1.1rem">{emotion}</div>
            </div>
            <div class="result-card">
                <div class="card-label">Confidence</div>
                <div class="card-value big">{confidence}%</div>
            </div>
            <div class="result-card">
                <div class="card-label">Risk Level</div>
                <div class="card-value {cfg['risk_class']}" style="font-size:1rem;margin-top:0.5rem">{cfg['risk']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Insight
        st.markdown(f'<div class="insight-box">💡 {cfg["insight"]}</div>', unsafe_allow_html=True)

        # Confidence bar chart
        st.markdown('<div class="section-label">Model Confidence Breakdown</div>', unsafe_allow_html=True)
        classes = [c.capitalize() for c in le.classes_]
        proba_pct = (proba * 100).round(1)
        proba_df = pd.DataFrame({'Emotion': classes, 'Confidence (%)': proba_pct})
        proba_df = proba_df.sort_values('Confidence (%)', ascending=True)

        colors = ['#6366f1' if e == emotion else '#2a2a3d' for e in proba_df['Emotion']]

        fig = go.Figure(go.Bar(
            x=proba_df['Confidence (%)'],
            y=proba_df['Emotion'],
            orientation='h',
            marker_color=colors,
            text=[f"{v}%" for v in proba_df['Confidence (%)']],
            textposition='outside',
            textfont=dict(color='#9ca3af', size=12)
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(19,19,31,1)',
            font=dict(family='DM Sans', color='#9ca3af'),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 115]),
            yaxis=dict(showgrid=False, tickfont=dict(size=13, color='#e8e8f0')),
            margin=dict(l=10, r=40, t=10, b=10),
            height=240, bargap=0.35
        )
        st.plotly_chart(fig, use_container_width=True)

# ── HISTORY ──
if len(st.session_state.history) > 1:
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Session Emotion Trend</div>', unsafe_allow_html=True)

    hist_df = pd.DataFrame(st.session_state.history)
    conf_values = [float(c.replace('%', '')) for c in hist_df['Confidence']]
    emotions_clean = [e.split(' ')[-1] for e in hist_df['Emotion']]

    emotion_colors = {
        'Happy': '#34d399', 'Neutral': '#60a5fa',
        'Angry': '#fbbf24', 'Fear': '#f87171', 'Sad': '#f87171'
    }
    point_colors = [emotion_colors.get(e, '#6366f1') for e in emotions_clean]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist_df['Time'], y=conf_values,
        mode='lines+markers+text',
        line=dict(color='#6366f1', width=2),
        marker=dict(color=point_colors, size=12, line=dict(color='#0a0a0f', width=2)),
        text=emotions_clean, textposition='top center',
        textfont=dict(size=11, color='#9ca3af'),
        fill='tozeroy', fillcolor='rgba(99,102,241,0.06)'
    ))
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(19,19,31,1)',
        font=dict(family='DM Sans', color='#9ca3af'),
        xaxis=dict(showgrid=False, title='', tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor='#1e1e2e', title='Confidence %', range=[0, 110]),
        margin=dict(l=10, r=10, t=20, b=10), height=280, showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-label">Entry Log</div>', unsafe_allow_html=True)
    st.dataframe(hist_df[['Time', 'Emotion', 'Confidence', 'Risk']], use_container_width=True, hide_index=True)

# ── FOOTER ──
st.markdown("""
<div class="app-footer">
    MindScan AI &nbsp;·&nbsp; AI-Based Emotional Health Detection System &nbsp;·&nbsp; Powered by Bidirectional LSTM + GloVe<br>
    MIET Jammu &nbsp;·&nbsp; B.Tech CSE &nbsp;·&nbsp; Manik Raina (2023a1r180) &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)