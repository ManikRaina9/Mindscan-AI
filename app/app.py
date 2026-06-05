import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import json
import os
import io

# Import Ingestion & Deep Learning Modules
from streamlit_mic_recorder import mic_recorder
import whisper
import torch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindScan AI — Multi-Modal Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-high   { background:#ffe0e0; color:#7f0000; padding:6px 14px; border-radius:8px; font-weight:600; display:inline-block; }
.risk-medium { background:#fff3cd; color:#7d4800; padding:6px 14px; border-radius:8px; font-weight:600; display:inline-block; }
.risk-low    { background:#d4edda; color:#155724; padding:6px 14px; border-radius:8px; font-weight:600; display:inline-block; }
.risk-minimal{ background:#d1ecf1; color:#0c5460; padding:6px 14px; border-radius:8px; font-weight:600; display:inline-block; }
.alert-box   { background:#f8f9fa; border-left:4px solid #ffc107; padding:12px 16px; border-radius:4px; margin:8px 0; }
.crisis-box  { background:#ffe0e0; border-left:4px solid #dc3545; padding:12px 16px; border-radius:4px; margin:8px 0; }
.index-card  { background:#1e222b; border: 1px solid #3e4451; padding:20px; border-radius:10px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Crisis & Negative Keywords ────────────────────────────────────────────────
CRISIS_KEYWORDS = [
    "want to die", "kill myself", "end my life", "no reason to live",
    "can't go on", "cannot go on", "give up", "worthless", "hopeless",
    "nobody cares", "disappear forever", "harm myself", "hurt myself",
    "suicidal", "don't want to be here", "hate myself", "i'm nothing",
    "everything is pointless", "can't take it anymore", "cannot take it"
]

NEGATIVE_KEYWORDS = [
    "tired", "exhausted", "depressed", "anxious", "stressed", "lonely",
    "sad", "empty", "numb", "broken", "lost", "crying", "scared",
    "worried", "overwhelmed", "trapped", "suffocating", "failing",
    "nothing feels right", "i don't want to talk", "leave me alone"
]

RISK_MAP = {
    "sad":     ("High",    "risk-high"),
    "fear":    ("High",    "risk-high"),
    "angry":   ("Medium",  "risk-medium"),
    "neutral": ("Low",     "risk-low"),
    "happy":   ("Minimal", "risk-minimal")
}

EMOTION_COLORS = {
    "sad":     "#4e8adb",
    "fear":    "#9b59b6",
    "angry":   "#e74c3c",
    "neutral": "#95a5a6",
    "happy":   "#2ecc71",
}

# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Attempt to load BiLSTM model, tokenizer, and label encoder.
    Falls back to rule_based mode if any artifact is missing or broken.
    IMPORTANT: Even if loading succeeds, we verify label order before trusting it.
    """
    try:
        import tensorflow as tf
        import pickle

        model_path        = "model/bilstm_model.h5"
        tokenizer_path    = "model/tokenizer.pkl"
        label_encoder_path = "model/label_encoder_v2.pkl"

        if (os.path.exists(model_path)
                and os.path.exists(tokenizer_path)
                and os.path.exists(label_encoder_path)):

            model = tf.keras.models.load_model(model_path)

            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)

            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)

            # ── CRITICAL LABEL ORDER VERIFICATION ────────────────────────────
            # The model output neuron order MUST match the label encoder class
            # order. Expected alphabetical order after LabelEncoder.fit():
            #   ['angry', 'fear', 'happy', 'neutral', 'sad']  (indices 0-4)
            # If this doesn't match what your encoder has, predictions will be
            # scrambled. We check here and fall back to rule_based if wrong.
            expected_classes = ['angry', 'fear', 'happy', 'neutral', 'sad']
            actual_classes   = list(label_encoder.classes_)

            if actual_classes != expected_classes:
                # Label mismatch detected — model artifacts are inconsistent.
                # The rule-based engine is safer than wrong predictions.
                return (None, None, None), "rule_based"

            return (model, tokenizer, label_encoder), "bilstm"

    except Exception as e:
        print(f"[MindScan AI] Model load failed, using rule-based: {e}")

    return (None, None, None), "rule_based"


@st.cache_resource
def load_whisper_model():
    """Cache Whisper tiny model — loads once per session."""
    return whisper.load_model("tiny")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Rule-Based Emotion Engine (primary reliable fallback) ─────────────────────
# This engine does NOT depend on any trained model artifacts. It uses keyword
# matching and is the safe, predictable option when the BiLSTM is unavailable
# or its labels are unverified.

EMOTION_WORD_SETS = {
    "happy": {
        "happy", "happiness", "joyful", "joy", "excited", "excitement",
        "cheerful", "delighted", "delight", "amazing", "wonderful",
        "fantastic", "great", "awesome", "love", "loving", "smile",
        "smiling", "glad", "thrilled", "ecstatic", "blissful", "elated",
        "content", "pleased", "grateful", "thankful", "celebrate",
        "celebrating", "fun", "positive", "good", "excellent", "perfect"
    },
    "sad": {
        "sad", "sadness", "cry", "crying", "tears", "depressed", "depression",
        "lonely", "loneliness", "miss", "missing", "heartbroken", "miserable",
        "unhappy", "grief", "grieve", "sorrow", "sorrowful", "hopeless",
        "worthless", "empty", "numb", "broken", "hurt", "pain", "suffering",
        "lost", "alone", "isolated", "gloomy", "down", "low", "blue"
    },
    "fear": {
        "scared", "scaredy", "afraid", "fear", "fearful", "anxious",
        "anxiety", "panic", "panicking", "worried", "worry", "stress",
        "stressed", "overwhelmed", "overwhelm", "nervous", "nervousness",
        "terrified", "terror", "dread", "dreading", "uneasy", "apprehensive",
        "phobia", "frightened", "fright", "shaking", "trembling"
    },
    "angry": {
        "angry", "anger", "hate", "hatred", "furious", "fury", "annoyed",
        "annoyance", "frustrated", "frustration", "rage", "raging", "mad",
        "irritated", "irritation", "outraged", "outrage", "disgusted",
        "disgust", "bitter", "resentful", "resentment", "hostile", "hostility",
        "livid", "enraged", "infuriated", "fed up", "sick of"
    }
}

def rule_based_predict(text: str):
    """
    Keyword-based emotion classifier. Returns (emotion, confidence).
    Confidence is a heuristic score between 0.55 and 0.92.
    No model artifacts required — always produces a valid prediction.
    """
    clean = preprocess(text)
    words = set(clean.split())

    scores = {emotion: len(words & keywords)
              for emotion, keywords in EMOTION_WORD_SETS.items()}

    top_emotion = max(scores, key=scores.get)
    total_hits  = sum(scores.values())

    if total_hits == 0:
        return "neutral", 0.55

    # Confidence scales with how many emotion words matched
    confidence = min(0.55 + (scores[top_emotion] / max(total_hits, 1)) * 0.40, 0.92)
    return top_emotion, round(confidence, 2)


# ── BiLSTM Prediction (only used when label order is verified) ────────────────
def bilstm_predict(text: str, model, tokenizer, label_encoder):
    """
    Run a text through the loaded BiLSTM model.
    Only called when load_model() has verified label_encoder.classes_ order.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq   = tokenizer.texts_to_sequences([preprocess(text)])
    seq   = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
    probs = model.predict(seq, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = label_encoder.inverse_transform([idx])[0]
    conf  = round(float(probs[idx]), 2)
    return label, conf


# ── Unified prediction router ─────────────────────────────────────────────────
def predict_emotion(text: str, model, tokenizer, label_encoder, mode: str):
    """
    Routes prediction to BiLSTM or rule-based depending on mode.
    mode == "bilstm"     → use trained model (label order already verified)
    mode == "rule_based" → use keyword engine (always reliable)
    """
    if len(text.strip()) < 3:
        return "neutral", 0.5

    if mode == "bilstm" and model is not None:
        try:
            return bilstm_predict(text, model, tokenizer, label_encoder)
        except Exception:
            pass  # fall through to rule-based on any inference error

    return rule_based_predict(text)


# ── Chat parsers ──────────────────────────────────────────────────────────────
WA_PATTERN = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}:\d{2}(?:\s*[APap][Mm])?)\]\s+([^:]+?):\s+(.*)$"
)

def parse_whatsapp(text: str) -> pd.DataFrame:
    rows        = []
    current_row = None

    static_ignored = {"<media omitted>", "this message was deleted", "null"}
    media_keywords = [
        "image omitted", "sticker omitted", "video omitted",
        "document omitted", "gif omitted", "audio omitted"
    ]

    for line in text.splitlines():
        line_stripped = (
            line.replace("\u200e", "")
                .replace("\u202f", " ")
                .replace("\xa0", " ")
                .strip()
        )
        if not line_stripped:
            continue

        m = WA_PATTERN.match(line_stripped)
        if m:
            if current_row:
                rows.append(current_row)
            date_str, time_str, sender, message = m.groups()
            dt = None
            for fmt in (
                "%d/%m/%y %I:%M:%S %p",
                "%d/%m/%Y %I:%M:%S %p",
                "%d/%m/%y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", fmt)
                    break
                except ValueError:
                    continue

            if dt:
                msg_lower = message.lower().strip()
                is_system = msg_lower in static_ignored or any(
                    kw in msg_lower for kw in media_keywords
                )
                if not is_system:
                    current_row = {
                        "datetime": dt,
                        "sender":   sender.strip(),
                        "message":  message.strip(),
                    }
                else:
                    current_row = None
            else:
                current_row = None
        else:
            if current_row:
                current_row["message"] += "\n" + line_stripped

    if current_row:
        rows.append(current_row)
    return pd.DataFrame(rows)


TG_PATTERN = re.compile(
    r"^\[(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2}:\d{2})\]\s+([^:]+?):\s+(.*)$"
)

def parse_telegram(text: str) -> pd.DataFrame:
    rows        = []
    current_row = None

    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        m = TG_PATTERN.match(line_stripped)
        if m:
            if current_row:
                rows.append(current_row)
            date_str, time_str, sender, message = m.groups()
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M:%S")
                current_row = {
                    "datetime": dt,
                    "sender":   sender.strip(),
                    "message":  message.strip(),
                }
            except ValueError:
                current_row = None
        else:
            if current_row:
                current_row["message"] += "\n" + line_stripped

    if current_row:
        rows.append(current_row)
    return pd.DataFrame(rows)


def auto_parse(t: str):
    """Try both parsers and return whichever extracted more messages."""
    wa = parse_whatsapp(t)
    tg = parse_telegram(t)
    return (wa, "WhatsApp") if len(wa) >= len(tg) else (tg, "Telegram")


# ── Chat analysis pipeline ────────────────────────────────────────────────────
def analyse_chat(
    df: pd.DataFrame, model, tokenizer, label_encoder, mode: str
) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        msg      = row["message"]
        emotion, conf = predict_emotion(msg, model, tokenizer, label_encoder, mode)

        msg_lower   = msg.lower()
        crisis_hits = [kw for kw in CRISIS_KEYWORDS if kw in msg_lower]
        is_crisis   = bool(crisis_hits)
        is_negative = any(kw in msg_lower for kw in NEGATIVE_KEYWORDS)

        # Safety-first: crisis keywords always override model prediction
        if is_crisis:
            emotion = "sad"
            conf    = max(conf, 0.85)

        risk, risk_cls = RISK_MAP.get(emotion, ("Low", "risk-low"))

        results.append({
            "datetime":    row["datetime"],
            "sender":      row["sender"],
            "message":     msg,
            "emotion":     emotion,
            "confidence":  conf,
            "risk":        risk,
            "risk_class":  risk_cls,
            "is_crisis":   is_crisis,
            "is_negative": is_negative,
            "crisis_words": crisis_hits,
        })
    return pd.DataFrame(results)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_hour_bucket(hour: int) -> str:
    if   5  <= hour < 12: return "Morning"
    elif 12 <= hour < 17: return "Afternoon"
    elif 17 <= hour < 22: return "Evening"
    return "Late Night"


def plot_emotion_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    if df.empty:
        return fig

    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    daily = df.groupby(["date", "emotion"]).size().unstack(fill_value=0)

    if daily.empty:
        return fig

    for col in daily.columns:
        color = EMOTION_COLORS.get(col, "#888888")
        ax.fill_between(daily.index, daily[col], alpha=0.2, color=color)
        ax.plot(daily.index, daily[col],
                label=col.capitalize(), color=color, linewidth=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[["top", "right", "left", "bottom"]].set_color("#444444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.title("🧠 MindScan AI — Multi-Modal Behavioral Engine")
    st.caption(
        "Synchronized analysis of communication markers, "
        "streaming logs, and active self-assessment check-ins."
    )

    # Load model once and cache
    model_tuple, mode = load_model()
    model, tokenizer, label_encoder = model_tuple

    # Show which engine is active (useful for demo transparency)
    if mode == "bilstm":
        st.sidebar.success("✅ BiLSTM model loaded and verified")
    else:
        st.sidebar.info("ℹ️ Using rule-based engine (reliable keyword classifier)")

    # Load Whisper once and cache
    whisper_engine = load_whisper_model()

    # ── Sidebar uploads ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📥 Data Ingestion")
        uploaded_chat    = st.file_uploader("1. Upload Chat Export (.txt)", type=["txt"])
        st.divider()
        uploaded_spotify = st.file_uploader(
            "2. Upload Spotify Streaming History (.json)", type=["json"]
        )
        st.info("💡 Export from Spotify Account → Privacy → Download your data")

    # ── REAL-TIME SELF-ASSESSMENT ─────────────────────────────────────────────
    st.subheader("🗣️ Real-Time Self-Assessment")
    st.markdown("Type freely or record your voice to get an instant emotional analysis.")

    col_text, col_voice = st.columns(2)

    with col_text:
        st.markdown("#### 📝 Text Input")
        user_text = st.text_area(
            "Describe how you're feeling today...",
            height=115,
            placeholder="e.g. I'm feeling really stressed about my exams today..."
        )
        if st.button("Analyse Text"):
            if user_text.strip():
                with st.spinner("Analysing..."):
                    u_emotion, u_conf = predict_emotion(
                        user_text, model, tokenizer, label_encoder, mode
                    )
                    u_risk, u_cls = RISK_MAP.get(u_emotion, ("Low", "risk-low"))
                    st.markdown(
                        f"**Result:** Emotion detected as **{u_emotion.capitalize()}** "
                        f"(Confidence: {u_conf:.0%}) &nbsp;→&nbsp; "
                        f"Risk: <span class='{u_cls}'>{u_risk}</span>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Please enter some text first.")

    with col_voice:
        st.markdown("#### 🎤 Voice Input")
        st.markdown(
            "<small style='color:gray;'>Click Start, speak, then click Stop to process.</small>",
            unsafe_allow_html=True
        )
        audio_buffer = mic_recorder(
            start_prompt="🔴 Start Recording",
            stop_prompt="⬛ Stop & Process",
            key="voice_recorder"
        )

        if audio_buffer is not None:
            with st.spinner("Transcribing audio..."):
                try:
                    import tempfile
                    raw_bytes = audio_buffer["bytes"]
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".webm"
                    ) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name

                    result          = whisper_engine.transcribe(tmp_path)
                    transcribed_txt = result.get("text", "").strip()

                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                    if not transcribed_txt:
                        st.error("❌ Could not transcribe — silent or unclear recording.")
                    else:
                        st.info(f"📋 **Transcribed:** \"{transcribed_txt}\"")
                        v_emotion, v_conf = predict_emotion(
                            transcribed_txt, model, tokenizer, label_encoder, mode
                        )
                        v_risk, v_cls = RISK_MAP.get(v_emotion, ("Low", "risk-low"))
                        st.markdown(
                            f"**Result:** Emotion detected as **{v_emotion.capitalize()}** "
                            f"(Confidence: {v_conf:.0%}) &nbsp;→&nbsp; "
                            f"Risk: <span class='{v_cls}'>{v_risk}</span>",
                            unsafe_allow_html=True
                        )
                except Exception as ex:
                    st.error(f"❌ Voice processing error: {ex}")

    st.divider()

    # ── If no chat uploaded, stop here ────────────────────────────────────────
    if uploaded_chat is None:
        st.markdown(
            "### 👆 Upload a chat export in the sidebar to enable "
            "historical conversation analytics."
        )
        return

    # ── Parse and analyse chat ────────────────────────────────────────────────
    raw_text = uploaded_chat.read().decode("utf-8", errors="ignore")
    df_raw, detected_fmt = auto_parse(raw_text)

    if df_raw.empty:
        st.error(
            "Could not parse the uploaded file. "
            "Please check it is a valid WhatsApp or Telegram export."
        )
        return

    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
    st.sidebar.success(f"✅ Parsed {len(df_raw):,} messages ({detected_fmt} format)")

    with st.spinner("Running emotion analysis on chat messages..."):
        df_analyzed = analyse_chat(df_raw, model, tokenizer, label_encoder, mode)

    # ── Spotify pipeline ──────────────────────────────────────────────────────
    spotify_active            = False
    valence_alert_triggered   = False
    late_night_spike_triggered = False
    cache_found               = False
    df_spot                   = pd.DataFrame()

    if uploaded_spotify is not None:
        try:
            spot_data = json.load(uploaded_spotify)
            df_spot   = pd.DataFrame(spot_data)

            if not df_spot.empty:
                # Handle both Spotify export schema variants
                if "endTime" in df_spot.columns:
                    df_spot["dt"] = pd.to_datetime(df_spot["endTime"])
                elif "ts" in df_spot.columns:
                    df_spot["dt"] = pd.to_datetime(df_spot["ts"])

                if "artistName" in df_spot.columns:
                    df_spot["artist"] = df_spot["artistName"]
                elif "master_metadata_album_artist_name" in df_spot.columns:
                    df_spot["artist"] = df_spot["master_metadata_album_artist_name"]
                else:
                    df_spot["artist"] = "Unknown"

                if "trackName" in df_spot.columns:
                    df_spot["track"] = df_spot["trackName"]
                elif "master_metadata_track_name" in df_spot.columns:
                    df_spot["track"] = df_spot["master_metadata_track_name"]
                else:
                    df_spot["track"] = "Unknown"

                df_spot["minutes"] = (
                    df_spot["msPlayed"] / 60000
                    if "msPlayed" in df_spot.columns else 1
                )

                df_spot = df_spot.dropna(subset=["dt"])
                df_spot["hour"] = df_spot["dt"].dt.hour
                df_spot["date"] = df_spot["dt"].dt.date
                spotify_active  = True

                # Optional valence cache
                cache_path = "model/track_features_cache.csv"
                if os.path.exists(cache_path):
                    try:
                        cache_df = pd.read_csv(cache_path)
                        cache_df.columns = (
                            cache_df.columns.str.lower().str.replace(" ", "_")
                        )
                        c_track  = next(
                            (c for c in ["track_name", "track"] if c in cache_df.columns),
                            None
                        )
                        c_artist = next(
                            (c for c in ["track_artist", "artist_name", "artist"]
                             if c in cache_df.columns), None
                        )
                        if c_track and c_artist and "valence" in cache_df.columns:
                            lookup = cache_df[[c_track, c_artist, "valence"]].copy()
                            lookup.columns = ["track", "artist", "valence"]
                            lookup = lookup.drop_duplicates(subset=["track", "artist"])
                            df_spot = df_spot.merge(
                                lookup, on=["track", "artist"], how="left"
                            )
                            if (
                                "valence" in df_spot.columns
                                and df_spot["valence"].notna().mean() >= 0.20
                            ):
                                cache_found = True
                    except Exception:
                        pass

                # Valence drop detection (only if cache matched enough tracks)
                if cache_found:
                    df_spot["valence"] = df_spot["valence"].fillna(0.5)
                    daily_audio = (
                        df_spot.groupby("date")
                        .agg(valence_avg=("valence", "mean"))
                        .sort_index()
                    )
                    if len(daily_audio) >= 2:
                        daily_audio["baseline"] = (
                            daily_audio["valence_avg"]
                            .shift(1)
                            .rolling(window=7, min_periods=1)
                            .mean()
                        )
                        daily_audio["drop"] = (
                            (daily_audio["baseline"] - daily_audio["valence_avg"])
                            / daily_audio["baseline"]
                        )
                        if daily_audio.iloc[-1]["drop"] > 0.30:
                            valence_alert_triggered = True

                # Late-night detection (independent of valence cache)
                daily_time = (
                    df_spot.groupby("date")
                    .agg(
                        track_count=("track", "count"),
                        late_night_count=(
                            "hour", lambda x: int(sum((x >= 0) & (x <= 5)))
                        ),
                    )
                    .sort_index()
                )
                if len(daily_time) >= 1:
                    last = daily_time.iloc[-1]
                    if last["track_count"] > 0:
                        if last["late_night_count"] / last["track_count"] > 0.30:
                            late_night_spike_triggered = True

        except Exception as e:
            st.sidebar.error(f"Spotify parsing error: {e}")

    # ── Composite index (shown when Spotify also uploaded) ────────────────────
    if spotify_active:
        st.subheader("🧠 Integrated Behavioral Insights")

        chat_dominant_emotion = df_analyzed["emotion"].value_counts().idxmax()
        chat_dominant_risk    = df_analyzed["risk"].value_counts().idxmax()

        if cache_found and (valence_alert_triggered or chat_dominant_risk == "High"):
            index_str, index_css = "Elevated Passive Vulnerability Risk", "risk-high"
        elif chat_dominant_risk == "High":
            index_str, index_css = "Elevated Stress / Communication Risk", "risk-high"
        elif late_night_spike_triggered:
            index_str, index_css = "Altered Rhythm / Possible Sleep Disruption", "risk-medium"
        else:
            index_str, index_css = "Baseline Adaptive Balance", "risk-low"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div class='index-card'><h4>Chat Risk Vector</h4><br>"
                f"<span class='{RISK_MAP[chat_dominant_emotion][1]}'>"
                f"{chat_dominant_risk}</span></div>",
                unsafe_allow_html=True
            )
        with c2:
            if not cache_found:
                v_text, v_css = "VALENCE DATA UNAVAILABLE", "risk-medium"
            elif valence_alert_triggered:
                v_text, v_css = "ANOMALOUS DROP DETECTED", "risk-high"
            else:
                v_text, v_css = "STABLE", "risk-minimal"
            st.markdown(
                f"<div class='index-card'><h4>7-Day Valence Status</h4><br>"
                f"<span class='{v_css}'>{v_text}</span></div>",
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"<div class='index-card'><h4>Composite Risk Index</h4><br>"
                f"<span class='{index_css}'>{index_str}</span></div>",
                unsafe_allow_html=True
            )
        st.divider()

    # ── Sender filter ─────────────────────────────────────────────────────────
    senders   = ["All"] + sorted(df_analyzed["sender"].unique().tolist())
    user_sel  = st.selectbox("Filter Dashboard by Sender:", senders)
    df_view   = (
        df_analyzed if user_sel == "All"
        else df_analyzed[df_analyzed["sender"] == user_sel].copy()
    )

    # ── Core chat metrics ─────────────────────────────────────────────────────
    st.subheader("📊 Core Chat Analysis")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Messages",        f"{len(df_view):,}")
    m2.metric("Dominant Emotion",      df_view["emotion"].value_counts().idxmax().capitalize())
    m3.metric("Avg Confidence",        f"{df_view['confidence'].mean():.1%}")

    st.pyplot(plot_emotion_trend(df_view))
    st.divider()

    # ── Emotion distribution by sender ────────────────────────────────────────
    st.subheader("😊 Emotion Distribution by Sender")
    emo_pct = (
        df_analyzed
        .groupby("sender")["emotion"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .unstack(fill_value=0)
    )
    emo_pct.columns = [f"{c.capitalize()} (%)" for c in emo_pct.columns]
    st.dataframe(emo_pct, use_container_width=True)
    st.divider()

    # ── Response-time metrics ─────────────────────────────────────────────────
    st.subheader("⚡ Response Time Metrics")
    resp = []
    for i in range(1, len(df_analyzed)):
        if df_analyzed.iloc[i]["sender"] != df_analyzed.iloc[i - 1]["sender"]:
            delta = (
                df_analyzed.iloc[i]["datetime"]
                - df_analyzed.iloc[i - 1]["datetime"]
            ).total_seconds() / 60.0
            resp.append({
                "responder": df_analyzed.iloc[i]["sender"],
                "delay":     delta,
            })

    if resp:
        df_reply = pd.DataFrame(resp)
        r1, r2, r3 = st.columns(3)
        r1.metric("Average Response Time", f"{df_reply['delay'].mean():.1f} mins")
        r2.metric(
            "Fastest Responder",
            df_reply.groupby("responder")["delay"].mean().idxmin()
        )
        r3.metric(
            "Longest Gap",
            f"{df_reply['delay'].max() / 60:.1f} hours"
        )
    st.divider()

    # ── Spotify sub-module display ────────────────────────────────────────────
    if spotify_active and not df_spot.empty:
        st.subheader("🎵 Spotify Streaming Analytics")

        if cache_found and valence_alert_triggered:
            st.markdown(
                '<div class="crisis-box">🚨 <strong>Valence Alert:</strong> '
                "Musical positivity has dropped >30% compared to your 7-day "
                "rolling baseline. This may correlate with mood changes.</div>",
                unsafe_allow_html=True
            )
        if late_night_spike_triggered:
            st.markdown(
                '<div class="alert-box">⚠️ <strong>Late-Night Alert:</strong> '
                "More than 30% of today's listening occurred between midnight "
                "and 5 AM. Possible disrupted sleep pattern.</div>",
                unsafe_allow_html=True
            )

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("#### Time-of-Day Listening Distribution")
            df_spot["tod"] = df_spot["hour"].apply(get_hour_bucket)
            tod = (
                df_spot["tod"]
                .value_counts(normalize=True)
                .mul(100)
                .round(1)
                .reindex(["Morning", "Afternoon", "Evening", "Late Night"], fill_value=0)
            )
            st.dataframe(
                tod.to_frame(name="Share (%)"), use_container_width=True
            )

        with s2:
            st.markdown("#### Top 5 Artists")
            top_art = (
                df_spot["artist"]
                .value_counts()
                .head(5)
                .to_frame(name="Play Count")
            )
            st.dataframe(top_art, use_container_width=True)
            st.metric(
                "Total Listening Time",
                f"{df_spot['minutes'].sum():,.0f} mins"
            )
        st.divider()

    # ── Crisis alerts ─────────────────────────────────────────────────────────
    crisis_rows = df_view[df_view["is_crisis"] == True]
    if not crisis_rows.empty:
        st.markdown("#### 🔴 Crisis Keyword Alerts")
        for _, r in crisis_rows.iterrows():
            st.markdown(
                f'<div class="crisis-box"><strong>{r["sender"]}</strong>: '
                f'"{r["message"]}"</div>',
                unsafe_allow_html=True
            )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.info(
        "⚖️ **Disclaimer:** MindScan AI is an early-warning behavioral analytics "
        "tool only. It is not a medical diagnostic system and must not be used as "
        "a substitute for professional mental health support."
    )


if __name__ == "__main__":
    main()
