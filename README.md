
# 🧠 MindScan AI

MindScan AI is an advanced, multi-modal behavioral analytics and emotional health detection engine. By intersecting text inputs, voice acoustics, historical messaging exports, and digital footprint telemetry (Spotify), it maps a unified trajectory of behavioral wellness over time.

---

## 🚀 Key Features

### 🎙️ Real-Time Self-Assessment
* **Dual-Input Modality:** Evaluate active emotional currents using standard text descriptions or direct vocal reflections.
* **Acoustic Transcribing Engine:** Integrates a locally cached OpenAI **Whisper** pipeline to automatically parse and process recorded audio input data.

### 💬 Chat Analytics Pipeline
* **Auto-Parser Integration:** Automatically detects, structurally standardizes, and parses export files from both **WhatsApp** and **Telegram**.
* **Granular Metrics Dashboard:** Visualizes dynamic emotion trends over time, tracking sender-specific distributions, response lag metrics, and linguistic variance.

### 🎵 Digital Footprint Tracking (Spotify Sub-module)
* **Valence Drop Alert:** Matches cross-referenced cache profiles against streaming history, alerting when musical positivity drops $>30\%$ below your rolling 7-day baseline.
* **Circadian Rhythm Monitor:** Flags anomalous active listening hours spikes occurring during late-night hours (12 AM – 5 AM), tracing potential sleep disruptions.

### 🛡️ Layered Safe-State Architecture
* **Fallback Predictor Router:** Runs verification checks against Deep Learning components; safely drops back to a robust keyword matrix if local artifacts fail label validation.
* **Hardwired Crisis Layer:** Evaluates high-severity hazard indicators across strict safety thresholds, prioritizing critical safety protocols over abstract model confidence values.

---

## 🧠 Core Architecture & Pipelines


```

```
                    ┌─── Text Entry 
                    ├─── Voice Audio ──► [ OpenAI Whisper ]

```

Data Ingestion Channels ┼─── Messaging   ──► [ Auto-Parser Module ]
└─── Spotify Log ──► [ Valence/Circadian Trackers ]
|
▼
┌─────────────────────────────────┐
│ Unified Routing Engine          │
│  ├─ BiLSTM + GloVe (Verified)   │
│  └─ Rule-Based Keyword Fallback │
└────────────────┬────────────────┘
|
▼
[ Safety Layer Overrides / Risk Assessment ]
|
▼
┌─────────────────────────────────────────┐
│ Streamlit Dashboard Analytics Interface │
└─────────────────────────────────────────┘

```

* **Deep Learning Model:** Bidirectional LSTM (BiLSTM) utilizing GloVe word embeddings mapping 5 central classifications (`Happy`, `Sad`, `Angry`, `Fear`, `Neutral`).
* **Audio Engine:** OpenAI Whisper (`tiny` optimization) handling voice configurations.
* **Fallback Logic:** Word-set clustering matrices balancing adaptive heuristics ($0.55$ to $0.92$ confidence weight mapping).
* **Accuracy:** ~84% (when BiLSTM layer is verified).

---

## 📂 Expected Directory Layout

To build out historical indicators and maintain runtime prediction integrity, your project workspace should be structured as follows:

```text
mindscan-ai/
├── app/
│   └── app.py                  # Multi-Modal Streamlit Dashboard App
├── model/
│   ├── bilstm_model.h5         # Trained Deep Learning Model Binary
│   ├── tokenizer.pkl           # Saved Text Tokenizer Artifact
│   ├── label_encoder_v2.pkl    # Serialized Class Categorization Pickle
│   └── track_features_cache.csv # Spotify Valence Tracking Cache Matrix
├── requirements.txt            # Package Dependencies
└── README.md

```

> [!WARNING]
> **Model Files Exclusion:** The trained deep learning binaries (`.h5`, `.pkl`) are excluded from repository tracking due to file size constraints. Ensure your trained dependencies are localized within the `/model` directory prior to initialization.

---

## ▶️ Run Locally

### 1. Environment Setup

Initialize your environment and install the underlying multi-modal system modules:

```bash
# Enter the repository directory
cd mindscan-ai

# Install required packages
pip install -r requirements.txt

```

### 2. Launch the Web Application

Deploy the analytical dashboard locally using Streamlit:

```bash
streamlit run app/app.py

```

---

## 💡 System Workflow

1. **Active Self-Assessment:** Type your current mental state into the text module or use the microphone recorder to test real-time Whisper-based transcription and inference routing.
2. **Historical Ingestion Parsing:** Drop a conversation export file (`.txt`) directly into the sidebar tracker. Use the interface selection dropdowns to break down emotional distributions by specific senders.
3. **Cross-Referenced Wellness Indexes:** Upload exported Spotify history files (`.json`) into the interface. If matching cache data is available, the engine will cross-examine the text analytics and streaming trends to output a unified **Composite Risk Index**.
