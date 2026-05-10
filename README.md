# 🧠 MindScan AI

AI-based emotional health detection system using NLP and deep learning.

## 🚀 Features
- Emotion detection (Happy, Sad, Angry, Fear, Neutral)
- Risk mapping (mental health interpretation)
- Keyword safety layer for distress signals
- Confidence threshold handling
- Streamlit web app interface

## 🧠 Model
- BiLSTM with GloVe embeddings
- Accuracy: ~84%

## ⚠️ Note
The trained model file (.h5) is not included due to size constraints.  
To run the project, place the model file inside the `model/` folder.

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
