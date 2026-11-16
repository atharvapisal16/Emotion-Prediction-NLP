<div align="center">

# 🎭 NLP Emotion Prediction

**Detect emotions in English text using Machine Learning & Natural Language Processing**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end NLP pipeline that classifies text into **6 human emotions** — Joy, Sadness, Anger, Fear, Love, and Surprise — powered by Logistic Regression and TF-IDF vectorization, achieving **86.28% accuracy** on the test set.

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Usage](#-usage) · [Model Details](#-model-details) · [Project Structure](#-project-structure)

</div>

---

## ✨ Features

- 🔮 **Single Text Prediction** — Enter any English sentence and get instant emotion detection with confidence scores
- 📄 **Batch Analysis** — Analyse multiple texts at once via multi-line input or file upload (.txt / .csv)
- 📊 **Interactive Visualisations** — Plotly-powered probability bar charts and radar charts for every prediction
- 🔍 **Word Importance Analysis** — See which words drove the prediction using TF-IDF × model coefficient scoring
- 📜 **Prediction History** — Session-scoped history with CSV export for all past predictions
- 🛠️ **Preprocessing Pipeline View** — Expandable before/after text comparison with token count and latency
- 🧠 **Model Info Sidebar** — Algorithm details, vocabulary size, accuracy metrics, and supported emotions
- 🎨 **Polished Dark Theme UI** — Gradient headers, animated result cards, confidence badges, and responsive layout

---

## 🎯 Demo

### Single Prediction
| Input | Predicted Emotion | Confidence |
|---|---|---|
| *"I am so happy and grateful for everything!"* | 😄 Joy | 75.7% |
| *"I feel completely devastated and broken inside."* | 😢 Sadness | 91.0% |
| *"This makes me absolutely furious and outraged!"* | 😡 Anger | 54.4% |
| *"I was shocked when I heard the unexpected news."* | 😲 Surprise | 67.1% |
| *"I am terrified of what might happen tomorrow."* | 😨 Fear | 89.7% |
| *"I love you more than words can ever express."* | ❤️ Love | 27.8% |

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/NLP_Emotion_Prediction.git
cd NLP_Emotion_Prediction

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Quick Python Usage

```python
import joblib, string
from nltk.corpus import stopwords

# Load artifacts
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("emotion_model.pkl")
mapping = joblib.load("emotion_mapping.pkl")
inv_map = {v: k for k, v in mapping.items()}

# Preprocess
text = "I feel so excited today!"
text = text.lower().translate(str.maketrans("", "", string.punctuation))
text = "".join(ch for ch in text if not ch.isdigit() and ch.isascii())
stop_words = set(stopwords.words("english"))
clean = " ".join(w for w in text.split() if w not in stop_words)

# Predict
features = vectorizer.transform([clean])
prediction = inv_map[model.predict(features)[0]]
confidence = model.predict_proba(features).max()

print(f"Emotion: {prediction}, Confidence: {confidence:.1%}")
# Output → Emotion: joy, Confidence: 95.6%
```

---

## 🧠 Model Details

### Dataset
- **Source:** Text-based emotion dataset (`train.txt`)
- **Size:** 16,000 labelled samples
- **Format:** Semicolon-separated (`text;emotion`)
- **Classes:** 6 emotions — Sadness, Anger, Love, Surprise, Fear, Joy

### Preprocessing Pipeline

```
Raw Text
  │
  ├── 1. Lowercase conversion
  ├── 2. Punctuation removal
  ├── 3. Digit removal
  ├── 4. Non-ASCII / Emoji removal
  └── 5. English stop-word removal (NLTK)
          │
          ▼
     Clean Tokens → TF-IDF Vectorization → Model Prediction
```

### Model Comparison

| # | Model | Vectorizer | Test Accuracy |
|---|---|---|---|
| 1 | Multinomial Naive Bayes | Bag of Words (BoW) | 76.81% |
| 2 | Multinomial Naive Bayes | TF-IDF | 66.09% |
| **3** | **Logistic Regression** | **TF-IDF** | **86.28% ✅** |

> **Logistic Regression + TF-IDF** was selected as the final model for its superior accuracy.

### Model Configuration
- **Algorithm:** Logistic Regression (`max_iter=1000`)
- **Vectorizer:** TF-IDF (default scikit-learn parameters)
- **Train/Test Split:** 80/20 with `random_state=42`

---

## 📁 Project Structure

```
NLP_Emotion_Prediction/
│
├── app.py                    # Streamlit web application (advanced UI)
├── NLP.ipynb                 # Jupyter notebook — EDA, training & evaluation
├── requirements.txt          # Python dependencies with pinned versions
├── train.txt                 # Training dataset (16,000 samples)
│
├── emotion_model.pkl         # Trained Logistic Regression model
├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
├── emotion_mapping.pkl       # Emotion label ↔ integer mapping
│
└── README.md                 # Project documentation
```

---

## 📦 Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.11** | Core programming language |
| **Streamlit** | Interactive web application framework |
| **scikit-learn** | Machine learning (Logistic Regression, TF-IDF) |
| **NLTK** | Natural language processing (stopwords) |
| **Plotly** | Interactive data visualisations |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Joblib** | Model serialisation / deserialisation |

---

## 🔮 Future Improvements

- [ ] Add LSTM / Transformer-based deep learning models for higher accuracy
- [ ] Integrate multi-language support with translation APIs
- [ ] Deploy on Streamlit Community Cloud / Hugging Face Spaces
- [ ] Add real-time social media emotion tracking (Twitter/Reddit)
- [ ] Implement emotion timeline analysis for long documents
- [ ] Add confusion matrix and per-class metrics visualisation

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you found it useful!**

Made with ❤️ using Python & Streamlit

</div>
