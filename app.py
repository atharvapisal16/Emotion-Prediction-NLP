"""
Emotion Predictor — Advanced NLP Streamlit Application
========================================================
Predicts one of six emotions (sadness, anger, love, surprise, fear, joy)
from free-form English text using a Logistic Regression model trained on
TF-IDF features over a 16 000-sample corpus.

Features
--------
• Single-text and batch (multi-line / file upload) prediction
• Confidence-aware output with contextual emoji, colour & description
• Interactive Plotly probability radar chart + horizontal bar breakdown
• Per-word importance highlights via TF-IDF coefficient analysis
• Session-scoped prediction history with downloadable CSV export
• Sidebar with model metadata, dataset stats & preprocessing pipeline info
• Responsive, polished dark-theme UI with smooth CSS animations
"""

from __future__ import annotations

import io
import os
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

EMOTION_META: Dict[str, dict] = {
    "joy":      {"emoji": "😄", "colour": "#22c55e", "description": "Happiness, delight, or pleasure"},
    "sadness":  {"emoji": "😢", "colour": "#3b82f6", "description": "Grief, sorrow, or unhappiness"},
    "anger":    {"emoji": "😡", "colour": "#ef4444", "description": "Frustration, rage, or irritation"},
    "fear":     {"emoji": "😨", "colour": "#a855f7", "description": "Anxiety, dread, or apprehension"},
    "love":     {"emoji": "❤️", "colour": "#ec4899", "description": "Affection, fondness, or tenderness"},
    "surprise": {"emoji": "😲", "colour": "#f59e0b", "description": "Astonishment or unexpectedness"},
}

CONFIDENCE_THRESHOLDS = {
    "high":   (0.70, "High confidence"),
    "medium": (0.45, "Moderate confidence"),
    "low":    (0.00, "Low confidence — prediction may be uncertain"),
}

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Emotion Predictor | NLP",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — polished dark-theme with animations
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* ---- global ---- */
        .stApp { background: #0b1120; }
        .block-container { max-width: 960px; padding-top: 2rem; }

        /* ---- header ---- */
        .app-header {
            text-align: center;
            padding: 2rem 0 1rem;
        }
        .app-header h1 {
            font-size: 2.6rem;
            background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: .25rem;
        }
        .app-header p {
            color: #94a3b8;
            font-size: 1.05rem;
        }

        /* ---- result card ---- */
        .result-card {
            margin-top: 1.5rem;
            padding: 2rem 2.5rem;
            border-radius: 16px;
            background: linear-gradient(145deg, #131a2b, #1a2236);
            border: 1px solid #1e293b;
            text-align: center;
            animation: fadeSlide .45s ease-out;
        }
        .result-emoji { font-size: 3.4rem; margin-bottom: .5rem; }
        .result-emotion {
            font-size: 2.2rem;
            font-weight: 700;
            text-transform: capitalize;
        }
        .result-description {
            color: #94a3b8;
            margin-top: .25rem;
            font-size: .95rem;
        }

        /* confidence badge */
        .confidence-badge {
            display: inline-block;
            margin-top: .75rem;
            padding: .35rem 1.1rem;
            border-radius: 999px;
            font-size: .85rem;
            font-weight: 600;
        }
        .conf-high   { background: rgba(34,197,94,.15);  color: #22c55e; border: 1px solid rgba(34,197,94,.3); }
        .conf-medium { background: rgba(245,158,11,.15); color: #f59e0b; border: 1px solid rgba(245,158,11,.3); }
        .conf-low    { background: rgba(239,68,68,.15);  color: #ef4444; border: 1px solid rgba(239,68,68,.3); }

        /* ---- word highlight chips ---- */
        .word-chip {
            display: inline-block;
            padding: .2rem .55rem;
            margin: .15rem .1rem;
            border-radius: 6px;
            font-size: .82rem;
            font-weight: 500;
        }

        /* ---- animations ---- */
        @keyframes fadeSlide {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ---- sidebar tweaks ---- */
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #e2e8f0;
        }

        /* ---- metric cards ---- */
        .metric-row {
            display: flex;
            gap: .75rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            flex: 1;
            background: #131a2b;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #e2e8f0;
        }
        .metric-label {
            font-size: .78rem;
            color: #64748b;
            margin-top: .2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data-class for a single prediction result
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    text: str
    processed_text: str
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    top_words: List[Tuple[str, float]]
    latency_ms: float


# ---------------------------------------------------------------------------
# Cached loading helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_stop_words() -> set:
    """Download (once) and return the NLTK English stop-words set."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple:
    """Load the TF-IDF vectorizer, logistic-regression model, and label map."""
    vectorizer = joblib.load(APP_DIR / "tfidf_vectorizer.pkl")
    model = joblib.load(APP_DIR / "emotion_model.pkl")
    emotion_mapping: dict = joblib.load(APP_DIR / "emotion_mapping.pkl")
    inverse = {v: k for k, v in emotion_mapping.items()}
    return vectorizer, model, inverse


# ---------------------------------------------------------------------------
# Text preprocessing (mirrors notebook pipeline exactly)
# ---------------------------------------------------------------------------


def preprocess_text(text: str) -> str:
    """Lowercase → remove punctuation → remove digits → remove non-ASCII → remove stop-words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(ch for ch in text if not ch.isdigit())
    text = "".join(ch for ch in text if ch.isascii())
    sw = load_stop_words()
    return " ".join(w for w in text.split() if w not in sw)


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------


def predict_emotion(text: str, vectorizer, model, inverse_map) -> PredictionResult:
    """Run the full pipeline and return a rich PredictionResult."""
    t0 = time.perf_counter()
    processed = preprocess_text(text)

    features = vectorizer.transform([processed])
    pred_id = model.predict(features)[0]
    emotion = inverse_map.get(int(pred_id), "unknown")

    # Probability distribution
    probs_arr = model.predict_proba(features)[0]
    prob_dict = {
        inverse_map.get(int(cid), str(cid)): float(p)
        for cid, p in zip(model.classes_, probs_arr)
    }
    confidence = prob_dict.get(emotion, 0.0)

    # Per-word importance: TF-IDF weight × model coefficient for predicted class
    top_words = _extract_word_importance(processed, vectorizer, model, pred_id)

    latency = (time.perf_counter() - t0) * 1000
    return PredictionResult(
        text=text,
        processed_text=processed,
        emotion=emotion,
        confidence=confidence,
        probabilities=prob_dict,
        top_words=top_words,
        latency_ms=latency,
    )


def _extract_word_importance(
    processed: str, vectorizer, model, pred_class
) -> List[Tuple[str, float]]:
    """Return top contributing words sorted by (tfidf × coefficient) score."""
    if not processed.strip():
        return []
    feature_names = vectorizer.get_feature_names_out()
    tfidf_vec = vectorizer.transform([processed])
    class_idx = list(model.classes_).index(pred_class)
    coefficients = model.coef_[class_idx]

    word_scores: List[Tuple[str, float]] = []
    for idx in tfidf_vec.nonzero()[1]:
        word = feature_names[idx]
        score = float(tfidf_vec[0, idx] * coefficients[idx])
        word_scores.append((word, score))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:12]


# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------


def confidence_level(conf: float) -> Tuple[str, str, str]:
    """Return (level_name, css_class, description)."""
    if conf >= 0.70:
        return "high", "conf-high", CONFIDENCE_THRESHOLDS["high"][1]
    if conf >= 0.45:
        return "medium", "conf-medium", CONFIDENCE_THRESHOLDS["medium"][1]
    return "low", "conf-low", CONFIDENCE_THRESHOLDS["low"][1]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_result_card(result: PredictionResult) -> None:
    meta = EMOTION_META.get(result.emotion, {"emoji": "❓", "colour": "#94a3b8", "description": ""})
    _, css_cls, conf_desc = confidence_level(result.confidence)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-emoji">{meta["emoji"]}</div>
            <div class="result-emotion" style="color:{meta['colour']}">{result.emotion}</div>
            <div class="result-description">{meta["description"]}</div>
            <div class="confidence-badge {css_cls}">{result.confidence:.1%} — {conf_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_charts(probs: Dict[str, float]) -> None:
    """Show a horizontal bar chart + radar chart side by side."""
    df = pd.DataFrame(
        [{"Emotion": k.capitalize(), "Probability": v} for k, v in probs.items()]
    ).sort_values("Probability", ascending=True)

    colours = [EMOTION_META.get(e.lower(), {}).get("colour", "#64748b") for e in df["Emotion"]]

    col1, col2 = st.columns(2)

    with col1:
        fig_bar = go.Figure(
            go.Bar(
                x=df["Probability"],
                y=df["Emotion"],
                orientation="h",
                marker_color=colours,
                text=[f"{v:.1%}" for v in df["Probability"]],
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            title="Probability Breakdown",
            xaxis_title="Probability",
            yaxis_title="",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=10, r=10, t=40, b=30),
            xaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        emotions_ordered = list(probs.keys())
        values = [probs[e] for e in emotions_ordered]
        radar_colours = [EMOTION_META.get(e, {}).get("colour", "#64748b") for e in emotions_ordered]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=[e.capitalize() for e in emotions_ordered] + [emotions_ordered[0].capitalize()],
                fill="toself",
                fillcolor="rgba(99,102,241,.18)",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=6, color=radar_colours + [radar_colours[0]]),
            )
        )
        fig_radar.update_layout(
            title="Emotion Radar",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=40, r=40, t=40, b=30),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)


def render_word_importance(result: PredictionResult) -> None:
    """Show which words pushed the prediction toward the chosen emotion."""
    if not result.top_words:
        return

    emotion_colour = EMOTION_META.get(result.emotion, {}).get("colour", "#6366f1")
    max_score = max(abs(s) for _, s in result.top_words) if result.top_words else 1

    st.markdown("##### 🔍 Key Words Driving the Prediction")
    chips_html = ""
    for word, score in result.top_words:
        norm = max(0.15, min(1.0, abs(score) / max_score))
        bg = emotion_colour if score > 0 else "#64748b"
        chips_html += (
            f'<span class="word-chip" style="background:{bg}{int(norm*60):02x};'
            f'color:#e2e8f0;border:1px solid {bg}40">{word}'
            f'<span style="font-size:.7rem;color:#94a3b8;margin-left:.3rem">'
            f'{score:+.3f}</span></span>'
        )
    st.markdown(chips_html, unsafe_allow_html=True)

    st.caption(
        "Scores = TF-IDF weight × model coefficient. "
        "Positive scores push toward the predicted emotion; "
        "higher magnitude = stronger influence."
    )


def render_preprocessing_details(result: PredictionResult) -> None:
    """Expandable section showing the cleaning pipeline output."""
    with st.expander("🛠️ Preprocessing Details", expanded=False):
        c1, c2 = st.columns(2)
        c1.markdown("**Original text**")
        c1.code(result.text, language=None)
        c2.markdown("**After preprocessing**")
        c2.code(result.processed_text if result.processed_text.strip() else "(empty after cleaning)", language=None)

        st.markdown(
            f"**Token count:** {len(result.processed_text.split()) if result.processed_text.strip() else 0}  •  "
            f"**Inference latency:** {result.latency_ms:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Sidebar — model info & history
# ---------------------------------------------------------------------------


def render_sidebar(model, vectorizer, inverse_map) -> None:
    with st.sidebar:
        st.markdown("### 🧠 Model Information")

        vocab_size = len(vectorizer.get_feature_names_out())
        num_classes = len(model.classes_)
        model_type = type(model).__name__

        st.markdown(
            f"""
            | Property | Value |
            |---|---|
            | **Algorithm** | {model_type} |
            | **Vectorizer** | TF-IDF |
            | **Vocabulary** | {vocab_size:,} terms |
            | **Classes** | {num_classes} emotions |
            | **Training samples** | 16,000 |
            | **Test accuracy** | 86.28% |
            """
        )

        st.markdown("---")
        st.markdown("### 🏷️ Supported Emotions")
        for emo, meta in EMOTION_META.items():
            st.markdown(f"{meta['emoji']}  **{emo.capitalize()}** — {meta['description']}")

        st.markdown("---")
        st.markdown("### ⚙️ Preprocessing Pipeline")
        st.markdown(
            """
            1. Lowercase conversion
            2. Punctuation removal
            3. Digit removal
            4. Non-ASCII / emoji removal
            5. English stop-word removal
            """
        )

        # History section
        st.markdown("---")
        st.markdown("### 📜 Prediction History")
        if st.session_state.get("history"):
            hist_df = pd.DataFrame(
                [
                    {
                        "Text": r.text[:60] + ("…" if len(r.text) > 60 else ""),
                        "Emotion": r.emotion.capitalize(),
                        "Confidence": f"{r.confidence:.1%}",
                        "Latency": f"{r.latency_ms:.0f}ms",
                    }
                    for r in reversed(st.session_state["history"])
                ]
            )
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            csv = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download History CSV", csv, "prediction_history.csv", "text/csv")

            if st.button("🗑️ Clear History"):
                st.session_state["history"] = []
                st.rerun()
        else:
            st.caption("No predictions yet.")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    # Initialise session state
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Load model artifacts
    try:
        vectorizer, model, inverse_map = load_artifacts()
    except Exception as exc:
        st.error(f"❌ Failed to load model artifacts: {exc}")
        st.info("Ensure `tfidf_vectorizer.pkl`, `emotion_model.pkl` and `emotion_mapping.pkl` are in the app directory.")
        st.stop()

    # Sidebar
    render_sidebar(model, vectorizer, inverse_map)

    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1>🎭 Emotion Predictor</h1>
            <p>Detect emotions in English text using NLP & Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input mode tabs
    tab_single, tab_batch = st.tabs(["✏️ Single Text", "📄 Batch Analysis"])

    # ------------------------------------------------------------------ single
    with tab_single:
        user_text = st.text_area(
            "Enter text to analyse",
            placeholder="Type or paste any English sentence here — e.g. 'I feel so happy and excited today!'",
            height=130,
            key="single_input",
        )

        col_btn, col_example = st.columns([1, 2])
        with col_btn:
            predict_clicked = st.button("🔮 Predict Emotion", type="primary", use_container_width=True)
        with col_example:
            example = st.selectbox(
                "Or try an example",
                [
                    "",
                    "I am so happy and grateful for everything!",
                    "I feel completely devastated and broken inside.",
                    "This makes me absolutely furious and outraged!",
                    "I was shocked when I heard the unexpected news.",
                    "I'm terrified of what might happen tomorrow.",
                    "I love you more than words can ever express.",
                ],
                label_visibility="collapsed",
            )

        input_text = user_text.strip() if predict_clicked else (example if example else "")

        if input_text:
            with st.spinner("Analysing…"):
                result = predict_emotion(input_text, vectorizer, model, inverse_map)

            st.session_state["history"].append(result)

            render_result_card(result)

            st.markdown("")  # spacing

            render_probability_charts(result.probabilities)
            render_word_importance(result)
            render_preprocessing_details(result)

    # ------------------------------------------------------------------ batch
    with tab_batch:
        st.markdown("Predict emotions for **multiple texts at once**. Enter one sentence per line or upload a `.txt` / `.csv` file.")

        batch_method = st.radio("Input method", ["Paste text (one per line)", "Upload file"], horizontal=True, label_visibility="collapsed")

        lines: List[str] = []

        if batch_method == "Paste text (one per line)":
            batch_text = st.text_area(
                "Batch input",
                placeholder="Line 1: I feel great!\nLine 2: This is terrible.\nLine 3: …",
                height=160,
                key="batch_input",
            )
            if batch_text.strip():
                lines = [l.strip() for l in batch_text.strip().splitlines() if l.strip()]
        else:
            uploaded = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
            if uploaded is not None:
                content = uploaded.read().decode("utf-8", errors="replace")
                if uploaded.name.endswith(".csv"):
                    df_up = pd.read_csv(io.StringIO(content))
                    text_col = df_up.columns[0]
                    lines = df_up[text_col].dropna().astype(str).tolist()
                else:
                    lines = [l.strip() for l in content.splitlines() if l.strip()]

        if lines:
            st.info(f"📝 **{len(lines)}** text(s) ready for analysis")

            if st.button("🚀 Analyse All", type="primary", use_container_width=True):
                results: List[PredictionResult] = []
                progress = st.progress(0, text="Processing…")

                for i, line in enumerate(lines):
                    r = predict_emotion(line, vectorizer, model, inverse_map)
                    results.append(r)
                    progress.progress((i + 1) / len(lines), text=f"Processing {i+1}/{len(lines)}")

                progress.empty()

                # Build results dataframe
                rows = []
                for r in results:
                    row = {
                        "Text": r.text,
                        "Predicted Emotion": r.emotion.capitalize(),
                        "Confidence": round(r.confidence, 4),
                    }
                    for emo, prob in r.probabilities.items():
                        row[f"P({emo})"] = round(prob, 4)
                    rows.append(row)

                df_results = pd.DataFrame(rows)

                st.markdown("#### 📊 Results")
                st.dataframe(df_results, use_container_width=True, hide_index=True)

                # Summary metrics
                st.markdown("#### 📈 Distribution Summary")

                dist = df_results["Predicted Emotion"].value_counts()
                fig_dist = px.pie(
                    names=dist.index,
                    values=dist.values,
                    color=dist.index,
                    color_discrete_map={
                        e.capitalize(): EMOTION_META.get(e, {}).get("colour", "#64748b")
                        for e in EMOTION_META
                    },
                    hole=0.4,
                )
                fig_dist.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                avg_conf = df_results["Confidence"].mean()
                st.markdown(
                    f"""
                    <div class="metric-row">
                        <div class="metric-card">
                            <div class="metric-value">{len(results)}</div>
                            <div class="metric-label">Texts Analysed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{avg_conf:.1%}</div>
                            <div class="metric-label">Avg Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{dist.index[0]}</div>
                            <div class="metric-label">Most Common</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Download results
                csv_out = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out,
                    "batch_results.csv",
                    "text/csv",
                    use_container_width=True,
                )

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#475569;font-size:.8rem">'
        "Built with Streamlit • Logistic Regression + TF-IDF • Trained on 16,000 samples"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
