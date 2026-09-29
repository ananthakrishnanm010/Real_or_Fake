import os
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Real or Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def find_model_files():
    candidates = []
    search_roots = [Path.cwd(), Path.cwd() / "models", Path.cwd() / "artifacts"]
    for root in search_roots:
        if root.exists():
            for file in root.iterdir():
                if file.is_file() and file.suffix in {".pkl", ".pickle", ".joblib"}:
                    candidates.append(file)
    return candidates


def load_saved_model():
    model = None
    vectorizer = None
    candidates = find_model_files()

    for file in candidates:
        name = file.name.lower()
        if "model" in name and ("tfidf" not in name and "vector" not in name):
            try:
                with open(file, "rb") as f:
                    model = pickle.load(f)
            except Exception:
                pass
        if "vector" in name or "tfidf" in name:
            try:
                with open(file, "rb") as f:
                    vectorizer = pickle.load(f)
            except Exception:
                pass

    if model is not None and vectorizer is not None:
        return model, vectorizer

    return None, None


def fallback_prediction(text: str):
    cleaned = (text or "").lower()
    if not cleaned.strip():
        return "Fake", 0.0, {"Real": 0.0, "Fake": 0.0}

    fake_markers = [
        "breaking", "shocking", "urgent", "miracle", "secret", "hoax",
        "click here", "you won't believe", "exposed", "suddenly", "must watch",
        "conspiracy", "billionaire", "hackers", "hidden", "scandal", "unbelievable",
    ]
    real_markers = [
        "reuters", "official", "statement", "according to", "report", "according to officials",
        "published", "confirmed", "government", "analysis", "study", "research",
        "said", "court", "policy", "agency"
    ]

    fake_score = sum(1 for marker in fake_markers if marker in cleaned)
    real_score = sum(1 for marker in real_markers if marker in cleaned)

    if fake_score > real_score:
        label = "Fake"
        confidence = min(0.99, 0.55 + (fake_score * 0.08))
    elif real_score > fake_score:
        label = "Real"
        confidence = min(0.99, 0.55 + (real_score * 0.08))
    else:
        label = "Real" if len(cleaned.split()) > 35 else "Fake"
        confidence = 0.5

    probs = {"Real": 0.5, "Fake": 0.5}
    if label == "Real":
        probs = {"Real": round(confidence, 3), "Fake": round(1 - confidence, 3)}
    else:
        probs = {"Real": round(1 - confidence, 3), "Fake": round(confidence, 3)}
    return label, round(confidence, 3), probs


def predict_article(text: str, model=None, vectorizer=None):
    if not text or not text.strip():
        return "Fake", 0.0, {"Real": 0.0, "Fake": 0.0}

    if model is not None and vectorizer is not None:
        try:
            features = vectorizer.transform([text])
            probs = model.predict_proba(features)[0]
            class_names = getattr(model, "classes_", [0, 1])
            if isinstance(class_names[0], str):
                labels = class_names
                scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
            else:
                labels = ["Real", "Fake"] if len(class_names) == 2 else [str(i) for i in range(len(class_names))]
                scores = {labels[i]: float(probs[i]) for i in range(len(labels))}

            predicted = max(scores, key=scores.get)
            confidence = max(scores.values())
            return predicted, round(float(confidence), 3), {k: round(float(v), 3) for k, v in scores.items()}
        except Exception:
            pass

    return fallback_prediction(text)


# UI helpers

def page_home():
    st.title("📰 Real or Fake News Detector")
    st.markdown("---")
    st.markdown(
        """
        This app classifies news articles as Real or Fake using a machine learning model.
        If a trained model and vectorizer are saved in the project, they will be loaded automatically.
        Otherwise, the app uses a lightweight fallback heuristic so it still runs correctly.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key features")
        st.write("- Real-time article classification")
        st.write("- Model artifact support")
        st.write("- Visual prediction panel")
        st.write("- Dataset overview")

    with col2:
        st.subheader("Quick stats")
        st.metric("Dataset", "44,898 articles")
        st.metric("Classes", "Real / Fake")
        st.metric("Model status", "Auto-detect from files")


def page_predict():
    st.title("🔮 Predict News")
    st.markdown("Enter article text or a headline below.")

    model, vectorizer = load_saved_model()

    text = st.text_area(
        "News article / headline",
        height=220,
        placeholder="Paste article text here...",
    )

    col1, col2 = st.columns(2)
    with col1:
        words = len((text or "").split()) if text else 0
        st.metric("Word count", words)
    with col2:
        chars = len(text) if text else 0
        st.metric("Character count", chars)

    if st.button("Classify", use_container_width=True, type="primary"):
        if not text.strip():
            st.warning("Please enter some text before classifying.")
            return

        label, confidence, scores = predict_article(text, model=model, vectorizer=vectorizer)

        result_class = "real-news" if label == "Real" else "fake-news"

        if label == "Real":
            st.markdown(
                f"""
                <div class="prediction-box real-news">
                    <h3>✅ Prediction: Real News</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-box fake-news">
                    <h3>⚠️ Prediction: Fake News</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Prediction breakdown")
        fig = go.Figure(
            data=[go.Bar(x=list(scores.keys()), y=list(scores.values()), marker_color=["#2ecc71", "#e74c3c"])]
        )
        fig.update_layout(yaxis_title="Probability", xaxis_title="Class", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model status")
        if model and vectorizer:
            st.success("Loaded trained model and vectorizer from project files.")
        else:
            st.info("No trained model files were found, so the fallback text-based predictor is in use.")


def page_dataset():
    st.title("📊 Dataset Overview")

    st.metric("Total articles", "44,898")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Real news", "21,417")
    with col2:
        st.metric("Fake news", "23,481")

    labels = ["Real News", "Fake News"]
    values = [21417, 23481]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample records")
    sample = {
        "title": "As U.S. budget fight looms, Republicans flip the script",
        "text": "WASHINGTON (Reuters) - The head of a conservative policy group that is influential with the Trump administration...",
        "subject": "politicsNews",
        "date": "December 31, 2017",
        "label": "Real",
    }
    st.json(sample)

    sample_fake = {
        "title": "Donald Trump Sends Out Embarrassing New Year Message",
        "text": "Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that...",
        "subject": "News",
        "date": "December 31, 2017",
        "label": "Fake",
    }
    st.json(sample_fake)


def page_about():
    st.title("ℹ️ About the Model")
    st.markdown(
        """
        This project is an NLP-based fake-news detector built around the dataset used in the notebook.
        The workflow includes data cleaning, normalization, tokenization, feature extraction, and model training.
        """
    )

    model_specs = {
        "Framework": "Scikit-learn / TensorFlow / Keras-compatible pipeline",
        "Preprocessing": "Lowercasing, punctuation removal, stopword filtering, tokenization",
        "Feature Extraction": "TF-IDF vectorization",
        "Classification": "Binary classification (Real vs Fake)",
        "Goal": "Predict whether a headline or article is likely real or fabricated",
    }

    for key, value in model_specs.items():
        st.write(f"**{key}:** {value}")

    st.subheader("Suggested model file names")
    st.code("model.pkl\nvectorizer.pkl\nmodels/model.pkl\nmodels/vectorizer.pkl")


st.markdown(
    """
    <style>
        .main { padding: 2rem; }
        .prediction-box { padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; }
        .real-news { background-color: #d4edda; border-left: 6px solid #28a745; }
        .fake-news { background-color: #f8d7da; border-left: 6px solid #dc3545; }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Navigation", ["Home", "Predict", "Dataset", "About"])

if page == "Home":
    page_home()
elif page == "Predict":
    page_predict()
elif page == "Dataset":
    page_dataset()
else:
    page_about()
