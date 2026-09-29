import os
import re
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        padding: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
    }
    .real-news {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .fake-news {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_file(names):
    for name in names:
        if os.path.exists(name):
            return name
    return None


@st.cache_resource
def load_model_and_vectorizer():
    model_path = find_file(['model.pkl', 'model.joblib', 'model.h5'])
    vectorizer_path = find_file(['vectorizer.pkl', 'vectorizer.joblib'])

    if not model_path:
        st.error("❌ Model file not found. Please save it as 'model.pkl' or 'model.h5' in the app folder.")
        st.stop()

    if not vectorizer_path:
        st.error("❌ TF-IDF vectorizer file not found. Please save it as 'vectorizer.pkl' in the app folder.")
        st.stop()

    try:
        if model_path.endswith('.pkl') or model_path.endswith('.pickle'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.endswith('.joblib'):
            import joblib
            model = joblib.load(model_path)
        elif model_path.endswith('.h5'):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        if vectorizer_path.endswith('.pkl') or vectorizer_path.endswith('.pickle'):
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
        elif vectorizer_path.endswith('.joblib'):
            import joblib
            vectorizer = joblib.load(vectorizer_path)
        else:
            raise ValueError(f"Unsupported vectorizer format: {vectorizer_path}")

        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Failed to load model/vectorizer: {e}")
        st.stop()


def clean_text(text):
    if pd.isna(text) or text == "":
        return ""

    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


def predict_news(title, text, model, vectorizer):
    combined = str(title or '') + ' ' + str(text or '')
    cleaned = clean_text(combined)

    if not cleaned:
        return None, None

    features = vectorizer.transform([cleaned])

    if hasattr(model, 'predict_proba'):
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        return pred, prob

    if hasattr(model, 'predict'):
        pred = model.predict(features)[0]
        return pred, None

    return None, None


model, vectorizer = load_model_and_vectorizer()

st.markdown("<h1 class='main-header'>📰 Fake News Detector 🔍</h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This app predicts whether a news article is Real or Fake.

    Enter an article title and text, and the model will classify it.
    """
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Batch Prediction", "ℹ️ Information"])

with tab1:
    st.subheader("Enter News Article Details")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("📌 News Title", placeholder="Enter article title...")
    with col2:
        subject = st.selectbox(
            "📂 Subject Category",
            ["News", "politicsNews", "worldnews", "US_News", "Middle-east", "Other"],
        )

    text = st.text_area("📝 News Content", placeholder="Paste article text here...", height=220)

    if st.button("🔍 Check Article", use_container_width=True):
        if title.strip() and text.strip():
            with st.spinner("Analyzing article..."):
                prediction, probability = predict_news(title, text, model, vectorizer)

                if prediction is not None:
                    is_real = int(prediction) == 1
                    label = "✅ REAL NEWS" if is_real else "❌ FAKE NEWS"
                    css = "real-news" if is_real else "fake-news"

                    st.markdown(
                        f"<div class='prediction-box {css}'><h2>{label}</h2></div>",
                        unsafe_allow_html=True,
                    )

                    if probability is not None:
                        real_prob = float(probability[1]) if len(probability) > 1 else 0.0
                        fake_prob = float(probability[0]) if len(probability) > 0 else 0.0

                        colA, colB, colC = st.columns(3)
                        with colA:
                            st.metric("Real News Probability", f"{real_prob:.2%}")
                        with colB:
                            st.metric("Fake News Probability", f"{fake_prob:.2%}")
                        with colC:
                            st.metric("Confidence", f"{max(real_prob, fake_prob):.2%}")

                    with st.expander("🔍 View Cleaned Text"):
                        st.text(clean_text(str(title) + ' ' + str(text)))
                else:
                    st.warning("⚠️ Unable to make prediction from the provided text.")
        else:
            st.warning("⚠️ Please provide both title and article content.")

with tab2:
    st.subheader("Batch Prediction from CSV")
    uploaded = st.file_uploader("Upload CSV with 'title' and 'text' columns", type='csv')

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)

            if 'title' not in df.columns or 'text' not in df.columns:
                st.error("❌ CSV file must contain 'title' and 'text' columns.")
            else:
                st.info(f"📁 Loaded {len(df)} rows.")

                if st.button("🚀 Predict All", use_container_width=True):
                    results = []
                    progress = st.progress(0)

                    for idx, row in df.iterrows():
                        pred, prob = predict_news(row['title'], row['text'], model, vectorizer)
                        if prob is not None:
                            results.append({
                                'title': row['title'],
                                'prediction': int(pred),
                                'real_prob': float(prob[1]) if len(prob) > 1 else 0.0,
                                'fake_prob': float(prob[0]) if len(prob) > 0 else 0.0,
                                'label': 'REAL' if int(pred) == 1 else 'FAKE',
                            })
                        else:
                            results.append({
                                'title': row['title'],
                                'prediction': -1,
                                'real_prob': 0.0,
                                'fake_prob': 0.0,
                                'label': 'UNKNOWN',
                            })

                        progress.progress((idx + 1) / len(df))

                    result_df = pd.DataFrame(results)
                    st.success("✅ Batch prediction complete!")
                    st.dataframe(result_df, use_container_width=True)

                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_data,
                        file_name="fake_news_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"❌ Error while reading CSV: {e}")

with tab3:
    st.subheader("Model Information")
    st.markdown(
        """
        ### Dataset
        - Fake vs Real news dataset
        - Text classification task
        - Includes title + article body

        ### Preprocessing
        - Lowercasing
        - URL removal
        - Punctuation removal
        - Stopword removal
        - Lemmatization
        - TF-IDF vectorization

        ### Output
        - 1 = Real News
        - 0 = Fake News
        """
    )
    st.markdown("---")
    st.markdown(
        """
        ### Usage
        Save your trained model as:
        - `model.pkl` or `model.joblib` or `model.h5`
        Save your vectorizer as:
        - `vectorizer.pkl` or `vectorizer.joblib`

        Then run:
        ```bash
        streamlit run app.py
        ```
        """
    )


# End of app.py
