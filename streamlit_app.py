from pathlib import Path

import streamlit as st

from fake_news_project import (
    DEFAULT_FAKE_CSV,
    DEFAULT_TRUE_CSV,
    FAKE_LABEL,
    REAL_LABEL,
    load_and_prepare_data,
    load_artifacts,
    plot_confusion_matrix,
    predict_single_text,
    save_artifacts,
    train_pipeline,
)

DEFAULT_MODEL_PATH = Path("model.joblib")
DEFAULT_VECTORIZER_PATH = Path("vectorizer.joblib")
DEFAULT_CM_PATH = Path("confusion_matrix.png")


@st.cache_resource
def get_loaded_artifacts(model_path: str, vectorizer_path: str):
    return load_artifacts(Path(model_path), Path(vectorizer_path))


def train_from_ui(
    true_csv: str,
    fake_csv: str,
    test_size: float,
    random_state: int,
    model_path: str,
    vectorizer_path: str,
    cm_path: str,
):
    df = load_and_prepare_data(Path(true_csv), Path(fake_csv))
    model, vectorizer, accuracy, cm = train_pipeline(df, test_size, random_state)
    save_artifacts(model, vectorizer, Path(model_path), Path(vectorizer_path))
    plot_confusion_matrix(cm, Path(cm_path))
    get_loaded_artifacts.clear()
    return accuracy


def main() -> None:
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
    st.title("📰 Fake News Detector")
    st.caption("Train and run predictions with TF-IDF + Logistic Regression.")

    with st.sidebar:
        st.header("Configuration")
        true_csv = st.text_input("True CSV path", value=str(DEFAULT_TRUE_CSV))
        fake_csv = st.text_input("Fake CSV path", value=str(DEFAULT_FAKE_CSV))
        model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
        vectorizer_path = st.text_input("Vectorizer path", value=str(DEFAULT_VECTORIZER_PATH))
        cm_path = st.text_input("Confusion matrix image", value=str(DEFAULT_CM_PATH))
        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

        if st.button("Train / Retrain Model", type="primary", use_container_width=True):
            with st.spinner("Training model... this can take a while on large CSV files."):
                try:
                    accuracy = train_from_ui(
                        true_csv=true_csv,
                        fake_csv=fake_csv,
                        test_size=test_size,
                        random_state=int(random_state),
                        model_path=model_path,
                        vectorizer_path=vectorizer_path,
                        cm_path=cm_path,
                    )
                    st.success(f"Training complete. Accuracy: {accuracy * 100:.2f}%")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    st.subheader("Predict News Text")
    text = st.text_area(
        "Enter a headline or article text",
        height=180,
        placeholder="Paste a news headline or article content...",
    )

    if st.button("Predict", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text before predicting.")
            return
        try:
            model, vectorizer = get_loaded_artifacts(model_path, vectorizer_path)
            result = predict_single_text(model, vectorizer, text)
            label = result["prediction"]
            real_prob = result["real_prob"]
            fake_prob = result["fake_prob"]

            if label == REAL_LABEL:
                st.success(f"Prediction: {REAL_LABEL}")
            elif label == FAKE_LABEL:
                st.error(f"Prediction: {FAKE_LABEL}")
            else:
                st.info(f"Prediction: {label}")

            c1, c2 = st.columns(2)
            c1.metric("Confidence REAL", f"{real_prob:.1f}%")
            c2.metric("Confidence FAKE", f"{fake_prob:.1f}%")
            st.progress(min(int(max(real_prob, fake_prob)), 100))
        except FileNotFoundError:
            st.warning("Model artifacts not found. Train the model first from the sidebar.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    cm_file = Path(cm_path)
    if cm_file.exists():
        st.subheader("Confusion Matrix")
        st.image(str(cm_file), caption="Latest training run")


if __name__ == "__main__":
    main()
