import argparse
import json
import logging
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

REAL_LABEL = "REAL"
FAKE_LABEL = "FAKE"
DEFAULT_TRUE_CSV = Path("data/True.csv")
DEFAULT_FAKE_CSV = Path("data/Fake.csv")
LOG = logging.getLogger("fake_news_cli")


def clean_text(text: str) -> str:
    """Normalize article text and remove obvious source-marking bias tokens."""
    text = str(text).lower()
    text = re.sub(r"^.*?\(reuters\)\s*-\s*", "", text)
    text = text.replace("reuters", "")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"\W", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_prepare_data(true_csv: Path, fake_csv: Path) -> pd.DataFrame:
    """Load datasets, label them, and create a clean_text column."""
    if not true_csv.exists():
        raise FileNotFoundError(f"Missing file: {true_csv}")
    if not fake_csv.exists():
        raise FileNotFoundError(f"Missing file: {fake_csv}")

    true_df = pd.read_csv(true_csv)
    fake_df = pd.read_csv(fake_csv)

    if "text" not in true_df.columns or "text" not in fake_df.columns:
        raise ValueError("Both CSV files must include a 'text' column.")

    true_df["label"] = REAL_LABEL
    fake_df["label"] = FAKE_LABEL

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df["clean_text"] = df["text"].map(clean_text)
    return df


def train_pipeline(df: pd.DataFrame, test_size: float, random_state: int):
    """Train TF-IDF + logistic regression and return artifacts and metrics."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vec, y_train)

    predictions = model.predict(x_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=[FAKE_LABEL, REAL_LABEL])

    return model, vectorizer, accuracy, cm


def save_artifacts(model, vectorizer, model_path: Path, vectorizer_path: Path) -> None:
    """Persist trained model and vectorizer for later CLI predictions."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def load_artifacts(model_path: Path, vectorizer_path: Path):
    """Load previously persisted model and vectorizer artifacts."""
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Model/vectorizer files are missing. Run 'train' first."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_single_text(model, vectorizer, text: str) -> dict:
    """Predict label and confidence scores for a single input text."""
    cleaned = clean_text(text)
    x_vec = vectorizer.transform([cleaned])
    prediction = model.predict(x_vec)[0]
    probabilities = model.predict_proba(x_vec)[0]
    classes = list(model.classes_)

    fake_prob = probabilities[classes.index(FAKE_LABEL)] * 100
    real_prob = probabilities[classes.index(REAL_LABEL)] * 100

    return {
        "prediction": prediction,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
    }


def plot_confusion_matrix(cm, output_path: Path) -> None:
    """Save confusion matrix chart as an image file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[FAKE_LABEL, REAL_LABEL],
        yticklabels=[FAKE_LABEL, REAL_LABEL],
    )
    plt.title("Fake News Classifier - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def interactive_mode(model, vectorizer) -> None:
    """Start interactive terminal loop for repeated fake/real checks."""
    print("\n--- INTERACTIVE MODE ---")
    print("Enter a headline or article text.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("Paste News Here: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Shutting down. Goodbye!")
            break
        if not user_input:
            continue

        result = predict_single_text(model, vectorizer, user_input)
        print("\n" + "-" * 34)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence REAL: {result['real_prob']:.1f}%")
        print(f"Confidence FAKE: {result['fake_prob']:.1f}%")
        print("-" * 34 + "\n")


def _print_prediction(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence REAL: {result['real_prob']:.1f}%")
    print(f"Confidence FAKE: {result['fake_prob']:.1f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fake News classifier CLI (train and predict)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model artifacts")
    train_parser.add_argument("--true-csv", type=Path, default=DEFAULT_TRUE_CSV)
    train_parser.add_argument("--fake-csv", type=Path, default=DEFAULT_FAKE_CSV)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--model-path", type=Path, default=Path("model.joblib"))
    train_parser.add_argument(
        "--vectorizer-path", type=Path, default=Path("vectorizer.joblib")
    )
    train_parser.add_argument(
        "--cm-output",
        type=Path,
        default=Path("confusion_matrix.png"),
        help="Where to save confusion matrix image",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict one text item")
    predict_parser.add_argument("--text", required=True, help="Headline/article text")
    predict_parser.add_argument("--model-path", type=Path, default=Path("model.joblib"))
    predict_parser.add_argument(
        "--vectorizer-path", type=Path, default=Path("vectorizer.joblib")
    )
    predict_parser.add_argument(
        "--json",
        action="store_true",
        help="Print prediction in JSON format",
    )

    interactive_parser = subparsers.add_parser(
        "interactive", help="Interactive prediction loop"
    )
    interactive_parser.add_argument(
        "--model-path", type=Path, default=Path("model.joblib")
    )
    interactive_parser.add_argument(
        "--vectorizer-path", type=Path, default=Path("vectorizer.joblib")
    )

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            LOG.info("Loading and preparing data...")
            df = load_and_prepare_data(args.true_csv, args.fake_csv)
            LOG.info("Training model...")
            model, vectorizer, accuracy, cm = train_pipeline(
                df=df, test_size=args.test_size, random_state=args.random_state
            )
            save_artifacts(model, vectorizer, args.model_path, args.vectorizer_path)
            plot_confusion_matrix(cm, args.cm_output)
            print(f"Training complete. Accuracy: {accuracy * 100:.2f}%")
            print(f"Saved model: {args.model_path}")
            print(f"Saved vectorizer: {args.vectorizer_path}")
            print(f"Saved confusion matrix: {args.cm_output}")
            return

        if args.command == "predict":
            model, vectorizer = load_artifacts(args.model_path, args.vectorizer_path)
            result = predict_single_text(model, vectorizer, args.text)
            _print_prediction(result, as_json=args.json)
            return

        if args.command == "interactive":
            model, vectorizer = load_artifacts(args.model_path, args.vectorizer_path)
            interactive_mode(model, vectorizer)
            return
    except Exception as exc:
        LOG.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()