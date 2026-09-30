"""Predict whether two English passages have the same author.

Run without text arguments for an interactive terminal prompt. Model artifacts
must first be downloaded into the matching directory under ``models/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIRS = {
    "stylometric": ("stylometric-ensemble", "Models_ML"),
    "roberta": ("roberta-asymmetric-loss", "Models_ASL"),
}
REQUIRED_FILES = {
    "stylometric": (
        "ensemble_config.json",
        "xgb_model.json",
        "lgb_model.pkl",
        "tfidf_word.pkl",
        "tfidf_char.pkl",
        "tfidf_fw.pkl",
    ),
    "roberta": (
        "best_threshold.json",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
}


def normalize_stylometric_text(text: str) -> str:
    """Apply the same text normalization used by the training notebook."""
    text = re.sub(r"[\w.+-]+@[\w.]+\.[a-z]{2,}", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", " ", text)
    text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", " ", text)
    text = re.sub(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_stylometric(text_1: str, text_2: str, model_dir: Path) -> tuple[float, float]:
    # A single pair does not use joblib workers; avoid its Windows core-count probe.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    import joblib
    import numpy as np
    import scipy.sparse as sp
    import xgboost as xgb
    from sklearn.metrics.pairwise import paired_cosine_distances

    def style_features(text: str):
        words = text.split()
        count = len(words)
        if count < 5:
            return np.zeros(24, dtype="float32")

        sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
        word_lengths = [len(word) for word in words]
        sentence_lengths = [len(part.split()) for part in sentences] if sentences else [count]
        return np.array(
            [
                count,
                np.mean(word_lengths),
                np.mean(sentence_lengths),
                len(set(words)) / count,
                np.std(word_lengths),
                np.std(sentence_lengths),
                np.max(word_lengths),
                np.median(sentence_lengths),
                text.count(",") / count,
                text.count(".") / count,
                text.count("!") / count,
                text.count("?") / count,
                text.count(";") / count,
                text.count(":") / count,
                text.count('"') / count,
                text.count("'") / count,
                sum(word.isupper() for word in words) / count,
                sum(len(word) > 6 for word in words) / count,
                sum(len(word) <= 3 for word in words) / count,
                sum(word[0].isupper() for word in words if word) / count,
                sum(word.endswith("ly") for word in words) / count,
                sum(word.endswith(("tion", "sion")) for word in words) / count,
                sum(word.endswith("ing") for word in words) / count,
                sum(word.endswith("ed") for word in words) / count,
            ],
            dtype="float32",
        )

    text_1 = normalize_stylometric_text(text_1)
    text_2 = normalize_stylometric_text(text_2)
    style_1 = style_features(text_1)
    style_2 = style_features(text_2)
    style_pair = sp.csr_matrix(
        np.hstack(
            [
                np.abs(style_1 - style_2),
                np.minimum(style_1, style_2) / (np.maximum(style_1, style_2) + 1e-9),
                (style_1 + style_2) / 2,
            ]
        ).reshape(1, -1).astype("float32")
    )

    word_vectorizer = joblib.load(model_dir / "tfidf_word.pkl")
    char_vectorizer = joblib.load(model_dir / "tfidf_char.pkl")
    function_word_vectorizer = joblib.load(model_dir / "tfidf_fw.pkl")

    word_1, word_2 = (word_vectorizer.transform([text]) for text in (text_1, text_2))
    char_1, char_2 = (char_vectorizer.transform([text]) for text in (text_1, text_2))
    function_1, function_2 = (
        function_word_vectorizer.transform([text]) for text in (text_1, text_2)
    )
    combined_1 = sp.hstack([word_1, char_1])
    combined_2 = sp.hstack([word_2, char_2])
    cosine = paired_cosine_distances(combined_1, combined_2).reshape(-1, 1).astype("float32")
    function_cosine = (
        paired_cosine_distances(function_1, function_2).reshape(-1, 1).astype("float32")
    )
    features = sp.hstack(
        [
            abs(combined_1 - combined_2),
            combined_1.multiply(combined_2),
            abs(function_1 - function_2),
            function_1.multiply(function_2),
            style_pair,
            cosine,
            function_cosine,
        ],
        format="csr",
        dtype="float32",
    )

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_dir / "xgb_model.json"))
    lgb_model = joblib.load(model_dir / "lgb_model.pkl")
    with (model_dir / "ensemble_config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        probability = (
            float(xgb_model.predict_proba(features)[0, 1]) * config["xgb_weight"]
            + float(lgb_model.predict_proba(features)[0, 1]) * config["lgb_weight"]
        )
    return probability, float(config["threshold"])


def clean_roberta_text(text: str) -> str:
    """Apply the same text normalization used by the RoBERTa notebook."""
    text = re.sub(r"(From|To|Cc|Subject|Date|Forwarded by)[^\n]*\n", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_roberta(text_1: str, text_2: str, model_dir: Path) -> tuple[float, float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running RoBERTa on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    model.to(device)
    model.eval()

    inputs = tokenizer(
        clean_roberta_text(text_1),
        clean_roberta_text(text_2),
        max_length=512,
        truncation="longest_first",
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        probability = torch.softmax(model(**inputs).logits, dim=-1)[0, 1].item()

    with (model_dir / "best_threshold.json").open(encoding="utf-8") as threshold_file:
        threshold = json.load(threshold_file)["best_threshold"]
    return probability, float(threshold)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=MODEL_DIRS,
        default="stylometric",
        help="model to run (default: stylometric, CPU friendly)",
    )
    parser.add_argument("--text-1", help="first English passage; prompts if omitted")
    parser.add_argument("--text-2", help="second English passage; prompts if omitted")
    parser.add_argument("--model-dir", type=Path, help="override the model artifact directory")
    args = parser.parse_args(argv)

    if (args.text_1 is None) != (args.text_2 is None):
        parser.error("provide both --text-1 and --text-2, or neither")

    model_candidates = [PROJECT_ROOT / "models" / name for name in MODEL_DIRS[args.model]]
    model_dir = args.model_dir or next(
        (
            path
            for path in model_candidates
            if all((path / name).is_file() for name in REQUIRED_FILES[args.model])
        ),
        model_candidates[0],
    )
    missing = [name for name in REQUIRED_FILES[args.model] if not (model_dir / name).is_file()]
    if missing:
        print(
            f"Model files missing from {model_dir}: {', '.join(missing)}\n"
            "Download the model files described in docs/reproduction.md first.",
            file=sys.stderr,
        )
        return 1

    try:
        text_1 = args.text_1 if args.text_1 is not None else input("First English passage: ")
        text_2 = args.text_2 if args.text_2 is not None else input("Second English passage: ")
    except EOFError:
        print("No text received. Provide both --text-1 and --text-2.", file=sys.stderr)
        return 1
    if not text_1.strip() or not text_2.strip():
        print("Both passages must contain text.", file=sys.stderr)
        return 1

    try:
        if args.model == "stylometric":
            probability, threshold = predict_stylometric(text_1, text_2, model_dir)
        else:
            probability, threshold = predict_roberta(text_1, text_2, model_dir)
    except ModuleNotFoundError as error:
        print(
            f"Missing Python dependency: {error.name}. Run: pip install -r requirements-inference.txt",
            file=sys.stderr,
        )
        return 1

    print(f"Model: {args.model}")
    print(f"Prediction: {'Same author' if probability >= threshold else 'Different authors'}")
    print(f"Same-author probability: {probability:.2%}")
    print(f"Decision threshold: {threshold:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
