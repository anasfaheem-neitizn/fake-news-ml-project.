"""Fake News Detector: training script.

Chalane ka tareeka:
    python train.py --data data

data/ folder mein Fake.csv aur True.csv honi chahiye (ISOT dataset).

Script do experiment chalati hai:
  1. source words ke saath (Reuters jaise naam rehne dete hain)
  2. source words hatakar (asli, imaandaar score)
Dono ka comparison results.csv mein jata hai. Model (2) wale setup se save hota hai.
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from utils import clean_text, strip_source

LABELS = {0: "Fake", 1: "Real"}


def load_data(data_dir: Path) -> pd.DataFrame:
    fake = pd.read_csv(data_dir / "Fake.csv")
    true = pd.read_csv(data_dir / "True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    before = len(df)
    df = df.drop_duplicates(subset="content")
    print(f"Rows: {before} -> {len(df)} (duplicates hataye)")
    print(df["label"].map(LABELS).value_counts().to_string(), "\n")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        # Passive Aggressive (PA-I) algorithm, SGDClassifier ke through
        "Passive Aggressive": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1",
                                            eta0=1.0, max_iter=50, random_state=42),
    }


def run_experiment(df: pd.DataFrame, remove_source: bool):
    setting = "source words hata ke" if remove_source else "source words ke saath"
    print(f"=== Experiment: {setting} ===")
    texts = df["content"].apply(lambda t: clean_text(strip_source(t) if remove_source else t))
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2)
    Xtr = tfidf.fit_transform(X_tr)
    Xte = tfidf.transform(X_te)

    rows, fitted, preds = [], {}, {}
    for name, model in build_models().items():
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte)
        p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary")
        acc = accuracy_score(y_te, pred)
        rows.append({"setting": setting, "model": name, "accuracy": acc,
                     "precision": p, "recall": r, "f1": f1})
        fitted[name], preds[name] = model, pred
        print(f"{name:22s} acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    print()
    return rows, fitted, tfidf, y_te, preds


def save_confusion_matrix(y_true, y_pred, title: str, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], [LABELS[0], LABELS[1]])
    ax.set_yticks([0, 1], [LABELS[0], LABELS[1]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_top_words(model, tfidf, path: Path, n: int = 20):
    words = np.array(tfidf.get_feature_names_out())
    order = np.argsort(model.coef_[0])
    pd.DataFrame({
        "top_fake_words": words[order[:n]],
        "top_real_words": words[order[::-1][:n]],
    }).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="Fake.csv aur True.csv wala folder")
    ap.add_argument("--out", default=".", help="model.pkl, tfidf.pkl yahan save honge")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data), Path(args.out)

    df = load_data(data_dir)
    all_rows = []

    rows_a, _, _, y_te_a, preds_a = run_experiment(df, remove_source=False)
    all_rows += rows_a
    save_confusion_matrix(y_te_a, preds_a["Logistic Regression"],
                          "LogReg (source words ke saath)", out_dir / "confusion_matrix_with_source.png")

    rows_b, fitted_b, tfidf_b, y_te_b, preds_b = run_experiment(df, remove_source=True)
    all_rows += rows_b

    results = pd.DataFrame(all_rows).round(4)
    results.to_csv(out_dir / "results.csv", index=False)
    print("Comparison table results.csv mein save hui.\n")

    # Sirf LR / NB save karte hain kyunki unse probability (confidence) milti hai.
    candidates = [r for r in rows_b if r["model"] in ("Logistic Regression", "Naive Bayes")]
    best_name = max(candidates, key=lambda r: r["f1"])["model"]
    joblib.dump(fitted_b[best_name], out_dir / "model.pkl")
    joblib.dump(tfidf_b, out_dir / "tfidf.pkl")
    save_confusion_matrix(y_te_b, preds_b[best_name], f"{best_name} (source words hata ke)",
                          out_dir / "confusion_matrix.png")
    save_top_words(fitted_b["Logistic Regression"], tfidf_b, out_dir / "top_words.csv")
    print(f"Saved: model.pkl ({best_name}), tfidf.pkl, confusion_matrix.png, top_words.csv")


if __name__ == "__main__":
    main()
