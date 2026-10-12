"""
train_model.py - Fake News Detection: load data, clean, explore, train, evaluate, save.
Run:  python train_model.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import clean_text

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
BASE = Path(__file__).parent
FAKE_PATH = BASE / "dataset" / "Fake.csv"
TRUE_PATH = BASE / "dataset" / "True.csv"
MODELS = BASE / "models"
REPORTS = BASE / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


# ---------- 1. Load and inspect ----------
def load_data():
    for p in (FAKE_PATH, TRUE_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}\n"
                                    "Put Fake.csv and True.csv inside the 'dataset' folder.")
    fake = pd.read_csv(FAKE_PATH)
    true = pd.read_csv(TRUE_PATH)
    print("Fake.csv columns:", list(fake.columns), "| rows:", len(fake))
    print("True.csv columns:", list(true.columns), "| rows:", len(true))
    for name, df in (("Fake.csv", fake), ("True.csv", true)):
        if "text" not in df.columns:
            raise ValueError(f"{name} has no 'text' column. Found: {list(df.columns)}")
    fake["label"] = 0  # 0 = Fake
    true["label"] = 1  # 1 = Real
    return pd.concat([fake, true], ignore_index=True)


# ---------- 2. Clean data ----------
def prepare(df):
    print("\nMissing values per column:\n", df.isna().sum())
    for col in ("title", "text"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    before = len(df)
    df = df[df["content"].str.len() > 0]
    df = df.drop_duplicates(subset=["content"])
    print(f"Rows before: {before} | after removing empty/duplicates: {len(df)}")
    # An article that appears in BOTH classes would confuse the model; drop it.
    df = df.drop_duplicates(subset=["content"], keep=False) if df["content"].duplicated().any() else df
    df["clean_text"] = df["content"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df.reset_index(drop=True)


# ---------- 3. Exploratory data analysis ----------
def run_eda(df):
    names = df["label"].map({0: "Fake", 1: "Real"})

    plt.figure(figsize=(5, 4))
    sns.countplot(x=names, hue=names, palette="Set2", legend=False)
    plt.title("Class distribution"); plt.xlabel("Class"); plt.ylabel("Articles")
    plt.tight_layout(); plt.savefig(REPORTS / "01_class_distribution.png", dpi=120); plt.close()

    if "subject" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x="subject", hue=names, palette="Set2")
        plt.title("Subject by class"); plt.xticks(rotation=30); plt.tight_layout()
        plt.savefig(REPORTS / "02_subject_distribution.png", dpi=120); plt.close()

    df = df.assign(word_count=df["clean_text"].str.split().str.len())
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df[df["word_count"] < 2000], x="word_count", hue=names, bins=50,
                 palette="Set2", element="step")
    plt.title("Article length (words)"); plt.tight_layout()
    plt.savefig(REPORTS / "03_text_length.png", dpi=120); plt.close()
    print("EDA charts saved in 'reports/'")


# ---------- 4. Train and evaluate ----------
def make_pipeline(model):
    # TF-IDF is inside the pipeline so it is fitted ONLY on training data (no leakage).
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7, min_df=5,
                                  ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf", model),
    ])


def evaluate(pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    return pred, {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


def plot_confusion(y_test, pred, model_name):
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion matrix - {model_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(REPORTS / "04_confusion_matrix.png", dpi=120); plt.close()


def plot_top_words(pipe, n=15):
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return
    words = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = clf.coef_[0]
    top_real = np.argsort(coef)[-n:]
    top_fake = np.argsort(coef)[:n]
    idx = np.concatenate([top_fake, top_real])
    plt.figure(figsize=(8, 7))
    plt.barh(words[idx], coef[idx], color=["#e07b7b"] * n + ["#6aa84f"] * n)
    plt.title("Most influential words (red = Fake, green = Real)")
    plt.tight_layout(); plt.savefig(REPORTS / "06_top_words.png", dpi=120); plt.close()


def main():
    df = prepare(load_data())
    print("\nClass counts:\n", df["label"].value_counts().rename({0: "Fake", 1: "Real"}))
    run_eda(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2,
        stratify=df["label"], random_state=RANDOM_STATE)
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=RANDOM_STATE),
    }
    results, fitted, preds = {}, {}, {}
    for name, model in candidates.items():
        print(f"\nTraining {name} ...")
        pipe = make_pipeline(model).fit(X_train, y_train)
        preds[name], results[name] = evaluate(pipe, X_test, y_test)
        fitted[name] = pipe
        print({k: round(v, 4) for k, v in results[name].items()})

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest model (by F1): {best}")
    print(classification_report(y_test, preds[best], target_names=["Fake", "Real"]))

    plot_confusion(y_test, preds[best], best)
    comp = pd.DataFrame(results).T
    comp.plot(kind="bar", figsize=(8, 4), ylim=(comp.values.min() - 0.02, 1.005))
    plt.title("Model comparison"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(REPORTS / "05_model_comparison.png", dpi=120); plt.close()
    plot_top_words(fitted[best])

    # Save: whole pipeline + separate vectorizer/classifier for clarity
    best_pipe = fitted[best]
    joblib.dump(best_pipe, MODELS / "fake_news_pipeline.joblib")
    joblib.dump(best_pipe.named_steps["tfidf"], MODELS / "tfidf_vectorizer.joblib")
    joblib.dump(best_pipe.named_steps["clf"], MODELS / "classifier.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(
        {"best_model": best, "results": results}, indent=2))
    print("\nSaved model files in 'models/' and charts in 'reports/'.")
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    main()
