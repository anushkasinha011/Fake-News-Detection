# 📰 Fake News Detection Using Machine Learning

A college mini-project that classifies a news article as **likely Fake** or **likely Real**
using TF-IDF text features and a scikit-learn classifier, with a Streamlit web interface.

> ⚠️ **Disclaimer:** Predictions come from writing patterns learned from the training dataset.
> The app does **not** verify facts or check whether a story is actually true.

## Features
- Loads and combines `Fake.csv` (label 0) and `True.csv` (label 1)
- Handles missing values and duplicates; cleans text (and removes the "(Reuters)" tag that would leak the answer)
- EDA charts, stratified 80/20 split, TF-IDF inside a Pipeline (no data leakage)
- Compares Logistic Regression vs Passive Aggressive Classifier; keeps the best by F1
- Accuracy, precision, recall, F1, classification report, confusion matrix
- Saves the model with Joblib; Streamlit app with input validation and error handling

## Project structure
```
Fake-News-Detection/
├── dataset/            # put Fake.csv and True.csv here
├── models/             # trained model files (created by train_model.py)
├── reports/            # charts (created by train_model.py)
├── utils.py            # shared text-cleaning function
├── check_setup.py      # verifies Python, libraries, dataset
├── train_model.py      # trains, evaluates, saves the model
├── app.py              # Streamlit web app
├── requirements.txt
├── VIVA_AND_REPORT.md  # viva Q&A and report outline
├── README.md
└── .gitignore
```

## Dataset
Download the **Fake and Real News Dataset** (Kaggle, by ISOT / Clément Bisaillon) and place
`Fake.csv` and `True.csv` inside the `dataset/` folder.

## Installation (Windows, VS Code terminal)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

(macOS/Linux: `source venv/bin/activate`)

## Run
```
python check_setup.py        # optional: confirm everything is in place
python train_model.py        # trains the model (takes ~1-3 minutes)
streamlit run app.py         # opens the web app in your browser
```

## Output
- `models/fake_news_pipeline.joblib` – TF-IDF + classifier pipeline used by the app
- `models/tfidf_vectorizer.joblib`, `models/classifier.joblib` – the two parts saved separately
- `models/metrics.json` – evaluation scores
- `reports/*.png` – class distribution, subject, length, confusion matrix, model comparison, top words

## Limitations and bias
- The dataset is mostly US political news from about 2016–2017; the model may do poorly on other topics, regions, or years.
- It learns **style** (e.g. word choice, formatting), not truth. Well-written fake news or sloppy real news can be misclassified.
- Very high accuracy (often 98–99%+) partly reflects how different the two sources are. It does **not** mean the model is reliable on real-world news.
- Source-specific patterns can leak into the model; this project removes the obvious "Reuters" tag but other hints may remain.

## Results on the provided dataset
After removing duplicates, 39,098 articles remained (21,196 real, 17,902 fake). On a held-out 20% test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 98.31% | 97.81% | 99.10% | 98.45% |
| Passive Aggressive (selected) | 99.14% | 98.97% | 99.46% | 99.21% |

The model's most influential words include "getty images", "featured image" and "video" (fake) and weekday names like "tuesday" or "said thursday" (real). These are writing-style and source habits, not truth signals, which is why high accuracy here should not be read as real-world reliability.

## Future improvements
Test on a different dataset, try n-gram/word-embedding or transformer models, add explainability (LIME/SHAP), add fact-checking sources.
