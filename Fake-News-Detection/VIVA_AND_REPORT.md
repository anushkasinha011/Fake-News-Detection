# Viva Preparation and Report Outline

## Project explanation in 30 seconds
"We built a binary text classifier. News articles from Fake.csv (label 0) and True.csv (label 1) are
cleaned, converted to numbers with TF-IDF, and used to train Logistic Regression and Passive
Aggressive classifiers. The best model by F1-score is saved with Joblib and used by a Streamlit app
that predicts whether a pasted article looks fake or real. It detects writing patterns, not facts."

## Common viva questions and answers
1. **What is the problem statement?** Automatically classify news as fake or real from its text.
2. **Is this supervised or unsupervised?** Supervised classification, because every article has a label.
3. **What is TF-IDF?** It scores each word by how often it appears in a document (TF) times how rare it is across all documents (IDF). Common words like "the" get low scores; distinctive words get high scores.
4. **Why not feed raw text to the model?** Models need numbers. TF-IDF turns text into numeric vectors.
5. **Why Logistic Regression?** It is fast, works well on high-dimensional sparse text, and is interpretable (each word has a weight).
6. **What is Passive Aggressive Classifier?** An online linear model: it stays passive when a prediction is correct and updates aggressively when wrong. Good for large text data.
7. **What is data leakage and how did you avoid it?** Information from the test set influencing training. We split first and fit TF-IDF only on training data via a Pipeline. We also removed the "(Reuters)" tag, which would otherwise let the model cheat.
8. **Why stratified split?** Keeps the fake/real proportion the same in train and test sets.
9. **Why random_state=42?** Makes results reproducible.
10. **Explain the confusion matrix.** TP, TN, FP, FN counts. Here a false positive is fake news predicted as real; a false negative is real news predicted as fake.
11. **Precision vs recall?** Precision = of predicted "Real", how many are truly real. Recall = of all truly real, how many we found. F1 is their harmonic mean.
12. **Why not trust high accuracy?** The dataset has strong source/style differences, so the model may learn shortcuts and fail on news from other sources or periods.
13. **What are the limitations?** Single dataset, mostly US politics, 2016–2017, no fact checking, English only, can be fooled.
14. **What is bias here?** Topic and source bias: the model may associate certain topics or writing styles with "fake" regardless of truth.
15. **Why Joblib?** Efficiently saves scikit-learn objects so the app does not retrain every time.
16. **What does the Streamlit app do?** Loads the saved pipeline, validates input, cleans text, predicts, and shows confidence.
17. **What is the confidence value?** Logistic Regression's predicted probability; for Passive Aggressive it is a rough score, not a true probability.
18. **How would you improve it?** More diverse data, transformer models (BERT), explainability tools, cross-validation, fact-checking integration.
19. **What preprocessing did you do?** Lowercasing, removing URLs/numbers/punctuation, removing source tag, stop-word removal in TF-IDF, dropping missing and duplicate rows.
20. **What are n-grams?** Sequences of n words. We use unigrams and bigrams (ngram_range=(1,2)).

## Project report outline
1. Title page, certificate, acknowledgement, abstract
2. Introduction (problem, motivation, objectives, scope)
3. Literature survey (existing fake-news detection work)
4. Dataset description (source, columns, size, class balance)
5. System design (flowchart, architecture, tools)
6. Methodology (cleaning, EDA, split, TF-IDF, models, evaluation)
7. Implementation (code modules, Streamlit app screenshots)
8. Results and discussion (metrics table, confusion matrix, charts, top words)
9. Limitations and ethical considerations
10. Conclusion and future work
11. References
12. Appendix (code listings, README)
