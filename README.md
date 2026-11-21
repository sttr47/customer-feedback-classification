# Customer Feedback Text Classification

Hybrid ensemble of a fine-tuned BERT model and a classical scikit-learn pipeline for 3-class customer feedback classification. The blend of both models outperforms each of them on its own.

## Task
Classify customer feedback (a `tag` field and free-text `feedback`) into one of three classes:

- `customer_support`
- `technical_support`
- `other`

## Approach

**Input text.** The `tag` is prepended to the feedback as `"{tag}, {feedback}"` when present. Missing values are replaced with empty strings.

**Classical model.**
- One-hot encoding of `tag`
- TF-IDF on `feedback` (1-2 grams, 5000 features)
- Logistic Regression (C=2.0, balanced class weights)

**Transformer.**
- `bert-base-uncased`, max sequence length 64
- Only the classification head and the last encoder layer are fine-tuned
- 3 epochs, batch size 32, warmup 100 steps, weight decay 0.01
- Best checkpoint selected by validation macro F1

**Ensemble.** Weighted average of the class probabilities of both models. The weight was selected on the validation set with a sweep from 0.0 to 1.0 in steps of 0.1.

## Results

Validation setup: stratified 80/20 split, metric is macro F1.

| Model | Validation macro F1 |
|-------|---------------------|
| Logistic Regression + TF-IDF (sklearn) | 0.8304 |
| BERT (partially fine-tuned) | 0.8203 |
| **Hybrid ensemble (0.4 BERT / 0.6 sklearn)** | **0.8650** |

Ensemble weight sweep (weight of the transformer):

| Transformer weight | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|--------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Macro F1 | 0.8304 | 0.8425 | 0.8526 | 0.8569 | **0.8650** | 0.8604 | 0.8507 | 0.8405 | 0.8323 | 0.8276 | 0.8203 |

The two models make different kinds of errors, so blending them gains about 3.5 points of macro F1 over the best single model.

## Data Leakage Fix
An earlier version of the pipeline contained a data leakage issue that inflated validation scores (around 90% F1). After fixing it, the score dropped to the ~86% range shown above, which is the reliable baseline for this project.

## Reproducibility
- Fixed random seed (42) for Python, NumPy and PyTorch
- Deterministic cuDNN settings
- Stratified train/validation split with a fixed `random_state`

## Limitations
- The same validation set is used to pick the best BERT checkpoint and the ensemble weight, so the reported F1 may be slightly optimistic.
- For the final test predictions, the sklearn pipeline is refit on the full training data, while BERT is trained on the 80% training split only.
- Only the last encoder layer of BERT is fine-tuned; unfreezing more layers was not explored.

## Tech Stack
Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, NumPy

## Author
Səttər Abdulxalıqov
