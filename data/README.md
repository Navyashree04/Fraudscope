# Data

**Dataset is public/synthetic — no real bank or customer data is used anywhere in this project.**

## Preferred: Kaggle "Credit Card Fraud Detection" dataset

FraudScope is designed around the public Kaggle **Credit Card Fraud Detection**
dataset (284,807 transactions made by European cardholders in September 2013,
492 of which are fraudulent — a ~0.17% positive class rate). Features `V1`–`V28`
are the output of a PCA transformation applied by the dataset's original
publishers specifically so that no raw cardholder or merchant data is exposed;
only `Time` and `Amount` are not PCA-transformed.

To use it:

1. Download `creditcard.csv` from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place it at `data/creditcard.csv` (this path is gitignored — the raw file
   is never committed).
3. Run `python -m src.train`.

## Fallback: synthetic stand-in dataset

This build environment has no network access to Kaggle, so `src/train.py`
will **automatically generate a synthetic stand-in dataset** with the exact
same schema (`Time`, `V1`-`V28`, `Amount`, `Class`) and a similarly severe
class imbalance (~0.17% fraud rate) if `data/creditcard.csv` is not present.

This fallback is clearly labeled `synthetic-for-demo` everywhere it appears
(logs, `metadata.json`, the evaluation report, and the README) so nobody
mistakes it for the real Kaggle data or, worse, real bank data. The synthetic
generator lives in `src/synthetic_data.py` — it samples PCA-like Gaussian
features for the legitimate class and shifts several `V` features for the
fraud class so a classifier has real (if easier-than-real-life) signal to
learn, rather than pure noise.

Swapping in the real Kaggle CSV later requires no code changes — just drop
the file in place and re-run training.
