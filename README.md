# Predicting Customer Churn Risk from App Store Reviews

**Course:** 23CSE452 — Business Analytics
**Student:** Danvanth S C
**Register Number:** CB.SC.U4CSE23241
**Section:** SEC-C

## Overview

Online retailers lose revenue when customers quietly disengage — often long before that shows up in
purchase data. This case study treats app store reviews as an early-warning signal for customer churn,
mined directly from real user-generated text rather than a pre-packaged transactional churn dataset.

## Data Collection

- **Source:** Google Play Store reviews for e-commerce/marketplace apps (Amazon, Flipkart, Myntra,
  Nykaa, Snapdeal, Ajio, Meesho, ShopClues) — a publicly accessible source, scraped directly rather
  than downloaded as a ready-made dataset.
- **Method:** Reviews were scraped per app using a Play Store review scraper (e.g. the
  `google-play-scraper` package), pulling recent reviews across each app rather than relying on any
  single static export. Each raw review was then enriched with derived fields (sentiment score,
  churn-risk flag, complaint theme) — see `Case_Study_Report.pdf`, Section 2, for the full scraping
  and labeling methodology.
- **Volume:** 12,800 reviews collected across 8 apps — well above the 10,000-record target from the
  original proposal.
- **Fields:** review ID, app name, app category, star rating, rating bucket, review text, review
  length/word count, review date, day of week, thumbs-up count, app version, developer reply content,
  has-reply flag, sentiment score, sentiment label, churn-risk flag, explicit-churn-mention flag,
  churn signal, theme cluster, complaint theme.
- **Privacy:** `review_id` values are anonymized UUIDs; no reviewer names, emails, or device identifiers
  are retained.

## Data Preprocessing (summary — full detail in `analysis.ipynb`)

- De-duplicated by `review_id`.
- Parsed `review_date` into a proper datetime and derived month/day-of-week features.
- Missing `app_version` (~17% of rows) and `reply_content` (~34% of rows) are legitimately absent
  (older reviews predate version tagging; most reviews never receive a developer reply) — flagged with
  explicit indicator columns (`has_app_version`, `has_reply`) rather than dropped.
- Basic text cleaning of `review_text` (lowercased, URLs/punctuation stripped) for TF-IDF analysis.
- **Label leakage identified and corrected:** initial models trained on `rating` and `sentiment_score`
  scored an unrealistic ~1.00 ROC-AUC. Diagnosis showed both features near-deterministically define
  `churn_risk` (e.g. rating ≥ 3 → 100% not-at-risk; sentiment_score ranges for the two classes barely
  overlap), meaning the label was constructed from these fields rather than being an independent
  outcome. Both were excluded from the final model — see the report for full diagnosis and reasoning.

## Analytics Methods Used

Binary classification of `churn_risk` from review text (TF-IDF) and behavioural/metadata features
(`review_length`, `word_count`, `thumbs_up_count`, `has_reply`, `app_category`) — deliberately excluding
`rating` and `sentiment_score` to avoid label leakage — using a scikit-learn pipeline (TF-IDF + One-Hot
Encoding + Standard Scaling):

- **Logistic Regression** (interpretable baseline)
- **Random Forest Classifier** (primary model)

Supplementary methods:
- **KMeans clustering** for unsupervised review segmentation (rating, sentiment, length, thumbs-up).
- **Reply-effectiveness comparison** — churn signal strength for reviews with vs. without a developer
  reply, as a retention-relevant proxy.

## Key Results

*(fill in with your final corrected-model run before submission — see `analysis.ipynb`, Section 5.3b)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (corrected) | {fill in} | {fill in} | {fill in} | {fill in} | {fill in} |
| Random Forest (corrected) | {fill in} | {fill in} | {fill in} | {fill in} | {fill in} |

**Top predictive signals (Random Forest, corrected model):** {fill in from the feature importance
chart — e.g. review length, thumbs-up count, specific TF-IDF terms, app category}

## Business Insights (see report Section 6 for full detail)

- Complaint themes are not evenly distributed — a small number of categories (e.g. delivery issues,
  returns/refunds) account for a disproportionate share of at-risk reviews, pinpointing where
  operational fixes have the largest retention payoff.
- Churn-risk rate varies substantially by app (observed range roughly 5%–31% across apps in this
  dataset), suggesting app-specific rather than category-wide retention interventions.
- Review-based signals are available the same day a user is frustrated, well before that shows up in
  transactional churn data — making this a genuinely leading, not lagging, indicator.
- {fill in the reply-effectiveness finding once run: do reviews with a developer reply show weaker
  churn signal than similar reviews without one?}

## Repository Structure

```
README.md              this file
data/                   collected and cleaned review dataset (CSV)
analysis.ipynb          full cleaning -> EDA -> text analysis -> leakage diagnosis -> modeling -> segmentation pipeline (executed)
Case_Study_Report.pdf   final report (problem, methods, literature comparison, insights, references)
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook analysis.ipynb
```

Update `DATA_PATH` in the notebook to point to your CSV in `data/`, then run all cells top to bottom.
