# 🎵 Spotify Hip-Hop Popularity & Mood Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science and machine learning capstone project analyzing 991 hip-hop tracks from Spotify. This project investigates the drivers of track popularity, evaluates supervised regression and classification models, clusters track mood profiles using unsupervised learning, and performs hyperparameter optimization.

---

## 📌 Executive Summary

* **The Problem:** Streaming platforms struggle to forecast song success and categorize vast catalogs using raw audio metadata alone.
* **The Goal:** Build predictive models for continuous popularity ($0\text{--}100$) and binary hit classification (`is_hit`), construct actionable mood profiles via $K$-Means clustering, and deliver strategic recommendations for playlist editors.
* **Key Finding:** Raw audio characteristics (e.g., danceability, valence, energy) explain **50.9% of popularity variance** ($R^2 = 0.509$). Non-audio metadata (artist fanbase, TikTok virality, promotional backing) accounts for the remaining variance.
* **Top Insight:** Unsupervised clustering ($k=4$) identified **Aggressive / High-Intensity Hype** tracks as the most popular mood cluster (**45.85 average popularity**).

---

## 📊 Dataset & Cleaning

The dataset was fetched from the Spotify Web API audio features endpoint and filtered strictly to the `hip-hop` genre to prevent cross-genre baseline distortion.

| Metric / Attribute | Value / Description |
| :--- | :--- |
| **Raw Tracks** | ~1,000 tracks |
| **Cleaned Sample Size** | **991 tracks** (post-deduplication & null removal) |
| **Continuous Target** | `popularity` ($0\text{--}100$ continuous scale based on recent streams) |
| **Classification Target** | `is_hit` (Binary: $1$ if popularity $>$ median, $0$ otherwise) |
| **Key Features** | `danceability`, `energy`, `valence`, `loudness`, `tempo`, `acousticness`, `duration_min` |

---

## 🚀 Machine Learning Pipeline

```
Raw Data ➔ Genre Filtering ➔ Cleaning & Deduplication ➔ Data Leakage Verification
                                                               │
     ┌─────────────────────────┬───────────────────────────────┴──────────────────────────────┐
     ▼                         ▼                                                             ▼
[Regression]            [Classification]                                            [Unsupervised Clustering]
Predict Popularity       Classify `is_hit`                                           K-Means on Valence & Energy
• Linear Regression      • Logistic Regression (Scaled)                              • Elbow Plot Evaluation
• Random Forest          • Decision Tree Classifier                                  • 4 Mood Profiles Created
                               │
                               ▼
                       [GridSearchCV]
                       Tune Decision Tree
```

---

## 📈 Key Results & Model Evaluation

### 1. Predicting Popularity (Regression)

| Model | RMSE | $R^2$ (Variance Explained) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | $30.605$ | $0.128$ ($12.8\%$) | Underfits due to linear constraints |
| **Random Forest Regressor** | **$22.978$** | **$0.509$ ($50.9\%$)** | Captures non-linear feature interactions |

* **Interpretation:** The best model predicts track popularity to within about **$\pm 23.0$ points** on a $0\text{--}100$ scale.

---

### 2. Spotting Hits (Classification)

Strict data leakage checks were conducted to ensure `popularity` and `is_hit` were completely removed from feature matrix $X$.

| Model | Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | $0.58$ | $0.59$ | $0.56$ |
| **Decision Tree (Baseline, depth=4)** | $0.61$ | $0.62$ | $0.59$ |
| **Tuned Decision Tree (`max_depth=8`)** | **$0.61\text{--}0.62$** | **$0.62$** | **$0.60$** |

* **Cost Matrix Evaluation:** **False Positives** (wrongly predicting a non-hit as a hit) cost streaming services significantly more than False Negatives, as placing weak tracks on flagship editorial playlists like *RapCaviar* causes user churn.

---

### 3. Mood Profiles ($K$-Means Clustering, $k=4$)

Tracks were clustered on standardized `valence` (positivity) and `energy` (intensity) to build $4$ distinct playlist mood profiles:

| Cluster | Playlist Mood Name | Valence | Energy | Danceability | Avg Popularity |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | **Upbeat / Laid-Back Vibe** | $0.64$ | $0.57$ | $0.77$ | $40.53$ |
| **1** | **Dark / Melancholic** | $0.29$ | $0.51$ | $0.71$ | $30.52$ |
| **2** | **High-Energy Party Anthem** | $0.76$ | $0.82$ | $0.76$ | $34.60$ |
| **3** | **Aggressive / High-Intensity Hype** ★ | **$0.39$** | **$0.73$** | **$0.70$** | **$45.85$** |

* **Representative Tracks (Mood 3):** *"STAY (with Justin Bieber)"*, *"Excuses"*, *"Brown Munde"*

---

## 💡 Strategic Recommendations

1. **Human-in-the-Loop Editorial Curation:** Audio classification models should be restricted to initial content screening. Human editor approval must be required before placing predicted hits onto major curated playlists to protect user experience.
2. **Prioritize Mood 3 (Hype) Recommendations:** Program recommendation algorithms to prioritize high-intensity, aggressive hip-hop tracks in primary genre radio feeds, as this mood profile yields the highest audience popularity ($45.85$ avg).
3. **Ingest External Contextual Metadata:** Expand future pipeline architectures to incorporate artist follower counts, social media virality scores (e.g., TikTok sound usage), and playlist placement history to bridge the $49.1\%$ unexplained variance gap.

---

## 🛠 Project Structure & Setup

```bash
├── data/
│   └── spotify_hiphop_tracks.csv   # Raw and cleaned dataset
├── notebooks/
│   └── capstone_analysis.ipynb     # Complete Jupyter Notebook (EDA, ML, Clustering)
├── reports/
│   └── spotify_hiphop_capstone_report.pdf # PDF Report
├── README.md                       # Project documentation
```

### Quickstart & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fullcapsule/Group-5---Spotify-Music-Popularity
   ```

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
