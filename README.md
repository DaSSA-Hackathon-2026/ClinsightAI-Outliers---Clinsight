# ClinsightAI — AI-Powered Healthcare Review Intelligence

**DaSSA Hackathon 2026 | ClinSight Track**

---

## Problem Statement

Healthcare providers receive hundreds of unstructured patient reviews, but have no systematic way to extract what operational issues actually drive low ratings. A hospital administrator reading reviews one by one cannot quantify whether "long wait times" or "rude staff" hurts their ratings more, how frequently each issue occurs, or what to fix first.

**The core problem:** Unstructured text feedback is not actionable. Hospitals need a system that converts raw reviews into quantified operational themes, ranks them by impact on ratings, and outputs a prioritized improvement roadmap with measurable KPIs.

### Why We Chose This Problem

- Patient reviews are abundant but underutilized — most hospitals never systematically analyze them.
- Sentiment analysis alone ("positive/negative") is shallow and tells administrators nothing about *what specifically* to fix.
- The gap between "we know patients are unhappy" and "here is exactly what to fix, in what order, with expected rating lift" is where real business value lives.
- This is a problem that can be solved *today* with existing NLP techniques and delivered as a working tool.

### Who Is the User

Hospital operations managers, clinic owners, and quality improvement teams who need data-driven decisions — not anecdotal impressions — about where to allocate resources.

### What Success Looks Like

- Every review is assigned to a meaningful operational theme (not just "positive" or "negative").
- Each theme has a quantified impact on star ratings.
- The system outputs a ranked improvement roadmap: what to fix first, expected rating lift, and KPIs to track.
- A non-technical stakeholder can look at the dashboard and make a resource allocation decision within 5 minutes.

### Key Assumptions

- The Kaggle hospital reviews dataset (996 reviews, ratings 1–5) is representative of real-world healthcare feedback.
- Star ratings are a valid proxy for patient satisfaction.
- Operational themes are discoverable through unsupervised clustering of review embeddings.
- Theme-level interventions can meaningfully shift aggregate ratings.

---

## Solution Overview

ClinsightAI is a four-stage pipeline:

1. **Embed** — Convert each review into a dense vector using a sentence transformer.
2. **Cluster** — Group reviews into operational themes via KMeans (optimal K selected by silhouette score).
3. **Quantify** — Measure each theme's impact on ratings using regression coefficients and ML feature importance.
4. **Act** — Generate a severity-ranked improvement roadmap with KPIs, expected rating lift, and effort buckets.

The output is a structured JSON report consumed by a Streamlit dashboard, enabling drill-down into any theme with evidence samples, impact metrics, and actionable recommendations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAW DATA LAYER                       │
│  hospital.csv (996 reviews, ratings 1-5)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               PREPROCESSING                             │
│  • Drop nulls & invalid ratings                         │
│  • Text normalization (whitespace, encoding)             │
│  • Save: reviews_clean.csv                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            EMBEDDING & CLUSTERING                       │
│  • SentenceTransformer (all-MiniLM-L6-v2) → 384-dim    │
│  • KMeans (K=6..20, best K=7 by silhouette)             │
│  • TF-IDF keyword extraction per cluster                │
│  • Representative review selection (centroid distance)   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           IMPACT QUANTIFICATION                         │
│  • LinearRegression: cluster → rating coefficient       │
│  • RandomForest: low vs high rating classification      │
│  • Severity score = |impact| × frequency                │
│  • Confidence scoring per theme                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          BUSINESS OUTPUT LAYER                          │
│  • clinsightai_report.json (structured report)          │
│  • Improvement roadmap (prioritized, with KPIs)         │
│  • Executive summary + demo script                      │
│  • Streamlit dashboard (app.py)                         │
└─────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing

**Source:** Kaggle hospital review dataset — 996 rows, 3 useful columns: `Feedback` (free text), `Ratings` (1–5), `Sentiment Label`.

**Steps performed:**

1. Dropped the empty `Unnamed: 3` column.
2. Renamed columns to a clean schema: `review_text`, `rating`, `sentiment_label`.
3. Normalized text: stripped newlines, tabs, collapsed whitespace.
4. Filtered to valid ratings (1–5 only).
5. Verified zero null values and zero empty review texts post-cleaning.
6. Saved to `data_clean/reviews_clean.csv`.

**What we did NOT do (and why):**

- No stopword removal or stemming at the preprocessing stage — the embedding model handles semantic meaning directly, and TF-IDF later applies its own stopword filtering.
- No deduplication — the dataset is small enough that duplicates (if any) would surface as natural cluster emphasis, not noise.

---

## Modeling & AI Strategy

### Why This Approach

We rejected the obvious "just run sentiment analysis" path. Sentiment tells you *how people feel* but not *what to fix*. Our approach answers: "Which specific operational issue, if addressed, would produce the largest rating improvement?"

### Model Stack

| Component | Model/Method | Why |
|---|---|---|
| Embedding | `all-MiniLM-L6-v2` (SentenceTransformer) | Fast, 384-dim, strong semantic capture. Good enough for clustering; no need for larger models on 996 reviews. |
| Clustering | KMeans (K=7, selected via silhouette score over K=6..20) | Simple, interpretable, deterministic. Silhouette-based selection avoids arbitrary K. |
| Theme labeling | TF-IDF (top-10 bigrams per cluster) + manual review of representative samples | Automated keyword extraction gives objectivity; human review gives meaning. |
| Impact estimation | LinearRegression (cluster dummies → rating) | Gives signed coefficients: which themes push ratings up vs. down, and by how much. |
| Classification | RandomForest (cluster dummies → low/high rating) | Feature importance validates regression findings from a different angle. ~83% accuracy. |
| Severity ranking | `|impact| × frequency` | Neither impact alone nor frequency alone is sufficient. A rare catastrophic issue and a frequent mild issue both matter — this composite captures both. |

### Alternatives Considered and Rejected

- **LDA/NMF topic modeling:** Produces word distributions, not semantically coherent themes. Harder to map to actionable categories.
- **GPT-based theme extraction:** Expensive, non-deterministic, and unnecessary for 996 reviews. The hybrid (embedding + clustering + TF-IDF) gives comparable theme quality with full reproducibility.
- **Deep learning classifiers (BERT fine-tuning):** Overkill for this dataset size. RandomForest on cluster features achieves 83% accuracy, which is sufficient for the business question.

### Hybrid Approach

The system is intentionally hybrid: semantic embeddings for clustering (captures meaning), TF-IDF for keyword extraction (captures surface-level terms), regression for impact estimation (captures linear effect), and classification for validation (captures nonlinear importance). No single method does all four well.

---

## Evaluation Metrics & Results

### Metrics Used

| Metric | What It Measures | Result |
|---|---|---|
| Silhouette Score | Cluster cohesion and separation | Best K=7 (highest silhouette in 6–20 range) |
| R² (Linear Regression) | Variance in ratings explained by theme membership | Reported in notebook output |
| Accuracy (RandomForest) | Correct classification of low vs. high rating from theme | ~83% |
| Feature Importance | Which themes most predict rating class | Bar chart saved to `outputs/eval_feature_importance.png` |
| Severity Score | Composite: `|rating_impact| × frequency` | Ranks themes by operational priority |
| Confidence Score | Per-theme reliability metric | Included in report JSON |

### Test Cases

5 structured test cases saved to `outputs/test_cases.csv`:

- **TC-01, TC-02:** Top 2 highest-risk themes — validates that the system correctly identifies the most damaging operational issues.
- **TC-03, TC-04:** Mid-tier themes — validates that the system distinguishes moderate from severe issues.
- **TC-05:** Top positive driver — validates that the system correctly identifies strengths worth protecting.

Each test case includes: input review text (from evidence samples), expected theme assignment, expected rating impact direction, risk score, and confidence.

### Evaluation Visualization

`outputs/eval_feature_importance.png` — horizontal bar chart showing RandomForest feature importance per theme. This directly answers: "Which themes are most predictive of whether a review is low-rated or high-rated?"

### Limitations

- Silhouette score optimizes geometric separation, not semantic meaningfulness. Some clusters may be geometrically distinct but operationally overlapping.
- Linear regression assumes additive, independent theme effects. In reality, "long wait + rude staff" may compound nonlinearly.
- 996 reviews is a small dataset. Confidence intervals on impact estimates are wide.
- Theme labels are partially manual — a different analyst might name them differently.

---

## Business Impact & Actionability

### What the System Outputs

1. **Clinic Summary:** Overall rating mean, top risk themes, top growth drivers.
2. **Theme Analysis Table:** For each theme — frequency %, average rating, rating impact, ML importance, risk score, confidence, evidence samples, suggested KPIs, and explanation of why it was identified.
3. **Improvement Roadmap:** Prioritized list with effort bucket (quick-win / high-effort), expected rating lift, confidence, specific recommendation, and KPIs to track.
4. **Executive Summary:** Plain-language paragraph for non-technical stakeholders.
5. **KPI Checklist:** Weekly tracking items tied to each roadmap action.

### Can This Be Used Tomorrow?

Yes. The output is structured enough that a hospital operations manager can:

- **Day 1:** Read the executive summary, identify the top 2 risk themes.
- **Week 1:** Implement quick-win recommendations (e.g., queue transparency, proactive patient updates, service recovery SOPs).
- **Week 2–4:** Launch high-effort changes (scheduling redesign, triage optimization, staff training).
- **Ongoing:** Track KPIs weekly against the checklist to measure whether interventions are working.

### Discovered Themes (K=7)

| Cluster | Theme | Avg Rating | Impact |
|---|---|---|---|
| 0 | Clinical Quality vs Management Gap | — | Negative |
| 1 | High Satisfaction – Overall Care | — | Positive |
| 2 | Wait Time & Poor Coordination | — | Negative |
| 3 | Staff Courtesy & Service Quality | — | Mixed |
| 4 | General Facilities & Infrastructure | — | Mixed |
| 5 | Premium Service Experience | — | Positive |
| 6 | Individual Staff Appreciation | — | Positive |

(Exact values in `outputs/clinsightai_report.json`)

---

## Visualization & Dashboard

The Streamlit dashboard (`app.py`) provides:

- **Summary metrics:** Overall rating, primary risks, growth drivers.
- **Risk & Impact Ranking Table:** Sortable by risk score, ML importance, or frequency.
- **Theme Drill-Down:** Select any theme to see evidence samples, KPIs, confidence, and explanation.
- **Improvement Roadmap:** Prioritized table with interactive KPI checklist (checkboxes per action item).
- **JSON Export:** Download the full structured report.

---

## How to Run the Project

### Prerequisites

```bash
Python 3.9+
pip install pandas numpy scikit-learn sentence-transformers matplotlib streamlit
```

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd clinsightai

# 2. Place the dataset
# Put hospital.csv in data_raw/

# 3. Run the analysis notebook
jupyter notebook notebooks/01_load_and_inspect.ipynb
# Execute all cells in order. This will:
#   - Clean data → data_clean/reviews_clean.csv
#   - Generate embeddings, clusters, impact scores
#   - Produce outputs/clinsightai_report.json
#   - Generate test cases, evaluation plots, executive summary

# 4. Launch the dashboard
streamlit run app.py
```

### Project Structure

```
clinsightai/
├── data_raw/
│   └── hospital.csv              # Raw Kaggle dataset
├── data_clean/
│   └── reviews_clean.csv         # Preprocessed reviews
├── notebooks/
│   └── 01_load_and_inspect.ipynb # Full analysis pipeline
├── outputs/
│   ├── clinsightai_report.json   # Structured report (main output)
│   ├── test_cases.csv            # 5 evaluation test cases
│   ├── eval_feature_importance.png # Feature importance chart
│   ├── theme_keywords.csv        # TF-IDF keywords per cluster
│   ├── executive_summary.txt     # Stakeholder-ready summary
│   ├── demo_script.txt           # 5-7 min presentation script
│   └── kpis.txt                  # Weekly KPI tracking list
├── app.py                        # Streamlit dashboard
└── README.md
```

---

## Compliance Statement

- **Data:** Publicly available Kaggle hospital review dataset. No Protected Health Information (PHI). No personally identifiable information (PII).
- **Models:** All models used are open-source (SentenceTransformers, scikit-learn). No proprietary APIs required for core functionality.
- **Reproducibility:** All random seeds are fixed (`random_state=42`). Pipeline is deterministic end-to-end.
- **Ethical considerations:** The system analyzes aggregate review patterns — it does not target individual patients or staff members. Theme labels are operational categories, not judgments about individuals. Recommendations are directional; actual implementation should involve clinical and operational leadership.

---

## Team

**DaSSA Hackathon 2026 — ClinSight Track**
