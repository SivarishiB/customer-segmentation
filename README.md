# Customer Personality Analysis — Customer Segmentation

Unsupervised machine learning project that groups a retail company's customers into segments, so marketing can target each product at the customers most likely to buy it instead of everyone in the database.

**Live app:** _add Streamlit link here_
**Presentation:** _add slides link here_

---

## Problem statement

Customer Personality Analysis is a detailed analysis of a company's ideal customers. It helps the business understand customers' needs, behaviours and concerns, and adapt products and campaigns to each type of customer.

**Objective:** perform clustering to summarise customer segments, describe each segment in business terms, and recommend a marketing approach for each.

## Dataset

`data/marketing_campaign.xlsx` — 2,240 customers × 29 columns, customers enrolled July 2012 – June 2014.

| Group | Columns |
|---|---|
| **People** | `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer`, `Recency`, `Complain` |
| **Products** (spend, last 2 years) | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| **Promotion** | `NumDealsPurchases`, `AcceptedCmp1`–`AcceptedCmp5`, `Response` |
| **Place** | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |

### Data quality issues handled

- `Income` has 24 missing values → filled with the median
- `Year_Birth` contains values as early as 1893 (age 115+) → rows with age ≥ 90 removed
- One `Income` value of 666,666 → removed as an outlier
- `Marital_Status` contains invalid values (`YOLO`, `Absurd`, `Alone`) → grouped as living alone
- `Z_CostContact` and `Z_Revenue` are constant → dropped

## Approach

1. **Data cleaning** — missing values, outliers, invalid categories
2. **Feature engineering** — `Age` (relative to 2014), `Children`, `Family_Size`, `Total_Spent`, `Total_Purchases`, `Total_Accepted`, `Customer_Days`, `Education_Level`
3. **Exploratory data analysis** — distributions, income vs spend, spend by children, channel usage, campaign acceptance, correlations
4. **Preprocessing** — `StandardScaler` on selected features, PCA to 3 components
5. **Choosing k** — elbow method, silhouette score, Davies–Bouldin index
6. **Modeling** — K-Means (final), compared with Agglomerative Clustering and Gaussian Mixture
7. **Segment profiling** — average behaviour per cluster, persona names, marketing recommendations
8. **Deployment** — Streamlit app that assigns a new customer (or an uploaded CSV of customers) to a segment

## Results

K-Means, k = 4, on 3 PCA components (silhouette 0.315, Davies–Bouldin 1.148). 2,236 customers after cleaning.

| Segment | Customers | Avg income | Avg 2-year spend |
|---|---|---|---|
| Premium | 534 | 76,376 | 1,397 |
| Steady families | 564 | 59,998 | 840 |
| Budget parents | 557 | 43,397 | 140 |
| Young low spenders | 581 | 29,898 | 100 |

| Segment | Profile | Marketing approach |
|---|---|---|
| Premium | High income, no kids, highest spend, most responsive to campaigns | Premium wine and meat offers, loyalty programme |
| Steady families | Mid-high income, ~1 child, frequent buyers, use deals | Family bundles, discount campaigns |
| Budget parents | Lower income, ~2 children, low spend | Essentials and price-led promotions |
| Young low spenders | Lowest income, youngest, visit the website often but buy little | Web retargeting, first-purchase offers |

## Project structure

```
customer-segmentation/
├── data/
│   └── marketing_campaign.xlsx
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_modeling_kmeans.ipynb
│   └── 05_model_comparison_profiling.ipynb
├── src/
│   ├── preprocess.py        # shared cleaning + feature code (used by notebooks and app)
│   └── train.py             # trains the final model and saves app/model.pkl
├── app/
│   ├── app.py               # Streamlit app
│   └── model.pkl            # saved scaler + PCA + K-Means
├── docs/                    # project brief and dataset details
├── reports/
│   ├── figures/             # charts saved by the notebooks
│   └── final_presentation.pptx
├── requirements.txt
└── README.md
```

## Tech stack

| Purpose | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy, openpyxl |
| Visualisation | matplotlib, seaborn, plotly |
| Machine learning | scikit-learn (StandardScaler, PCA, KMeans, AgglomerativeClustering, GaussianMixture) |
| Model saving | joblib |
| Deployment | Streamlit, Streamlit Community Cloud |
| Collaboration | Git, GitHub |

## Getting started

```bash
git clone https://github.com/SivarishiB/customer-segmentation.git
cd customer-segmentation
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Train the model (creates `app/model.pkl`) and start the app, from the repo root:

```bash
python src/train.py
streamlit run app/app.py
```

To explore the analysis, open Jupyter (`jupyter notebook`) and run the notebooks in `notebooks/` in order, `01` to `05`.

All dependencies are listed in `requirements.txt`.

## Team

| # | Name | Role | GitHub |
|---|---|---|---|
| 1 | | Project Lead & Integrator | |
| 2 | | Data Cleaning | |
| 3 | | Feature Engineering | |
| 4 | | EDA & Visualisation | |
| 5 | | Modeling — K-Means | |
| 6 | | Model Comparison & Profiling | |
| 7 | | Deployment | |

## Timeline

| Milestone | Duration |
|---|---|
| Kick-off and business objective discussion | 1 day |
| EDA | 1 week |
| Model building & evaluation | 1 week |
| Deployment | 1 week |
| Final presentation | 1 day |

## Contributing (team workflow)

1. `git pull origin main` before starting work
2. Work on your own branch: `git checkout -b <your-task>`
3. Commit and push: `git push -u origin <your-task>`
4. Open a pull request on GitHub — the Project Lead reviews and merges into `main`
5. Do not edit another member's notebook; use `random_state=42` everywhere for reproducible clusters
