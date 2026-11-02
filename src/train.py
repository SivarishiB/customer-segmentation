"""Train the final segmentation model and save it to app/model.pkl.

Run from the repo root:  python src/train.py
"""
import sys
from pathlib import Path

import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import (FEATURES, MODEL_PATH, RANDOM_STATE, clean, load_clean_data,  # noqa: E402
                            load_raw)

N_CLUSTERS = 4
N_COMPONENTS = 3

# Segments are named by average income rank (lowest -> highest), so names stay
# correct even if K-Means numbers the clusters differently.
SEGMENTS_BY_INCOME = [
    ("Young low spenders", "Web retargeting and first-purchase offers"),
    ("Budget parents", "Essentials and price-led promotions"),
    ("Steady families", "Family bundles and discount campaigns"),
    ("Premium", "Premium wine and meat offers, loyalty programme"),
]
PROFILE_COLS = ["Income", "Total_Spent", "Age", "Children", "Total_Purchases",
                "NumDealsPurchases", "NumWebVisitsMonth", "Total_Accepted"]


def train(n_clusters: int = N_CLUSTERS):
    df = load_clean_data()
    scaler = StandardScaler()
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    X = pca.fit_transform(scaler.fit_transform(df[FEATURES]))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    df["Cluster"] = kmeans.fit_predict(X)

    profile = df.groupby("Cluster")[PROFILE_COLS].mean().round(1)
    profile["Customers"] = df["Cluster"].value_counts()
    segments = {}
    if n_clusters == len(SEGMENTS_BY_INCOME):
        for rank, cluster in enumerate(profile["Income"].sort_values().index):
            name, action = SEGMENTS_BY_INCOME[rank]
            segments[int(cluster)] = {"name": name, "action": action}
    else:
        segments = {int(c): {"name": f"Segment {c}", "action": "-"} for c in profile.index}
    profile.insert(0, "Segment", [segments[int(c)]["name"] for c in profile.index])

    raw = clean(load_raw())
    model = {
        "scaler": scaler, "pca": pca, "kmeans": kmeans, "features": FEATURES,
        "segments": segments, "profile": profile,
        "income_median": float(raw["Income"].median()),
        "reference_date": raw["Dt_Customer"].max(),
        "silhouette": float(silhouette_score(X, df["Cluster"])),
        "davies_bouldin": float(davies_bouldin_score(X, df["Cluster"])),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model, df


if __name__ == "__main__":
    model, df = train()
    print(f"Saved {MODEL_PATH}")
    print(f"Silhouette: {model['silhouette']:.3f}   Davies-Bouldin: {model['davies_bouldin']:.3f}")
    print(model["profile"].to_string())
