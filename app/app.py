"""Customer Segment Predictor — Streamlit app.

Run from the repo root:  streamlit run app/app.py
(Train first if app/model.pkl is missing:  python src/train.py)
"""
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import MODEL_PATH, RAW_INPUT_COLS, add_features, clean  # noqa: E402

st.set_page_config(page_title="Customer Segment Predictor", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(raw: pd.DataFrame, model) -> pd.DataFrame:
    df = add_features(clean(raw, model["income_median"]), model["reference_date"])
    X = model["pca"].transform(model["scaler"].transform(df[model["features"]]))
    clusters = model["kmeans"].predict(X)
    out = raw.copy()
    out["Cluster"] = clusters
    out["Segment"] = [model["segments"][int(c)]["name"] for c in clusters]
    out["Recommended action"] = [model["segments"][int(c)]["action"] for c in clusters]
    return out


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train.py` from the repo root first.")
    st.stop()
model = load_model()

st.title("Customer Segment Predictor")
st.caption("Customer Personality Analysis · K-Means clustering into 4 segments")

with st.expander("Segment profiles (averages from training data)"):
    st.dataframe(model["profile"], width="stretch")

single, batch = st.tabs(["One customer", "Upload CSV"])

with single:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("People")
        year_birth = st.number_input("Year of birth", 1940, 2000, 1970)
        education = st.selectbox("Education", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"], index=2)
        marital = st.selectbox("Marital status", ["Married", "Together", "Single", "Divorced", "Widow"])
        income = st.number_input("Yearly household income", 0, 200_000, 50_000, step=1_000)
        kidhome = st.number_input("Children at home", 0, 5, 0)
        teenhome = st.number_input("Teenagers at home", 0, 5, 0)
        dt_customer = st.date_input("Enrolment date", date(2013, 6, 1),
                                    min_value=date(2012, 1, 1), max_value=date(2014, 12, 31))
        recency = st.number_input("Days since last purchase", 0, 365, 30)
    with c2:
        st.subheader("Spend (last 2 years)")
        wines = st.number_input("Wine", 0, 5000, 300)
        fruits = st.number_input("Fruits", 0, 5000, 25)
        meat = st.number_input("Meat", 0, 5000, 150)
        fish = st.number_input("Fish", 0, 5000, 35)
        sweets = st.number_input("Sweets", 0, 5000, 25)
        gold = st.number_input("Gold", 0, 5000, 40)
    with c3:
        st.subheader("Purchases and campaigns")
        deals = st.number_input("Purchases with discount", 0, 30, 2)
        web = st.number_input("Web purchases", 0, 30, 4)
        catalog = st.number_input("Catalogue purchases", 0, 30, 2)
        store = st.number_input("Store purchases", 0, 30, 6)
        visits = st.number_input("Website visits last month", 0, 30, 5)
        accepted = st.multiselect("Campaigns accepted",
                                  ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                                   "AcceptedCmp4", "AcceptedCmp5", "Response"])

    if st.button("Predict segment", type="primary"):
        row = {
            "Year_Birth": year_birth, "Education": education, "Marital_Status": marital,
            "Income": income, "Kidhome": kidhome, "Teenhome": teenhome,
            "Dt_Customer": pd.Timestamp(dt_customer), "Recency": recency,
            "MntWines": wines, "MntFruits": fruits, "MntMeatProducts": meat,
            "MntFishProducts": fish, "MntSweetProducts": sweets, "MntGoldProds": gold,
            "NumDealsPurchases": deals, "NumWebPurchases": web,
            "NumCatalogPurchases": catalog, "NumStorePurchases": store,
            "NumWebVisitsMonth": visits,
        }
        for cmp in ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]:
            row[cmp] = int(cmp in accepted)
        result = predict(pd.DataFrame([row]), model).iloc[0]
        st.success(f"Segment: **{result['Segment']}**")
        st.info(f"Recommended action: {result['Recommended action']}")

with batch:
    st.write("Upload a CSV with these columns (same names as the original dataset):")
    st.code(", ".join(RAW_INPUT_COLS), language=None)
    file = st.file_uploader("Customer CSV", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in RAW_INPUT_COLS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            result = predict(data, model)
            st.dataframe(result, width="stretch")
            st.bar_chart(result["Segment"].value_counts())
            st.download_button("Download results", result.to_csv(index=False),
                               "segmented_customers.csv", "text/csv")
