import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
DATA = BASE / "dashboard_data"
MODELS = BASE / "models"

st.set_page_config(page_title="Sri Lanka Dengue Forecasting", page_icon="🦟", layout="wide")

TYPE_COLORS = {"Baseline": "#8c8c8c", "Single model": "#c8b28a", "Bagging": "#4c78a8",
               "Boosting": "#f58518", "Stacking": "#54a24b", "Time series": "#b279a2",
               "Hybrid": "#e45756"}


# data loading
@st.cache_data
def load_data():
    history = pd.read_csv(DATA / "history.csv", parse_dates=["start"])
    results = pd.read_csv(DATA / "model_results.csv")
    preds = pd.read_csv(DATA / "test_predictions.csv", parse_dates=["start"])
    importance = pd.read_csv(DATA / "feature_importance.csv", index_col=0)["importance"]
    next_week = pd.read_csv(DATA / "next_week_features.csv", parse_dates=["start"])
    return history, results, preds, importance, next_week


@st.cache_resource
def load_model():
    model = joblib.load(MODELS / "lightgbm_model.pkl")
    feature_cols = json.load(open(MODELS / "feature_cols.json"))
    return model, feature_cols


def risk_level(cases):
    if cases <= 20:
        return "Low"
    if cases <= 100:
        return "Medium"
    return "High"


RISK_COLORS = {"Low": "#54a24b", "Medium": "#f58518", "High": "#e45756"}

history, results, preds, importance, next_week = load_data()
model, feature_cols = load_model()
best = results[results["Type"] != "Baseline"].iloc[0]
baseline_rmse = results.loc[results["Type"] == "Baseline", "RMSE"].iloc[0]
districts = sorted(history["district"].unique())

# sidebar 
st.sidebar.title("🦟 Dengue Forecasting")
page = st.sidebar.radio("Go to", ["Home", "Data explorer", "Model comparison",
                                  "Model insights", "Forecast"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {history['start'].min():%d %b %Y} to {history['start'].max():%d %b %Y}")
st.sidebar.caption(f"{len(districts)} districts · {len(history):,} district-weeks")
st.sidebar.caption("Sources: Epidemiology Unit, Ministry of Health Sri Lanka (via denguedatahub); Open-Meteo weather archive.")

# pages
if page == "Home":
    st.title("Forecasting weekly dengue cases in Sri Lanka")
    st.markdown(
        "This dashboard predicts **next week's dengue cases for each of Sri Lanka's 26 health districts** "
        "using past cases, rainfall, temperature and humidity. It compares bagging, boosting, stacking "
        "and a hybrid RF + ARIMA model against a simple *same as last week* baseline."
    )
    this_year = history["start"].dt.year.max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Cases in {this_year} (to {history['start'].max():%b})",
              f"{history.loc[history['start'].dt.year == this_year, 'cases'].sum():,.0f}")
    c2.metric("Worst year on record", "2017",
              f"{history.loc[history['start'].dt.year == 2017, 'cases'].sum():,.0f} cases", delta_color="off")
    c3.metric("Best model", best["Model"].split(" (")[0])
    c4.metric("Best model R²", f"{best['R2']:.3f}",
              f"{best['Skill vs baseline']:.1%} better than baseline")

    yearly = history.groupby(history["start"].dt.year)["cases"].sum().reset_index()
    yearly.columns = ["Year", "Cases"]
    fig = px.bar(yearly, x="Year", y="Cases", title="Total dengue cases per year",
                 color_discrete_sequence=["#b22222"])
    st.plotly_chart(fig, width="stretch")
    st.info("Use the menu on the left to explore the data, compare models and see next week's forecast.")

elif page == "Data explorer":
    st.title("Data explorer")
    district = st.selectbox("District", districts, index=districts.index("Colombo"))
    years = st.slider("Years", int(history["start"].dt.year.min()), int(history["start"].dt.year.max()),
                      (2017, int(history["start"].dt.year.max())))
    d = history[(history["district"] == district) & history["start"].dt.year.between(*years)]

    fig = go.Figure()
    fig.add_bar(x=d["start"], y=d["rain_mm"], name="Rainfall (mm/week)", yaxis="y2",
                marker_color="rgba(76,120,168,0.35)")
    fig.add_scatter(x=d["start"], y=d["cases"], name="Dengue cases", line=dict(color="#b22222"))
    fig.update_layout(title=f"{district}: weekly dengue cases and rainfall",
                      yaxis=dict(title="Cases per week"),
                      yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    totals = history.groupby("district")["cases"].sum().sort_values().reset_index()
    c1.plotly_chart(px.bar(totals, x="cases", y="district", orientation="h", height=650,
                           title="Total cases by district (all years)",
                           color_discrete_sequence=["#4c78a8"]), width="stretch")
    monthly = history[history["district"] == district].copy()
    monthly["Month"] = monthly["start"].dt.month
    season = monthly.groupby("Month")[["cases", "rain_mm"]].mean().reset_index()
    fig2 = go.Figure()
    fig2.add_bar(x=season["Month"], y=season["rain_mm"], name="Avg rainfall (mm/week)", yaxis="y2",
                 marker_color="rgba(76,120,168,0.35)")
    fig2.add_scatter(x=season["Month"], y=season["cases"], name="Avg cases/week", line=dict(color="#b22222"))
    fig2.update_layout(title=f"{district}: average by month (all years)", height=650,
                       xaxis=dict(title="Month", dtick=1),
                       yaxis=dict(title="Cases per week"),
                       yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right", showgrid=False),
                       legend=dict(orientation="h", y=1.08))
    c2.plotly_chart(fig2, width="stretch")

elif page == "Model comparison":
    st.title("Model comparison")
    st.markdown("All models were trained on **2008–2021** and tested on unseen weeks from **2022–2026**. "
                "Lower RMSE and MAE are better; higher R² is better. *Skill vs baseline* shows how much "
                "better each model is than simply predicting *same as last week*.")
    show = results.copy()
    st.dataframe(show.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.3f}",
                                    "Skill vs baseline": "{:.1%}", "Outbreak-week MAE": "{:.1f}",
                                    "Train time (s)": "{:.1f}"})
                 .background_gradient(subset=["RMSE"], cmap="RdYlGn_r"),
                 width="stretch", hide_index=True)

    fig = px.bar(results.sort_values("RMSE", ascending=False), x="RMSE", y="Model", orientation="h",
                 color="Type", color_discrete_map=TYPE_COLORS, height=500,
                 title="RMSE by model (cases per week, lower is better)")
    fig.add_vline(x=baseline_rmse, line_dash="dash", line_color="black",
                  annotation_text="Same as last week", annotation_position="top")
    fig.update_xaxes(range=[results["RMSE"].min() * 0.85, results["RMSE"].max() * 1.03])
    st.plotly_chart(fig, width="stretch")

    by_type = results.groupby("Type")["RMSE"].min().sort_values().reset_index()
    st.plotly_chart(px.bar(by_type, x="Type", y="RMSE", color="Type", color_discrete_map=TYPE_COLORS,
                           title="Best RMSE in each family"), width="stretch")

    st.subheader("Key findings")
    st.markdown(
        "- **Stacking** gave the lowest error; its meta-model relied mostly on CatBoost and LightGBM.\n"
        "- **Boosting** models beat bagging on this data: each new tree corrects earlier errors, "
        "so they follow rises and falls more closely.\n"
        "- **Bagging** (Random Forest) clearly beats a single decision tree, but is close to the baseline.\n"
        "- **ARIMA alone** is the weakest; adding a Random Forest correction (hybrid) improves it noticeably.\n"
        "- Weekly dengue cases change slowly, so *same as last week* is a strong baseline."
    )

elif page == "Model insights":
    st.title("Model insights")
    model_names = [m for m in preds.columns if m not in ["district", "start", "cases"]]
    c1, c2 = st.columns([1, 2])
    district = c1.selectbox("District", districts, index=districts.index("Colombo"))
    chosen = c2.multiselect("Models", model_names,
                            default=[m for m in [best["Model"], "Same as last week"] if m in model_names])
    d = preds[preds["district"] == district]
    fig = go.Figure()
    fig.add_scatter(x=d["start"], y=d["cases"], name="Actual", line=dict(color="black", width=2))
    for m in chosen:
        fig.add_scatter(x=d["start"], y=d[m], name=m, line=dict(width=1.5))
    fig.update_layout(title=f"{district}: actual vs predicted weekly cases (test period 2022–2026)",
                      yaxis_title="Cases per week", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    top = importance.head(15).sort_values().reset_index()
    top.columns = ["Feature", "Importance"]
    c1.plotly_chart(px.bar(top, x="Importance", y="Feature", orientation="h", height=500,
                           title="Random Forest: top 15 features", color_discrete_sequence=["#4c78a8"]),
                    width="stretch")
    ob = results.sort_values("Outbreak-week MAE")
    c2.plotly_chart(px.bar(ob, x="Outbreak-week MAE", y="Model", orientation="h", color="Type",
                           color_discrete_map=TYPE_COLORS, height=500,
                           title="Error in the busiest 10% of weeks (outbreaks)"),
                    width="stretch")
    st.caption("Recent case counts (4-week average, last week) dominate the forecast. "
               "Rainfall features add a smaller signal, mainly from 6–10 weeks earlier.")

elif page == "Forecast":
    st.title("Next week's forecast")
    week = next_week["start"].iloc[0]
    st.markdown(f"Forecast from the saved **LightGBM** model for the week starting **{week:%d %B %Y}**. "
                "Risk levels: Low (0–20 cases), Medium (21–100), High (over 100).")
    nw = next_week.copy()
    nw["forecast"] = np.round(np.clip(np.expm1(model.predict(nw[feature_cols])), 0, None))
    nw["risk"] = nw["forecast"].apply(risk_level)

    district = st.selectbox("District", districts, index=districts.index("Colombo"))
    row = nw[nw["district"] == district].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Last reported week", f"{row['cases_lag1']:.0f} cases")
    c2.metric("Forecast for next week", f"{row['forecast']:.0f} cases",
              f"{row['forecast'] - row['cases_lag1']:+.0f}")
    c3.markdown(f"### Risk: <span style='color:{RISK_COLORS[row['risk']]}'>{row['risk']}</span>",
                unsafe_allow_html=True)

    recent = history[history["district"] == district].tail(16)
    fig = go.Figure()
    fig.add_scatter(x=recent["start"], y=recent["cases"], name="Reported", mode="lines+markers",
                    line=dict(color="black"))
    fig.add_scatter(x=[recent["start"].iloc[-1], week], y=[recent["cases"].iloc[-1], row["forecast"]],
                    name="Forecast", mode="lines+markers", line=dict(color="#e45756", dash="dash"))
    fig.update_layout(title=f"{district}: last 16 weeks and next week's forecast", yaxis_title="Cases per week")
    st.plotly_chart(fig, width="stretch")

    st.subheader("All districts")
    table = nw[["district", "cases_lag1", "forecast", "risk"]].rename(columns={
        "district": "District", "cases_lag1": "Last week", "forecast": "Forecast", "risk": "Risk"})
    st.dataframe(table.sort_values("Forecast", ascending=False), width="stretch", hide_index=True)