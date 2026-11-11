import datetime as dt
import os
import sqlite3

import pandas as pd
import streamlit as st

DB = "traffic.db"
st.set_page_config(page_title="Website Traffic Forecasting", layout="wide")
st.title("AI-Based Website Traffic Forecasting")
st.caption("Daily website traffic, model forecasts and model performance")


def load(table):
    if not os.path.exists(DB):
        raise FileNotFoundError(f"{DB} not found in {os.getcwd()}")
    con = sqlite3.connect(DB)
    try:
        d = pd.read_sql(f"SELECT * FROM {table}", con)
    finally:
        con.close()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
    return d


try:
    hist = load("traffic_history").set_index("date").sort_index()
    fut = load("forecasts").set_index("date").sort_index()
    bt = load("backtest_predictions").set_index("date").sort_index()
    met = load("model_metrics")
except Exception as e:
    st.error(f"Could not read {DB} from {os.getcwd()}: {e}. Finish run_all.py (or unzip outputs.zip) in this same session, then refresh.")
    st.stop()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
lo, hi = hist.index.min().date(), hist.index.max().date()
rng = st.sidebar.date_input("Date range", value=(max(lo, hi - dt.timedelta(days=365)), hi),
                            min_value=lo, max_value=hi)
view_as = st.sidebar.selectbox("View as", ["Daily", "Weekly", "Monthly"])
rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[view_as]

if not isinstance(rng, (tuple, list)) or len(rng) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()
start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
view = hist.loc[start:end]
if view.empty:
    st.warning("No data in the selected range.")
    st.stop()

# ---------- Current and historical traffic ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Latest day ({hist.index[-1].date()})", f"{int(hist['unique_visits'].iloc[-1]):,}")
c2.metric("Last 30 days (avg/day)", f"{hist['unique_visits'].iloc[-30:].mean():,.0f}")
c3.metric("Selected range (avg/day)", f"{view['unique_visits'].mean():,.0f}")
c4.metric("Forecast next 7 days (avg/day)", f"{fut['predicted_unique_visits'].iloc[:7].mean():,.0f}")

# ---------- History + forecast ----------
st.subheader(f"Unique visits: history and forecast ({view_as.lower()} average per day)")
h = view["unique_visits"].resample(rule).mean().rename("History")
f = fut["predicted_unique_visits"].resample(rule).mean().rename("Forecast")
st.line_chart(pd.concat([h, f], axis=1, sort=True))
st.caption(f"Forecast model: {fut['model'].iloc[0]}. Forecast covers {fut.index.min().date()} to {fut.index.max().date()}.")

# ---------- Actual vs predicted ----------
st.subheader("Actual vs predicted (test period)")
options = [c for c in bt.columns if c != "actual"]
best = met.sort_values("RMSE").iloc[0]["model"] if "model" in met.columns else options[0]
model_pick = st.selectbox("Model", options, index=options.index(best) if best in options else 0)
st.line_chart(bt[["actual", model_pick]])

# ---------- Peaks, trends ----------
st.subheader("Peak and low traffic")
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
colA, colB = st.columns(2)
with colA:
    st.write("Average unique visits by weekday (selected range)")
    st.bar_chart(view.groupby("day_name")["unique_visits"].mean().reindex(order))
with colB:
    st.write("Busiest and quietest days (selected range)")
    peaks = pd.concat([view["unique_visits"].nlargest(5).rename("Busiest"),
                       view["unique_visits"].nsmallest(5).rename("Quietest")], axis=1, sort=True)
    st.dataframe(peaks)

st.write("Expected peak and low days in the forecast")
colC, colD = st.columns(2)
colC.dataframe(fut["predicted_unique_visits"].nlargest(5).rename("Peak (forecast)"))
colD.dataframe(fut["predicted_unique_visits"].nsmallest(5).rename("Low (forecast)"))

# ---------- Other metrics ----------
st.subheader("Other metrics (selected range)")
st.line_chart(view[["page_loads", "first_time_visits", "returning_visits"]].resample(rule).mean())

# ---------- Model performance ----------
st.subheader("Model performance (test period)")
st.dataframe(met.sort_values("RMSE"))

# ---------- Downloads ----------
st.subheader("Downloads")
st.download_button("Download forecast (CSV)", fut.reset_index().to_csv(index=False),
                   file_name="future_forecast.csv", mime="text/csv")
if os.path.exists("traffic_analysis_report.pdf"):
    with open("traffic_analysis_report.pdf", "rb") as fh:
        st.download_button("Download analysis report (PDF)", fh.read(),
                           file_name="traffic_analysis_report.pdf", mime="application/pdf")
