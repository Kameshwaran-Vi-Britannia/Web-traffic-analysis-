"""
AI-Based Website Traffic Forecasting - complete pipeline (one script).
Colab usage:   %run run_all.py
Needs: archive.zip (Kaggle "daily-website-visitors" download) in the same folder, or internet for kagglehub.

Stages (PDF section in brackets):
  1 Load data [1]  ->  2 Clean + validate [2]  ->  3 EDA charts [3]  ->  4 Train 5 models [4]
  5 Evaluate [6]   ->  6 Forecast next 90 days [5]  ->  7 Store in SQLite [8]  ->  8 PDF report [8]
"""
import glob
import os
import shutil
import sqlite3
import sys
import textwrap
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")
ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COUNTS = ["page_loads", "unique_visits", "first_time_visits", "returning_visits"]

# =====================================================================
# 1. LOAD DATA
# =====================================================================
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
csvs = glob.glob(f"{RAW_DIR}/**/*.csv", recursive=True)
if not csvs:
    for zp in glob.glob("*.zip"):
        with zipfile.ZipFile(zp) as z:
            z.extractall(RAW_DIR)
    csvs = glob.glob(f"{RAW_DIR}/**/*.csv", recursive=True)
if not csvs:                      # fallback: download with kagglehub (needs internet)
    try:
        import kagglehub
        p = kagglehub.dataset_download("bobnau/daily-website-visitors")
        for c in glob.glob(os.path.join(p, "**", "*.csv"), recursive=True):
            shutil.copy(c, RAW_DIR)
        csvs = glob.glob(f"{RAW_DIR}/**/*.csv", recursive=True)
    except Exception as e:
        print("kagglehub fallback failed:", type(e).__name__, e)
if not csvs:
    sys.exit("STOP: no CSV found. Upload archive.zip (Kaggle download) next to this script and run again.")
SRC = ([c for c in csvs if "visitor" in c.lower()] or csvs)[0]
print("Using data file:", SRC)

# =====================================================================
# 2. CLEAN + VALIDATE   (PDF Section 2)
# =====================================================================
raw = pd.read_csv(SRC, dtype=str)
raw.columns = [c.strip() for c in raw.columns]
rename = {"Row": "row", "Day": "day_name", "Day.Of.Week": "day_of_week", "Date": "date",
          "Page.Loads": "page_loads", "Unique.Visits": "unique_visits",
          "First.Time.Visits": "first_time_visits", "Returning.Visits": "returning_visits"}
missing_cols = [c for c in rename if c not in raw.columns]
if missing_cols:
    sys.exit(f"STOP: unexpected column names. Not found: {missing_cols}. Found: {list(raw.columns)}")
d0 = raw.rename(columns=rename)
for c in COUNTS + ["row", "day_of_week"]:
    d0[c] = pd.to_numeric(d0[c].str.replace(",", "", regex=False).str.strip(), errors="coerce")
d0["date"] = pd.to_datetime(d0["date"].str.strip(), format="%m/%d/%Y", errors="coerce")

print("\n--- VALIDATION ---")
print("Rows:", len(d0))
print("Missing values per column:\n", d0.isna().sum().to_string())
print("Duplicate rows:", d0.duplicated().sum(), "| duplicate dates:", d0["date"].duplicated().sum())
d0 = d0.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
full = pd.date_range(d0["date"].min(), d0["date"].max(), freq="D")
gaps = full.difference(d0["date"])
print("Date range:", d0["date"].min().date(), "to", d0["date"].max().date())
print("Expected days:", len(full), "| missing dates:", len(gaps))
print("Rows where unique != first_time + returning:",
      len(d0[d0["unique_visits"] != d0["first_time_visits"] + d0["returning_visits"]]))
print("Negative values:", int((d0[COUNTS] < 0).sum().sum()), "| zero values:", int((d0[COUNTS] == 0).sum().sum()))

df = d0.set_index("date").reindex(full)
df.index.name = "date"
if len(gaps):
    df[COUNTS] = df[COUNTS].interpolate(method="time")
    print("Interpolated", len(gaps), "missing days")


def robust_z(s):
    med, mad = s.median(), (s - s.median()).abs().median()
    return pd.Series(0.0, index=s.index) if mad == 0 else 0.6745 * (s - med) / mad


df["day_of_week"] = df.index.dayofweek
df["day_name"] = df.index.day_name()
df["is_outlier"] = df.groupby("day_of_week")["unique_visits"].transform(robust_z).abs() > 3.5
print("Outliers flagged (kept in data):", int(df["is_outlier"].sum()))
df = df.drop(columns=["row"])
df.to_csv("traffic_clean.csv")
df[COUNTS].resample("W").sum().to_csv("traffic_weekly.csv")
df[COUNTS].resample("MS").sum().to_csv("traffic_monthly.csv")

# =====================================================================
# 3. EDA CHARTS   (PDF Section 3)
# =====================================================================
print("\n--- EDA ---")
print(df[COUNTS].describe().round(0))
plt.figure(figsize=(11, 4)); plt.plot(df.index, df["unique_visits"])
plt.title("Daily unique visits"); plt.ylabel("Unique visits"); plt.show()
wk_all = df.groupby("day_name")["unique_visits"].mean().reindex(ORDER)
plt.figure(figsize=(8, 4)); plt.bar(ORDER, wk_all.values)
plt.title("Average unique visits by weekday"); plt.xticks(rotation=45); plt.show()
mo_all = df.groupby(df.index.month)["unique_visits"].mean()
plt.figure(figsize=(8, 4)); plt.bar(mo_all.index, mo_all.values)
plt.title("Average unique visits by month (1 = Jan)"); plt.show()
print("Top 5 busiest days:\n", df["unique_visits"].nlargest(5))
print("Top 5 quietest days:\n", df["unique_visits"].nsmallest(5))

# =====================================================================
# 4. MODELS   (PDF Section 4)
# =====================================================================
y = df["unique_visits"].astype(float).asfreq("D")
H = 90
train, test = y.iloc[:-H], y.iloc[-H:]
print(f"\nTrain: {len(train)} days to {train.index[-1].date()} | Test: {len(test)} days "
      f"{test.index[0].date()} to {test.index[-1].date()}")


def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    e = a - p
    mse = np.mean(e ** 2)
    return {"MAE": np.mean(np.abs(e)), "MSE": mse, "RMSE": np.sqrt(mse), "MAPE %": np.mean(np.abs(e / a)) * 100}


def fc_seasonal_naive(s, h):
    return np.tile(s.iloc[-7:].values, int(np.ceil(h / 7)))[:h]


def fc_arima(s, h):
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA(s, order=(2, 1, 2)).fit().forecast(h)


def fc_sarima(s, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    return SARIMAX(s, order=(2, 0, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False).forecast(h)


def fc_prophet(s, h):
    from prophet import Prophet
    p_df = s.reset_index()
    p_df.columns = ["ds", "y"]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(p_df)
    return m.predict(m.make_future_dataframe(periods=h, freq="D"))["yhat"].iloc[-h:].values


def fc_lstm(s, h):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    tf.random.set_seed(42)
    np.random.seed(42)
    L = 28
    sc = MinMaxScaler().fit(s.values.reshape(-1, 1))
    z = sc.transform(s.values.reshape(-1, 1)).ravel()
    X = np.array([z[i:i + L] for i in range(len(z) - L)])[..., None]
    t = z[L:]
    model = keras.Sequential([keras.Input(shape=(L, 1)), keras.layers.LSTM(32), keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, t, epochs=100, batch_size=32, validation_split=0.1, shuffle=False, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    window, out = list(z[-L:]), []
    for _ in range(h):
        p = model.predict(np.array(window[-L:]).reshape(1, L, 1), verbose=0)[0, 0]
        out.append(p)
        window.append(p)
    return sc.inverse_transform(np.array(out).reshape(-1, 1)).ravel()


FUNCS = {"Seasonal naive": fc_seasonal_naive, "ARIMA": fc_arima, "SARIMA": fc_sarima,
         "Prophet": fc_prophet, "LSTM": fc_lstm}

results, forecasts = {}, pd.DataFrame({"actual": test})
for name, fn in FUNCS.items():
    print(f"Training {name} ...")
    try:
        pred = np.asarray(fn(train, H), float)
        results[name] = metrics(test, pred)
        forecasts[name] = pred
        print(f"  {name}: done", {k: round(float(v), 2) for k, v in results[name].items()})
    except Exception as e:
        print(f"  {name}: FAILED -> {type(e).__name__}: {e}")

if not results:
    sys.exit("STOP: no model produced a result.")

# =====================================================================
# 5. EVALUATE   (PDF Section 6)
# =====================================================================
res = pd.DataFrame(results).T.sort_values("RMSE")
res.index.name = "model"
best = res.index[0]
print("\n--- MODEL COMPARISON (sorted by RMSE) ---")
print(res.round(2))
print("Best model:", best)

plt.figure(figsize=(11, 4))
plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, forecasts[best].values, label=f"Predicted ({best})")
plt.legend(); plt.title("Actual vs predicted, test period"); plt.show()

err = pd.DataFrame({"actual": test.values, "pred": forecasts[best].values}, index=test.index)
err["error"] = err["actual"] - err["pred"]
err["abs_pct_err"] = err["error"].abs() / err["actual"] * 100
print("\nMean absolute % error by weekday:")
print(err.groupby(err.index.day_name())["abs_pct_err"].mean().round(1).sort_values())
print("\nLargest 5 errors (visitors):")
print(err["error"].abs().nlargest(5).round(0))

# =====================================================================
# 6. FORECAST NEXT 90 DAYS   (PDF Section 5)
# =====================================================================
best_fc = best
FH = 90
print(f"\nRefitting {best_fc} on all {len(y)} days and forecasting {FH} days ...")
future_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=FH, freq="D")
pred_f = np.asarray(FUNCS[best_fc](y, FH), float)
future = pd.DataFrame({"date": future_idx, "predicted_unique_visits": pred_f.round(0), "model": best_fc})
future["horizon"] = np.where(np.arange(FH) < 7, "short-term (7 days)", "long-term (90 days)")
show_cols = ["date", "predicted_unique_visits"]
print(f"Model: {best_fc} | forecast {future['date'].min().date()} to {future['date'].max().date()}")
print("\nShort-term (next 7 days):"); print(future.head(7)[show_cols].to_string(index=False))
print("\nExpected peak days:"); print(future.nlargest(5, "predicted_unique_visits")[show_cols].to_string(index=False))
print("\nExpected low days:"); print(future.nsmallest(5, "predicted_unique_visits")[show_cols].to_string(index=False))

plt.figure(figsize=(11, 4))
plt.plot(y.index[-180:], y.values[-180:], label="History (last 180 days)")
plt.plot(future["date"], future["predicted_unique_visits"], label=f"Forecast ({best_fc})")
plt.legend(); plt.title("Historical and predicted unique visits"); plt.show()

# =====================================================================
# 7. STORE IN SQLITE   (PDF Section 8)
# =====================================================================
con = sqlite3.connect("traffic.db")
hist = df.reset_index(); hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
hist.to_sql("traffic_history", con, if_exists="replace", index=False)
fut = future.copy(); fut["date"] = fut["date"].dt.strftime("%Y-%m-%d")
fut.to_sql("forecasts", con, if_exists="replace", index=False)
res.reset_index().to_sql("model_metrics", con, if_exists="replace", index=False)
bt = forecasts.reset_index(); bt["date"] = bt["date"].dt.strftime("%Y-%m-%d")
bt.to_sql("backtest_predictions", con, if_exists="replace", index=False)
print()
for tname in ["traffic_history", "forecasts", "model_metrics", "backtest_predictions"]:
    print(tname, "rows:", con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0])
con.close()
res.round(2).to_csv("model_metrics.csv")
forecasts.to_csv("forecasts.csv")
future.to_csv("future_forecast.csv", index=False)

# =====================================================================
# 8. PDF REPORT   (PDF Section 8: downloadable report)
# =====================================================================
PAGE = (8.27, 11.69)


def text_page(pdf, title, lines, table=None, table_title=None):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.95, title, fontsize=16, weight="bold", va="top")
    y0 = 0.90
    for ln in lines:
        for w in (textwrap.wrap(ln, 95) or [""]):
            fig.text(0.07, y0, w, fontsize=9.5, va="top")
            y0 -= 0.022
        y0 -= 0.006
    if table is not None:
        ax = fig.add_axes([0.07, 0.08, 0.86, max(0.05, y0 - 0.12)])
        ax.axis("off")
        if table_title:
            ax.set_title(table_title, fontsize=10, weight="bold", loc="left")
        tb = ax.table(cellText=table.values, colLabels=list(table.columns), loc="upper center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.4)
    pdf.savefig(fig); plt.close(fig)


n_days = len(df)
expected = (df.index.max() - df.index.min()).days + 1
missing_vals = int(df[COUNTS].isna().sum().sum())
dups = int(df.index.duplicated().sum())
outl = int(df["is_outlier"].sum())
sm = res.round(2).reset_index()
sm.columns = ["Model", "MAE", "MSE", "RMSE", "MAPE %"]
sm = sm.astype(str)

with PdfPages("traffic_analysis_report.pdf") as pdf:
    text_page(pdf, "Website Traffic Forecasting - Analysis Report", [
        "Objective: analyse historical daily website traffic and forecast future visitor levels using ARIMA, SARIMA, Prophet and LSTM.",
        f"Data: {n_days} daily records, {df.index.min().date()} to {df.index.max().date()} (expected {expected} days).",
        f"Validation: {missing_vals} missing values, {dups} duplicate dates, {outl} outliers flagged.",
        f"Train / test split: first {len(train)} days for training, last {len(test)} days ({test.index[0].date()} to {test.index[-1].date()}) for testing.",
        f"Best model by RMSE on the test days: {best}.",
        "Metrics are measured in unique visitors per day (MAE, RMSE, MSE) and percent (MAPE).",
    ], table=sm, table_title="Model comparison (sorted by RMSE)")

    fig, axes = plt.subplots(3, 1, figsize=PAGE)
    axes[0].plot(df.index, df["unique_visits"]); axes[0].set_title("Daily unique visits")
    axes[1].bar(ORDER, wk_all.values); axes[1].set_title("Average unique visits by weekday"); axes[1].tick_params(axis="x", rotation=30)
    axes[2].bar(mo_all.index, mo_all.values); axes[2].set_title("Average unique visits by month (1 = Jan)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=PAGE)
    axes[0].plot(test.index, test.values, label="Actual")
    axes[0].plot(test.index, forecasts[best].values, label=f"Predicted ({best})")
    axes[0].legend(); axes[0].set_title("Actual vs predicted, test period"); axes[0].tick_params(axis="x", rotation=30)
    wk_err = err.groupby(err.index.day_name())["abs_pct_err"].mean().reindex(ORDER)
    axes[1].bar(ORDER, wk_err.values); axes[1].set_title(f"Mean absolute % error by weekday ({best})"); axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    top = future.nlargest(5, "predicted_unique_visits")[show_cols].copy()
    low = future.nsmallest(5, "predicted_unique_visits")[show_cols].copy()
    top["date"] = top["date"].dt.strftime("%Y-%m-%d"); low["date"] = low["date"].dt.strftime("%Y-%m-%d")
    pk = pd.concat([top.reset_index(drop=True), low.reset_index(drop=True)], axis=1)
    pk.columns = ["Peak date", "Peak visits", "Low date", "Low visits"]
    pk = pk.astype(str)
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.1, 0.68, 0.85, 0.25])
    ax.plot(y.index[-180:], y.values[-180:], label="History (last 180 days)")
    ax.plot(future["date"], future["predicted_unique_visits"], label=f"Forecast ({best_fc})")
    ax.legend(); ax.set_title(f"Forecast {future['date'].min().date()} to {future['date'].max().date()}"); ax.tick_params(axis="x", rotation=30)
    ax2 = fig.add_axes([0.07, 0.40, 0.86, 0.18]); ax2.axis("off")
    ax2.set_title("Expected peak and low days in the forecast", fontsize=10, weight="bold", loc="left")
    tb = ax2.table(cellText=pk.values, colLabels=list(pk.columns), loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.3)
    lim = ["Limitations:",
           "- Data is daily only; hourly forecasting is not possible with this dataset.",
           "- Sessions are not recorded; unique visits is used as the closest proxy.",
           "- The data covers a single website.",
           "- Models were compared on one 90-day test window.",
           "- Forecast dates follow the end of the dataset, so they cannot be checked against real data."]
    yy = 0.32
    for ln in lim:
        fig.text(0.07, yy, ln, fontsize=9.5, weight="bold" if ln.endswith(":") else None)
        yy -= 0.025
    pdf.savefig(fig); plt.close(fig)
print("\nSaved traffic_analysis_report.pdf")

# =====================================================================
# 9. PACK AND DOWNLOAD
# =====================================================================
outs = ["traffic_clean.csv", "traffic_weekly.csv", "traffic_monthly.csv", "model_metrics.csv",
        "forecasts.csv", "future_forecast.csv", "traffic.db", "traffic_analysis_report.pdf"]
with zipfile.ZipFile("outputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in outs:
        z.write(f)
print("Created outputs.zip with:", ", ".join(outs))
try:
    from google.colab import files
    files.download("outputs.zip")
    files.download("traffic_analysis_report.pdf")
except ImportError:
    print("(not in Colab: skipped download)")
print("\nALL DONE")
