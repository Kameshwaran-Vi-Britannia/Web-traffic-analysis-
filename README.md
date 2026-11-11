# AI-Based Website Traffic Forecasting

Forecasts daily website traffic and shows it in a Streamlit dashboard.
Built during my internship. Dataset: Kaggle "Daily Website Visitors" (bobnau/daily-website-visitors).

- Models compared: Seasonal naive, ARIMA, SARIMA, Prophet, LSTM
- Best model: SARIMA (RMSE ~454, MAPE ~13.4%)
- Output: 90-day forecast, SQLite database, PDF report, dashboard

## Run the dashboard in Google Colab (no laptop needed)
1. Upload `website-traffic-forecasting.zip` to Colab (Files icon on the left).
2. Run these cells:
```
!unzip -o website-traffic-forecasting.zip
%cd website-traffic-forecasting
!pip install -q -r requirements.txt
!python -u launch.py
```
3. Open the link it prints. Keep the cell running.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Re-run the full pipeline (optional, slow)
```
pip install -r requirements-pipeline.txt
python run_all.py
```
Rebuilds `traffic.db` and `traffic_analysis_report.pdf` from `data/raw/`.

## Files
- `app.py` dashboard
- `launch.py` Colab launcher (starts the dashboard and a public link)
- `run_all.py` full pipeline
- `traffic.db` stored results the dashboard reads
- `traffic_analysis_report.pdf` report
- `data/raw/` source dataset
