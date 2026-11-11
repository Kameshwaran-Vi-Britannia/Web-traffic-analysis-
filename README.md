# 🚀 AI-Based Website Traffic Forecasting

### AI-powered time-series forecasting system for analysing historical website traffic, comparing forecasting models, and predicting future visitor activity through an interactive dashboard.

---

## 📌 Project Abstract

Website traffic analysis helps understand visitor behaviour, identify traffic patterns, and estimate future website activity.

This project implements an AI-based website traffic forecasting system using historical daily visitor data. It applies multiple time-series forecasting techniques, compares their predictive performance, and selects the best-performing model based on evaluation metrics.

The system integrates data preprocessing, exploratory data analysis, forecasting, model evaluation, database management, and visualization into a single workflow.

The final results are presented through an interactive Streamlit dashboard, supported by an SQLite database and an automatically generated PDF analysis report.

---

# 🎯 Project Objectives

✅ Analyse historical website traffic data

✅ Identify daily, weekly, and monthly traffic patterns

✅ Implement and compare multiple forecasting models

✅ Evaluate forecasting accuracy using standard performance metrics

✅ Predict website traffic for the next 90 days

✅ Store results in a structured database

✅ Present analysis and predictions through an interactive dashboard

---

# ✨ Key Features

- Historical website traffic analysis
- Data cleaning and validation
- Exploratory Data Analysis (EDA)
- Five forecasting approaches
- Model performance comparison
- 90-day future traffic forecasting
- SQLite database integration
- Interactive Streamlit dashboard
- Automated PDF report generation
- Google Colab support
- One-step application launcher with a temporary public access link

---

# 🛠 Technologies and Requirements

## Technology Stack

| Technology | Application |
|------------|------------|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Graphs and visualizations |
| Scikit-learn | Data preprocessing and scaling |
| Statsmodels | ARIMA and SARIMA forecasting |
| Prophet | Time-series forecasting |
| TensorFlow / Keras | LSTM neural network |
| Streamlit | Interactive web dashboard |
| SQLite | Database storage |
| Cloudflare Tunnel | Temporary public access to the Colab dashboard |

---

## Software Requirements

- Python environment
- Google Colab or local Python installation
- Internet connection for installing dependencies and downloading Cloudflare Tunnel
- Original website traffic dataset for rebuilding the pipeline

> **Note**
>
> No dedicated GPU is required by the project instructions. Model training time depends on the available environment.

---

# 📈 Forecasting Models

The project implements five forecasting approaches.

| Model | Purpose |
|---------|---------|
| Seasonal Naive | Uses previous weekly traffic patterns as a baseline |
| ARIMA | Models trends and time-series dependencies |
| SARIMA | Models trends and seasonal patterns |
| Prophet | Handles time-series trends and seasonality |
| LSTM | Uses a neural network to learn sequential patterns |

---

## 📊 Model Evaluation Metrics

The models are compared using:

- MAE — Mean Absolute Error
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- MAPE — Mean Absolute Percentage Error

Lower values indicate better predictive performance for these error metrics.

---

# 🏗 Project Architecture and File Structure

```text
Web-traffic-analysis-/
│
├── README.md
│
├── app.py
├── launch.py
├── run_all.py
│
├── daily-website-visitors.csv
│
├── traffic.db
│
├── traffic_analysis_report.pdf
│
├── requirements.txt
└── requirements-pipeline.txt
```

The forecasting pipeline also creates additional files and directories when executed.

---

# 📂 Detailed File Description

---

## 1. README.md

### Purpose
Project documentation.

### Contains

- Project overview and objectives
- Technologies and dependencies
- File descriptions
- Installation instructions
- Google Colab execution instructions
- Local execution instructions
- Database and output information

---

## 2. app.py

### Purpose

Main Streamlit dashboard.

This file reads the existing SQLite database and presents the analysis through an interactive interface.

### Dashboard Components

- Traffic statistics and summary metrics
- Historical visitor trends
- Historical traffic and future forecast visualization
- Actual versus predicted traffic
- Date-range filtering
- Daily, weekly, and monthly views
- Busiest and quietest days
- Forecasted peak and low traffic days
- Model performance comparison
- CSV forecast download
- PDF report download

> **Important**
>
> `app.py` does not train the forecasting models.
>
> It reads the previously generated results from `traffic.db`.

---

## 3. launch.py

### Purpose

Automated Google Colab dashboard launcher.

This script simplifies the process of starting the application.

### Operations

1. Checks whether `app.py` exists
2. Checks whether `traffic.db` exists
3. Validates the required database tables
4. Installs Streamlit if it is missing
5. Downloads the Cloudflare Tunnel executable if necessary
6. Starts the Streamlit application
7. Checks whether Streamlit is running correctly
8. Creates a temporary Cloudflare public tunnel
9. Displays the generated dashboard URL
10. Keeps the application and tunnel running

> **Important**
>
> This file launches the existing dashboard.
>
> It does not rebuild the forecasting models or database.

---

## 4. run_all.py

### Purpose

Complete data analysis and forecasting pipeline.

This is the main processing script responsible for generating the project's results.

### Major Processing Stages

| Stage | Operation |
|---------|---------|
| 1 | Loads the original dataset |
| 2 | Cleans and validates the data |
| 3 | Performs exploratory data analysis |
| 4 | Trains five forecasting models |
| 5 | Evaluates and compares model performance |
| 6 | Generates a 90-day forecast |
| 7 | Stores results in SQLite |
| 8 | Generates a PDF analysis report |
| 9 | Packages the generated files into outputs.zip |

> **Important**
>
> Running this script rebuilds the database and generated reports.
>
> It is not necessary every time you open the dashboard.

---

## 5. daily-website-visitors.csv

### Purpose

Original dataset.

Contains historical daily website traffic information.

### Dataset Columns

- Row
- Day
- Day of Week
- Date
- Page Loads
- Unique Visits
- First-Time Visits
- Returning Visits

The pipeline uses this dataset to perform preprocessing, analysis, model training, and forecasting.

---

## 6. traffic.db

### Purpose

Main SQLite database.

This database stores the processed information required by the Streamlit dashboard.

### Database Tables

| Table | Purpose |
|---------|---------|
| traffic_history | Stores cleaned historical traffic records |
| forecasts | Stores future traffic predictions |
| model_metrics | Stores model evaluation results |
| backtest_predictions | Stores actual and predicted values from the test period |

### Database Workflow

```text
Original Dataset
       |
       v
   run_all.py
       |
       v
   Data Processing
       |
       v
 Model Training
       |
       v
 Model Evaluation
       |
       v
  Forecasting
       |
       v
    traffic.db
       |
       v
     app.py
       |
       v
 Streamlit Dashboard
```

The database is already generated and included in the repository.

Therefore, you can run the existing dashboard without retraining the models.

---

## 7. traffic_analysis_report.pdf

### Purpose

Automatically generated analysis report.

### Report Contents

- Project objective and dataset information
- Data validation results
- Model performance comparison
- Historical traffic visualization
- Average traffic by weekday and month
- Actual versus predicted traffic
- Forecast visualization
- Expected peak and low traffic days
- Project limitations

The report is generated by `run_all.py` and can also be downloaded from the dashboard when present.

---

## 8. requirements.txt

### Purpose

Dependencies for running the existing dashboard.

### Current Packages

```txt
streamlit
pandas
```

These are sufficient for the dashboard's direct Python imports.

SQLite, datetime and operating-system utilities used by the application are included in Python's standard library.

---

## 9. requirements-pipeline.txt

### Purpose

Dependencies for rebuilding the complete forecasting pipeline.

### Current Packages

```txt
pandas
numpy
matplotlib
scikit-learn
statsmodels
prophet
tensorflow
streamlit
```

These packages support data processing, visualization, model training, forecasting, evaluation, and dashboard execution.

### Difference Between Both Requirements Files

- `requirements.txt` → Used when you only want to run the existing dashboard.
- `requirements-pipeline.txt` → Used when you want to execute the complete data processing and model training workflow.

---

# 🚀 Installation and Execution

There are two main ways to use this project:

1. Run the already-generated dashboard
2. Rebuild the complete forecasting pipeline

---

# Method A: Run the Dashboard in Google Colab

> ⭐ Recommended for dashboard usage

This method does not require a personal laptop.

---

### Step 1: Open Google Colab

Visit:

https://colab.research.google.com/

Create a new notebook.

---

### Step 2: Clone the GitHub Repository

```python
!git clone https://github.com/Kameshwaran-Vi-Britannia/Web-traffic-analysis-.git
```

### Step 3: Enter the Project Directory

```python
%cd Web-traffic-analysis-
```

### Step 4: Install Dashboard Requirements

```python
!pip install -r requirements.txt
```

### Step 5: Launch the Application

```python
!python -u launch.py
```

### Step 6: Open the Dashboard

- Wait for the launcher to complete its checks
- Copy the generated Cloudflare URL
- Open the URL in your browser
- Keep the Colab execution cell running while using the dashboard

The launcher automatically handles Streamlit startup and tunnel creation.

> **Note**
>
> The public tunnel URL is temporary.
>
> It is not permanent hosting.
>
> Anyone with the URL may be able to access the dashboard while the tunnel is active.

---

# Method B: Run the Dashboard Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kameshwaran-Vi-Britannia/Web-traffic-analysis-.git
```

### Step 2: Navigate to Project

```bash
cd Web-traffic-analysis-
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start Streamlit

```bash
streamlit run app.py
```

### Step 5: Access Dashboard

Open:

```text
http://localhost:8501
```

The existing `traffic.db` must remain in the same directory as `app.py`.

---

# 🔄 Rebuilding the Complete Forecasting Pipeline

This process is optional when the generated database and report are already available.

Use it when you want to repeat the analysis, train the models again, or generate updated outputs from the original dataset.

---

### Step 1: Install Pipeline Dependencies

```python
!pip install -r requirements-pipeline.txt
```

### Step 2: Prepare Dataset Directory

```python
!mkdir -p data/raw
!cp daily-website-visitors.csv data/raw/
```

This step is important.

Without it, the pipeline will not automatically detect the CSV currently stored in the repository root.

### Step 3: Execute Complete Pipeline

```python
%run run_all.py
```

or

```python
!python -u run_all.py
```

### Step 4: Wait for Processing

The script will:

1. Load the dataset
2. Clean and validate records
3. Generate exploratory analysis
4. Train and evaluate the five models
5. Select the best-performing model based on RMSE
6. Generate a 90-day forecast
7. Create the SQLite database
8. Generate the PDF report
9. Create `outputs.zip`

> **Important Pipeline Information**
>
> - The original dataset must be available
> - Model training can take considerably longer than dashboard startup
> - `traffic.db` and the PDF report are regenerated
> - Generated results may differ slightly depending on the environment and model execution
> - Keep backups of existing generated files before rebuilding

If the CSV is not available, the script can also extract a ZIP dataset from the project root.

Its optional KaggleHub fallback requires the additional `kagglehub` package and internet access.

---

# 📁 Generated Files and Their Purpose

| Generated File | Purpose |
|---------------|----------|
| traffic_clean.csv | Cleaned daily traffic data |
| traffic_weekly.csv | Weekly traffic aggregation |
| traffic_monthly.csv | Monthly traffic aggregation |
| model_metrics.csv | Performance metrics for forecasting models |
| forecasts.csv | Actual and predicted values for the test period |
| future_forecast.csv | Predicted traffic for the next 90 days |
| traffic.db | Database containing dashboard information |
| traffic_analysis_report.pdf | Complete analysis report |
| outputs.zip | Compressed collection of generated outputs |

These files are generated by the pipeline.

They do not all need to be uploaded individually for the existing dashboard to work.

---

# 🏆 Model Performance Results

Dataset: Daily Website Visitors

Training period: First 2,077 days

Testing period: Final 90 days

| Model | MAE | MSE | RMSE | MAPE |
|---------|---------|---------|---------|---------|
| SARIMA | 354.44 | 205761.30 | 453.61 | 13.37% |
| Prophet | 446.42 | 279571.84 | 528.75 | 17.34% |
| Seasonal Naive | 945.44 | 1100619.13 | 1049.10 | 36.93% |
| LSTM | 940.72 | 1114307.01 | 1055.61 | 32.71% |
| ARIMA | 1218.88 | 1846973.96 | 1359.03 | 50.15% |

### ✅ Best Performing Model

**SARIMA**

SARIMA achieved the lowest RMSE among the evaluated models on the selected test period.

---

# 📊 Dataset Information

Dataset Name: Daily Website Visitors

Source: Kaggle

Dataset Author: bobnau

Dataset Link:

https://www.kaggle.com/datasets/bobnau/daily-website-visitors

| Property | Information |
|------------|------------|
| Total Records | 2,167 |
| Start Date | 14 September 2014 |
| End Date | 19 August 2020 |
| Frequency | Daily |
| Forecast Horizon | 90 Days |
| Target Variable | Unique Visits |

---

# 🛠 Troubleshooting

| Problem | Possible Solution |
|----------|----------|
| app.py not found | Enter the correct project directory |
| traffic.db not found | Upload or restore the generated database |
| Missing database tables | Rebuild or restore the correct database |
| Streamlit not installed | Install requirements.txt |
| Cloudflare link not generated | Check internet connectivity and rerun the launcher |
| Dataset not found | Place the CSV inside data/raw/ |
| Model training fails | Check pipeline dependencies and runtime resources |
| Dashboard stops working | Check whether the Colab execution cell is still running |

---

# ⚠ Project Limitations

- The dataset contains daily records, so hourly forecasting is not supported.
- The dataset covers a single website.
- Unique visits are used as the primary forecasting target.
- Model performance is evaluated using one 90-day test period.
- Future predictions extend beyond the available historical dataset and cannot be verified against actual observations in the supplied dataset.
- Google Colab sessions and temporary Cloudflare links are not permanent hosting solutions.

---

# 🚀 Future Enhancements

- Integration of real-time website traffic data
- Support for multiple website datasets
- Additional forecasting models
- Automated model retraining
- Improved forecasting accuracy through parameter optimization
- Persistent cloud deployment
- Enhanced dashboard analytics and visualization

---

# ℹ Project Information

| Category | Details |
|-----------|-----------|
| Project Name | AI-Based Website Traffic Forecasting |
| Project Type | Individual Project |
| Domain | Data Science and Machine Learning |
| Programming Language | Python |
| Framework | Streamlit |
| Database | SQLite |
| Development Environment | Google Colab |
| Forecasting Approach | Time-Series Forecasting |

---

# 🙏 Acknowledgements

- Kaggle for the Daily Website Visitors dataset
- Open-source Python libraries and frameworks used in the implementation

---

## End of Documentation
