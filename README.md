# Dengue Ensemble Forecasting – Sri Lanka

CM4371 Machine Learning & Pattern Recognition – Ensemble Learning Hands-on Assignment

**Author:** Kalansooriya S.I - 224101N

Forecasts **next week's dengue cases for 26 Sri Lankan health districts** and compares bagging, boosting, stacking and a hybrid model in a Streamlit dashboard.

- **Live dashboard:** [https://dengue-forecast-sl.streamlit.app/](https://dengue-forecast-sl.streamlit.app/)

## Data
- **Dengue cases:** weekly cases by district (Dec 2006 – Aug 2026), Epidemiology Unit, Ministry of Health Sri Lanka, via [denguedatahub](https://denguedatahub.netlify.app/srilanka)
- **Weather:** rainfall, temperature and humidity from the [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Cleaning:** fixed mixed date formats and repaired 7 corrupted weeks (scrambled or duplicated)

## Results (test period 2022–2026)

| Model | Approach | RMSE | R² |
|---|---|---|---|
| Stacking (RF + XGBoost + LightGBM + CatBoost) | Stacking | 29.50 | **0.908** |
| LightGBM | Boosting | 30.27 | 0.903 |
| Hybrid RF + ARIMA | Hybrid | 31.79 | 0.893 |
| Random Forest | Bagging | 32.36 | 0.889 |
| Same as last week | Baseline | 32.50 | 0.888 |
| Decision Tree | Single model | 35.78 | 0.864 |

**Stacking performed best**, explaining 90.8% of the variation in weekly cases. **LightGBM** is used for the dashboard forecast because it is almost as accurate and trains in seconds instead of minutes.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `Dengue_Ensemble_Forecasting.ipynb` – data preparation, model training, comparison and interpretation
- `app.py` – Streamlit dashboard
- `models/` – trained LightGBM model
- `dashboard_data/` – results and data used by the dashboard

## Use of Generative AI
Generative AI (Claude, by Anthropic) was used to help write parts of the code, including data loading, model training, and the Streamlit dashboard, and to help draft parts of the written interpretation. All code was reviewed, run, tested and modified by the author, and the final analysis and design decisions are the author's own.
