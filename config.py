#Load artifacts once at import time
MODEL_PATH           ="models/disaster_model.pkl"
SCALER_PATH          ="models/scaler.pkl"
FEATURE_COLUMNS_PATH ="models/feature_columns.pkl"
DATA_PATH ="data/country_year_features.csv"

#Columns scaled during training (must match training pipeline exactly)
SCALE_COLS = [
    "Start_Year",
    "prev_year_count",
    "log_prev_total_deaths",
    "log_prev_total_affected",
    "log_prev_total_damage",
]

#Raw impact columns present in country_year_features.csv that feed into lags
RAW_IMPACT_COLS = {
    "log_prev_total_deaths"   : "total_deaths",
    "log_prev_total_affected" : "total_affected",
    "log_prev_total_damage"   : "total_damage",
}
