
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# -------------------------------------------------
# 1. PATHS AND MODEL LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "data.csv"
MODEL_PATH = BASE_DIR / "models" / "crypto_volatility_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "crypto_volatility_metadata.joblib"

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

FEATURE_COLUMNS = metadata["feature_columns"]


# --------------------------------------------------
# 2. FASTAPI CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="Crypto Volatility Prediction API",
    description="Predicts future 7-day cryptocurrency volatility using XGBoost.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# 3. LOAD AND CLEAN DATA
# --------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)

    rename_map = {}

    for col in df_raw.columns:
        key = str(col).strip().lower().replace(" ", "").replace("_", "")

        mapping = {
            "date": "date",
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "marketcap": "marketCap",
            "cryptoname": "crypto_name",
            "symbol": "crypto_name",
            "unnamed:0": "drop_index"
        }

        if key in mapping:
            rename_map[col] = mapping[key]

    df = df_raw.rename(columns=rename_map).copy()

    if "drop_index" in df.columns:
        df = df.drop(columns=["drop_index"])

    if "date" in df.columns:
        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce", utc=True)
            .dt.tz_localize(None)
        )

    elif "timestamp" in df.columns:
        ts = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True
        )

        if ts.isna().mean() > 0.5:
            numeric_ts = pd.to_numeric(
                df["timestamp"], errors="coerce"
            )

            unit = (
                "ms"
                if numeric_ts.dropna().median() > 1e11
                else "s"
            )

            ts = pd.to_datetime(
                numeric_ts, unit=unit, errors="coerce", utc=True
            )

        df["date"] = ts.dt.tz_localize(None)

    else:
        raise ValueError("A date or timestamp column is required.")

    required = ["open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "crypto_name" not in df.columns:
        df["crypto_name"] = "UNKNOWN"

    numeric_cols = ["open", "high", "low", "close", "volume"]

    if "marketCap" in df.columns:
        numeric_cols.append("marketCap")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["crypto_name"] = (
        df["crypto_name"].astype(str).str.upper().str.strip()
    )

    df = df.dropna(
        subset=["date", "close", "open", "high", "low"]
    )

    df = df[df["close"] > 0].copy()

    df = (
        df.sort_values(["crypto_name", "date"])
        .drop_duplicates(
            subset=["crypto_name", "date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df


# --------------------------------------------------
# 4. FEATURE ENGINEERING
# Reproduces the notebook's feature calculations.
# --------------------------------------------------

def add_features(group):

    group = group.sort_values("date").copy()

    close = group["close"]
    high = group["high"]
    low = group["low"]
    volume = group["volume"]

    # Returns
    group["log_return"] = np.log(close / close.shift(1))
    group["daily_return"] = close.pct_change()

    # Rolling return statistics
    for window in [3, 7, 14, 30]:
        group[f"return_mean_{window}"] = (
            group["log_return"].rolling(window).mean()
        )
        group[f"return_vol_{window}"] = (
            group["log_return"].rolling(window).std()
        )

    # Moving averages
    for window in [7, 14, 30]:
        group[f"sma_{window}"] = close.rolling(window).mean()
        group[f"close_sma_ratio_{window}"] = (
            close / group[f"sma_{window}"]
        )

    # MACD
    group["ema_12"] = close.ewm(
        span=12, adjust=False
    ).mean()

    group["ema_26"] = close.ewm(
        span=26, adjust=False
    ).mean()

    group["macd"] = group["ema_12"] - group["ema_26"]

    group["macd_signal"] = group["macd"].ewm(
        span=9, adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    group["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()

    group["bb_width_20"] = (
        (4 * rolling_std) / rolling_mean.replace(0, np.nan)
    )

    group["bb_position_20"] = (
        (close - (rolling_mean - 2 * rolling_std))
        / (4 * rolling_std).replace(0, np.nan)
    )

    # ATR
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    group["atr_14"] = true_range.rolling(14).mean()
    group["atr_pct"] = group["atr_14"] / close

    # Volume features
    group["volume_change"] = (
        volume.pct_change().replace([np.inf, -np.inf], np.nan)
    )

    group["volume_mean_7"] = volume.rolling(7).mean()

    group["volume_ratio_7"] = (
        volume / group["volume_mean_7"].replace(0, np.nan)
    )

    # Lagged returns
    for lag in [1, 2, 3, 7]:
        group[f"return_lag_{lag}"] = (
            group["log_return"].shift(lag)
        )
        # Target: standard deviation of the next 7 daily log returns
    future_returns = group["log_return"].shift(-1)

    group["target_volatility_7d"] = (
        future_returns.rolling(7).std().shift(-6)
    )
    return group


# --------------------------------------------------
# 5. GENERATE LATEST FEATURES
# --------------------------------------------------

def get_latest_features():

    df = load_data()

    featured_data = pd.concat(
        [
            add_features(group)
            for _, group in df.groupby("crypto_name")
        ],
        ignore_index=True
    )

    featured_data = featured_data.replace(
        [np.inf, -np.inf], np.nan
    )

    latest_data = (
        featured_data.sort_values("date")
        .groupby("crypto_name")
        .tail(1)
        .copy()
    )

    return latest_data


# --------------------------------------------------
# 6. API ROUTES
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Crypto Volatility Prediction API is running",
        "model": metadata["selected_model"],
        "forecast_horizon_days": metadata["forecast_horizon_days"]
    }


@app.get("/assets")
def get_assets():

    try:
        df = load_data()

        return {
            "assets": sorted(df["crypto_name"].unique().tolist())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_model():
    global model, metadata, FEATURE_COLUMNS

    try:
        # 1. Load and engineer training data
        df = load_data()

        data = (
            df.groupby("crypto_name", group_keys=False)
            .apply(add_features)
            .replace([np.inf, -np.inf], np.nan)
            .reset_index(drop=True)
        )

        data = data.dropna(
            subset=["target_volatility_7d"]
        ).copy()

        # 2. Select features
        excluded = {
            "date", "timestamp", "target_volatility_7d",
            "open", "high", "low", "close",
            "volume", "marketCap", "ema_12", "ema_26"
        }

        feature_cols = [
            c for c in data.columns if c not in excluded
        ]

        if "crypto_name" not in feature_cols:
            feature_cols.append("crypto_name")

        X = data[feature_cols].copy()
        y = data["target_volatility_7d"].copy()

        # 3. Chronological split by unique dates
        unique_dates = np.array(
            sorted(data["date"].dropna().unique())
        )

        n_dates = len(unique_dates)

        if n_dates < 30:
            raise ValueError("Not enough unique dates for training.")

        train_end = unique_dates[int(n_dates * 0.70) - 1]
        valid_end = unique_dates[int(n_dates * 0.85) - 1]

        train_mask = data["date"] <= train_end

        valid_mask = (
            (data["date"] > train_end) &
            (data["date"] <= valid_end)
        )

        test_mask = data["date"] > valid_end

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        # 4. Build preprocessing pipeline
        numeric_features = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        def make_preprocessor(scale_numeric=False):
            numeric_steps = [
                ("imputer", SimpleImputer(strategy="median"))
            ]

            if scale_numeric:
                numeric_steps.append(
                    ("scaler", StandardScaler())
                )

            transformers = [
                (
                    "numeric",
                    Pipeline(numeric_steps),
                    numeric_features
                )
            ]

            if categorical_features:
                transformers.append(
                    (
                        "categorical",
                        Pipeline([
                            ("imputer", SimpleImputer(
                                strategy="most_frequent"
                            )),
                            ("onehot", OneHotEncoder(
                                handle_unknown="ignore"
                            ))
                        ]),
                        categorical_features
                    )
                )

            return ColumnTransformer(
                transformers,
                remainder="drop"
            )

        # 5. Define candidate models
        models = {
            "Linear Regression": Pipeline([
                ("preprocessor", make_preprocessor(True)),
                ("model", LinearRegression())
            ]),

            "Random Forest": Pipeline([
                ("preprocessor", make_preprocessor()),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),

            "Extra Trees": Pipeline([
                ("preprocessor", make_preprocessor()),
                ("model", ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),

            "XGBoost": Pipeline([
                ("preprocessor", make_preprocessor()),
                ("model", XGBRegressor(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        }

        # 6. Train and select using validation RMSE
        fitted_models = {}
        validation_results = []

        for name, pipeline in models.items():
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_valid)

            rmse = float(np.sqrt(
                mean_squared_error(y_valid, predictions)
            ))

            mae = float(mean_absolute_error(
                y_valid, predictions
            ))

            validation_results.append({
                "model": name,
                "MAE": mae,
                "RMSE": rmse
            })

            fitted_models[name] = pipeline

        best_result = min(
            validation_results,
            key=lambda item: item["RMSE"]
        )

        best_name = best_result["model"]
        best_model = fitted_models[best_name]

        # 7. Evaluate on untouched test data
        test_predictions = best_model.predict(X_test)

        test_metrics = {
            "MAE": float(mean_absolute_error(
                y_test, test_predictions
            )),
            "RMSE": float(np.sqrt(
                mean_squared_error(y_test, test_predictions)
            )),
            "R2": float(r2_score(
                y_test, test_predictions
            ))
        }

        # 8. Save model and metadata
        new_metadata = {
            "selected_model": best_name,
            "target": "target_volatility_7d",
            "forecast_horizon_days": 7,
            "feature_columns": feature_cols,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "test_metrics": test_metrics,
            "validation_results": validation_results,
            "train_end_date": str(pd.Timestamp(train_end).date()),
            "validation_end_date": str(pd.Timestamp(valid_end).date())
        }

        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(new_metadata, METADATA_PATH)

        # 9. Update currently running API model
        model = best_model
        metadata = new_metadata
        FEATURE_COLUMNS = feature_cols

        return {
            "message": "Model training completed successfully",
            "selected_model": best_name,
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(X_valid)),
            "test_samples": int(len(X_test)),
            "test_metrics": test_metrics,
            "validation_results": validation_results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@app.get("/predict")
def predict(asset: str):

    try:
        latest_data = get_latest_features()

        asset = asset.strip().upper()

        asset_data = latest_data[
            latest_data["crypto_name"] == asset
        ]

        if asset_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Asset {asset} not found in dataset."
            )

        # Select exactly the training features, in training order.
        X_live = asset_data[FEATURE_COLUMNS]

        prediction = float(model.predict(X_live)[0])

        row = asset_data.iloc[0]

        return {
            "asset": asset,
            "date": str(row["date"].date()),
            "close": float(row["close"]),
            "predicted_volatility_7d": prediction,
            "predicted_volatility_percent": prediction * 100,
            "forecast_horizon_days": 7,
            "model": metadata["selected_model"],
            "target": metadata["target"]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
