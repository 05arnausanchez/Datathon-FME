import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "train"

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "load_kw_lag1", "load_kw_lag24", "load_kw_roll24",
    "pv_kw_lag1", "pv_kw_lag24", "pv_kw_roll24",
    "price_lag1", "price_lag24", "price_roll24",
    "co2_lag1", "co2_lag24", "co2_roll24",
    "irradiancia_lag1", "irradiancia_roll24",
    "temp_ambiente_lag1",
]

def cargar_datos(data_dir=DATA_DIR):
    # Consumo del campus
    df_load = pd.read_csv(data_dir / "contador.csv", usecols=["Fecha", "Contador - POTENCIA ACTIVA EXPORTADA (kW)"])
    df_load["Fecha"] = pd.to_datetime(df_load["Fecha"])
    df_load = df_load.set_index("Fecha")
    load_kw = (-df_load["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(0, 5000).resample("1h").mean().rename("load_kw")

    # Generacion solar
    inv_cols = [f"Inversor {n} - POTENCIA ACTIVA (kW)" for n in (1, 2, 3)]
    df_inv = pd.read_csv(data_dir / "inversor.csv", usecols=["Fecha", *inv_cols])
    df_inv["Fecha"] = pd.to_datetime(df_inv["Fecha"])
    df_inv = df_inv.set_index("Fecha")
    pv_kw = df_inv[inv_cols].sum(axis=1, min_count=len(inv_cols)).clip(lower=0)
    pv_hourly = pv_kw.resample("1h").mean().rename("pv_kw")

    # Precio y CO2
    df_price = pd.read_csv(data_dir / "precio_pvpc.csv", usecols=["Fecha", "precio_eur_per_kwh"])
    df_price["Fecha"] = pd.to_datetime(df_price["Fecha"])
    df_price = df_price.set_index("Fecha")

    df_co2 = pd.read_csv(data_dir / "co2_red.csv", usecols=["Fecha", "co2_kg_per_kwh"])
    df_co2["Fecha"] = pd.to_datetime(df_co2["Fecha"])
    df_co2 = df_co2.set_index("Fecha")

    # Meteorologia
    df_meteo = pd.read_csv(data_dir / "meteo.csv", usecols=["Fecha", "Atersa - IRRADIANCIA (W/m2)", "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)"])
    df_meteo["Fecha"] = pd.to_datetime(df_meteo["Fecha"])
    df_meteo = df_meteo.set_index("Fecha")
    irrad_hourly = df_meteo["Atersa - IRRADIANCIA (W/m2)"].resample("1h").mean().rename("irradiancia")
    temp_hourly = df_meteo["Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)"].resample("1h").mean().rename("temp_ambiente")

    # Unir todo
    df = pd.concat([load_kw, pv_kw, df_price["precio_eur_per_kwh"],
                    df_co2["co2_kg_per_kwh"],
                    irrad_hourly, temp_hourly], axis=1)
    df.columns = ["load_kw", "pv_kw", "price", "co2", "irradiancia", "temp_ambiente"]
    df = df.dropna().sort_index()

    # Features de calendario
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Features rezagadas y rolling
    for col in ["load_kw", "pv_kw", "price", "co2", "irradiancia", "temp_ambiente"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag24"] = df[col].shift(24)
        df[f"{col}_roll24"] = df[col].shift(1).rolling(24, min_periods=1).mean()

    df = df.dropna()
    return df

def entrenar_modelos(df, modelos_dir=MODELS_DIR):
    modelos_dir.mkdir(parents=True, exist_ok=True)
    X = df[FEATURE_COLS].values
    objetivos = {
        "model_price.pkl": "price",
        "model_co2.pkl": "co2",
        "model_pv.pkl": "pv_kw",
    }
    metricas = {}
    for nombre_archivo, col_obj in objetivos.items():
        y = df[col_obj].values
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        # Modelos compactos para cumplir el limite de 25 MB del submission.zip
        model = RandomForestRegressor(
            n_estimators=50, max_depth=12, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        joblib.dump(model, modelos_dir / nombre_archivo, compress=3)
        metricas[nombre_archivo] = {"mae": mae, "r2": r2, "n_train": len(X_train)}
        size_mb = (modelos_dir / nombre_archivo).stat().st_size / 1e6
        print(f"[OK] {nombre_archivo}: MAE={mae:.6f} R2={r2:.4f} ({size_mb:.1f} MB)")
    joblib.dump(FEATURE_COLS, modelos_dir / "feature_cols.pkl")
    print(f"[OK] Features guardadas: {len(FEATURE_COLS)} columnas")
    return metricas

def evaluar_modelos(df, modelos_dir=MODELS_DIR):
    print("\n=== Evaluacion de modelos sobre datos de train ===")
    feature_cols = joblib.load(modelos_dir / "feature_cols.pkl")
    X = df[feature_cols].values
    for nombre, col in [("model_price.pkl", "price"), ("model_co2.pkl", "co2"), ("model_pv.pkl", "pv_kw")]:
        model = joblib.load(modelos_dir / nombre)
        y_true = df[col].values
        y_pred = model.predict(X)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"  {nombre:16s} -> MAE={mae:.6f}  R2={r2:.4f}")

def main():
    print("Cargando datos de train...")
    df = cargar_datos()
    print(f"  Datos: {len(df)} filas, {df.index.min()} -> {df.index.max()}")
    print("\nEntrenando modelos predictivos (compactos, <25 MB total)...")
    metricas = entrenar_modelos(df)
    evaluar_modelos(df)
    total_mb = sum((MODELS_DIR / f).stat().st_size for f in MODELS_DIR.iterdir() if f.suffix == ".pkl") / 1e6
    print(f"\nTamaño total de modelos: {total_mb:.1f} MB")

if __name__ == "__main__":
    main()