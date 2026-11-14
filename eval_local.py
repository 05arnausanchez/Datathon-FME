"""eval_local.py
Evaluador local rápido sobre data/train/ (alineado con tutorial.ipynb).
"""
import time
import importlib.util
from pathlib import Path
import pandas as pd
from starter_kit.environment import run_agent_simulation


def carregar_agent(ruta_py: str):
    spec = importlib.util.spec_from_file_location("mod_agent", ruta_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BatteryAgent()


def construir_dataset_horari(data_dir: Path = Path("data/train")) -> pd.DataFrame:
    # 1. Contador (5 min -> 1h): load_kw = -Potencia Exportada (igual que tutorial.ipynb)
    df_cont = pd.read_csv(data_dir / "contador.csv")
    df_cont["Fecha"] = pd.to_datetime(df_cont["Fecha"])
    df_cont = df_cont.set_index("Fecha").sort_index()
    load_h = (-df_cont["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(lower=0.0, upper=5000.0).resample("1h").mean()

    # 2. Inversores solares (5 min -> 1h)
    df_inv = pd.read_csv(data_dir / "inversor.csv")
    df_inv["Fecha"] = pd.to_datetime(df_inv["Fecha"])
    df_inv = df_inv.set_index("Fecha").sort_index()
    cols_inv = [
        "Inversor 1 - POTENCIA ACTIVA (kW)",
        "Inversor 2 - POTENCIA ACTIVA (kW)",
        "Inversor 3 - POTENCIA ACTIVA (kW)",
    ]
    pv_h = df_inv[cols_inv].sum(axis=1).clip(lower=0.0).resample("1h").mean()

    # 3. Precio PVPC y CO2
    df_preu = pd.read_csv(data_dir / "precio_pvpc.csv")
    df_preu["Fecha"] = pd.to_datetime(df_preu["Fecha"]).dt.floor("h")
    df_preu = df_preu.set_index("Fecha").sort_index()

    df_co2 = pd.read_csv(data_dir / "co2_red.csv")
    df_co2["Fecha"] = pd.to_datetime(df_co2["Fecha"]).dt.floor("h")
    df_co2 = df_co2.set_index("Fecha").sort_index()

    df = pd.DataFrame({
        "load_kw": load_h,
        "pv_kw": pv_h,
        "precio_eur_per_kwh": df_preu["precio_eur_per_kwh"],
        "co2_kg_per_kwh": df_co2["co2_kg_per_kwh"],
    }).dropna()

    return df.reset_index()


class ZeroAgent:
    def act(self, obs: dict) -> float:
        return 0.0


class HeuristicBaselineAgent:
    """Política heurística oficial del starter_kit."""
    def act(self, obs: dict) -> float:
        hora = obs["hour"]
        preu = obs["preu_actual"]
        soc = obs["soc_pct"]
        co2 = obs["co2_actual"]
        consigna = 0.0
        if 1 <= hora <= 6 and preu < 0.08 and soc < 0.85:
            consigna = 350.0
        elif 11 <= hora <= 15 and preu < 0.10 and soc < 0.95:
            consigna = 250.0
        elif 18 <= hora <= 22 and (preu > 0.16 or co2 > 0.10) and soc > 0.15:
            consigna = -450.0
        soc_kwh = float(obs["soc_kwh"])
        eta = 0.9487
        max_c = min(500.0, (2000.0 - soc_kwh) / eta)
        max_d = min(500.0, soc_kwh * eta)
        return float(max(-max_d, min(max_c, consigna)))


if __name__ == "__main__":
    df = construir_dataset_horari()
    print(f"Dataset horario cargado: {len(df)} horas\n")

    agents = [
        ("0. Sin batería", ZeroAgent()),
        ("1. Heurística oficial", HeuristicBaselineAgent()),
        ("2. Tu agent.py (MPC)", carregar_agent("agent.py")),
    ]

    resultats = []
    for nom, ag in agents:
        t0 = time.perf_counter()
        res = run_agent_simulation(ag, df)
        dt = time.perf_counter() - t0
        resultats.append({
            "Agente": nom,
            "Score": res["score"],
            "ΔCO2 (%)": res["co2_estalviat_pct"],
            "ΔCost (%)": res["cost_estalviat_pct"],
            "Infracciones": res["infraccions"],
            "Tiempo (s)": round(dt, 2),
        })

    print(pd.DataFrame(resultats).to_string(index=False))