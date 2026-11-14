"""
agent.py
Agent MPC (Model Predictive Control) + Machine Learning per al repte UPC Sostenible - Datathon 2026.
Optimitza el balanç ponderat 70% CO2 / 30% Cost amb 0 infraccions físiques.

Prediccio amb Random Forest entrenat sobre dades de train:
  - preu PVPC (€/kWh)
  - CO2 xarxa (kg CO2/kWh)
  - generacio solar (kW)

Consulta variables del entorn per les 24 hores futures (preus_properes_24h, co2_properes_24h)
i combina amb prediccio ML per optimitzar la bateria amb linprog (programacio lineal).
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog

try:
    import joblib
except Exception:
    joblib = None


class BatteryAgent:
    def __init__(self):
        # Especificacions físiques oficials
        self.cap_max_kwh = 2000.0
        self.pot_max_kw = 500.0
        self.eta = 0.9 ** 0.5
        self.inv_eta = 1.0 / self.eta
        self.eps_seguretat = 1e-4
        self.marge_soc = 0.5

        # Pesos de l’objectiu (70% CO2 / 30% Cost)
        self.pes_co2 = 0.70
        self.pes_cost = 0.30
        self.ref_co2 = 0.0847
        self.ref_cost = 0.1380
        self.preu_excedent = 0.04

        # Perfil adaptatiu de demanda neta (Demanda - Solar) en kW: [laborable/cap de setmana, 24 hores]
        perfil_base = np.array([
            220.0, 200.0, 195.0, 195.0, 205.0, 240.0,
            320.0, 480.0, 620.0, 650.0, 580.0, 420.0,
            310.0, 280.0, 320.0, 430.0, 560.0, 640.0,
            610.0, 520.0, 410.0, 320.0, 260.0, 230.0,
        ])
        self.perfil_demanda_neta = np.vstack([perfil_base, perfil_base * 0.55])
        self.alpha_ema = 0.25

        # Modelos ML
        self.model_price = None
        self.model_co2 = None
        self.model_pv = None
        self.feature_cols = None
        self._carregar_models()

        self.d_recent_anterior = 0.0

    def _carregar_models(self):
        directori = Path(__file__).resolve().parent
        try:
            if joblib is not None:
                f_price = directori / "model_price.pkl"
                f_co2 = directori / "model_co2.pkl"
                f_pv = directori / "model_pv.pkl"
                f_feats = directori / "feature_cols.pkl"
                if f_price.exists():
                    self.model_price = joblib.load(f_price)
                if f_co2.exists():
                    self.model_co2 = joblib.load(f_co2)
                if f_pv.exists():
                    self.model_pv = joblib.load(f_pv)
                if f_feats.exists():
                    self.feature_cols = joblib.load(f_feats)
                if self.model_price is not None:
                    print(f"[Agent] Modelos ML cargados desde {directori}")
        except Exception as e:
            print(f"[Agent] Aviso: no se pudieron cargar modelos ML ({e}). Usando heurística.")

    def _actualitzar_telemetria(self, obs: dict) -> float:
        dades = obs.get("noves_dades_5min") or {}
        consum = dades.get("ultim_consum_kw")
        solar = dades.get("ultima_solar_kw")

        hora_actual = int(obs["hour"])
        dow_actual = int(obs["day_of_week"])

        if consum is not None and solar is not None:
            demanda_neta_mesurada = float(consum) - float(solar)
            hora_prev = (hora_actual - 1) % 24
            dow_prev = (dow_actual - 1) % 7 if hora_actual == 0 else dow_actual
            tipus_dia_prev = 1 if dow_prev >= 5 else 0

            actual = float(self.perfil_demanda_neta[tipus_dia_prev, hora_prev])
            self.perfil_demanda_neta[tipus_dia_prev, hora_prev] = (
                (1.0 - self.alpha_ema) * actual + self.alpha_ema * demanda_neta_mesurada
            )
            self.d_recent_anterior = demanda_neta_mesurada
            return demanda_neta_mesurada

        tipus_dia = 1 if dow_actual >= 5 else 0
        return float(self.perfil_demanda_neta[tipus_dia, hora_actual])

    def _preveure_demanda_neta(self, hora_inici: int, dow_inici: int, horitzo: int, d_recent: float) -> np.ndarray:
        previsio = np.zeros(horitzo, dtype=float)
        tipus_dia_0 = 1 if dow_inici >= 5 else 0
        biaix_actual = d_recent - self.perfil_demanda_neta[tipus_dia_0, hora_inici]

        for k in range(horitzo):
            h = (hora_inici + k) % 24
            dies_extra = (hora_inici + k) // 24
            dow = (dow_inici + dies_extra) % 7
            tipus_dia = 1 if dow >= 5 else 0
            previsio[k] = self.perfil_demanda_neta[tipus_dia, h] + biaix_actual * (0.65 ** k)

        return previsio

    def _construir_features_ml(self, hora: int, dow: int, mes: int,
                               load_lag1: float, load_lag24: float, load_roll24: float,
                               pv_lag1: float, pv_lag24: float, pv_roll24: float,
                               price_lag1: float, price_lag24: float, price_roll24: float,
                               co2_lag1: float, co2_lag24: float, co2_roll24: float,
                               irrad_lag1: float, irrad_roll24: float,
                               temp_lag1: float) -> np.ndarray:
        is_weekend = 1 if dow >= 5 else 0
        return np.array([[
            hora, dow, mes, is_weekend,
            load_lag1, load_lag24, load_roll24,
            pv_lag1, pv_lag24, pv_roll24,
            price_lag1, price_lag24, price_roll24,
            co2_lag1, co2_lag24, co2_roll24,
            irrad_lag1, irrad_roll24,
            temp_lag1,
        ]], dtype=float)

    def _preveure_preu_co2_solar(self, obs: dict, preus_24h: np.ndarray,
                                  co2_24h: np.ndarray) -> tuple:
        hora = int(obs["hour"])
        dow = int(obs["day_of_week"])
        ts = obs.get("timestamp", None)
        mes = pd.Timestamp(ts).month if ts is not None else 1

        dades = obs.get("noves_dades_5min") or {}
        load_lag1 = float(dades.get("ultim_consum_kw", 1000.0))
        pv_lag1 = float(dades.get("ultima_solar_kw", 0.0))

        if self.model_price is None or self.model_co2 is None or self.feature_cols is None:
            return preus_24h, co2_24h

        features_base = self._construir_features_ml(
            hora=hora, dow=dow, mes=mes,
            load_lag1=load_lag1, load_lag24=load_lag1, load_roll24=load_lag1,
            pv_lag1=pv_lag1, pv_lag24=pv_lag1, pv_roll24=pv_lag1,
            price_lag1=float(obs["preu_actual"]), price_lag24=float(obs["preu_actual"]),
            price_roll24=float(obs["preu_actual"]),
            co2_lag1=float(obs["co2_actual"]), co2_lag24=float(obs["co2_actual"]),
            co2_roll24=float(obs["co2_actual"]),
            irrad_lag1=pv_lag1 * 10.0,
            irrad_roll24=pv_lag1 * 10.0,
            temp_lag1=22.0,
        )

        try:
            price_ml = float(self.model_price.predict(features_base)[0])
            co2_ml = float(self.model_co2.predict(features_base)[0])
        except Exception:
            price_ml = float(obs["preu_actual"])
            co2_ml = float(obs["co2_actual"])

        alpha = 0.30
        preus_ajustats = (1 - alpha) * preus_24h + alpha * price_ml
        co2_ajustats = (1 - alpha) * co2_24h + alpha * co2_ml

        return preus_ajustats, co2_ajustats

    def _resoldre_mpc(self, soc_0: float, preus: np.ndarray, co2: np.ndarray,
                      demanda_neta: np.ndarray) -> float:
        H = len(preus)
        if H == 0:
            return 0.0

        n_vars = 4 * H

        preus_imp = np.maximum(preus, self.preu_excedent + 1e-3)
        w_imp = (self.pes_co2 * (co2 / self.ref_co2)) + (self.pes_cost * (preus_imp / self.ref_cost))
        w_exp = -self.pes_cost * (self.preu_excedent / self.ref_cost)

        valor_terminal = self.eta * float(np.percentile(w_imp, 70))

        c = np.zeros(n_vars, dtype=float)
        eps_cicle = 1e-6
        for k in range(H):
            idx = 4 * k
            c[idx + 0] = eps_cicle - valor_terminal * self.eta
            c[idx + 1] = eps_cicle + valor_terminal / self.eta
            c[idx + 2] = w_imp[k]
            c[idx + 3] = w_exp

        A_eq = np.zeros((H, n_vars), dtype=float)
        b_eq = demanda_neta.copy()
        for k in range(H):
            idx = 4 * k
            A_eq[k, idx + 0] = -1.0
            A_eq[k, idx + 1] = 1.0
            A_eq[k, idx + 2] = 1.0
            A_eq[k, idx + 3] = -1.0

        marge = self.marge_soc
        espai_sup = max(0.0, self.cap_max_kwh - soc_0 - marge)
        espai_inf = max(0.0, soc_0 - marge)

        A_ub = np.zeros((2 * H, n_vars), dtype=float)
        b_ub = np.zeros(2 * H, dtype=float)

        for k in range(H):
            b_ub[k] = espai_sup
            b_ub[H + k] = espai_inf
            for j in range(k + 1):
                idx = 4 * j
                A_ub[k, idx + 0] = self.eta
                A_ub[k, idx + 1] = -self.inv_eta
                A_ub[H + k, idx + 0] = -self.eta
                A_ub[H + k, idx + 1] = self.inv_eta

        bounds = []
        for _ in range(H):
            bounds.extend([
                (0.0, self.pot_max_kw),
                (0.0, self.pot_max_kw),
                (0.0, None),
                (0.0, None),
            ])

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        if res.success and res.x is not None:
            p_ch = float(res.x[0])
            p_dis = float(res.x[1])
            return p_ch - p_dis
        return 0.0

    def act(self, obs: dict) -> float:
        hora = int(obs["hour"])
        dow = int(obs["day_of_week"])
        soc_kwh = float(obs["soc_kwh"])
        preu_actual = float(obs["preu_actual"])
        co2_actual = float(obs["co2_actual"])

        d_recent = self._actualitzar_telemetria(obs)

        preus_24h = np.array(list(obs.get("preus_properes_24h") or []), dtype=float)
        co2_24h = np.array(list(obs.get("co2_properes_24h") or []), dtype=float)

        if len(preus_24h) == 0:
            preus_24h = np.array([preu_actual], dtype=float)
        if len(co2_24h) == 0:
            co2_24h = np.array([co2_actual], dtype=float)

        preus_24h[0] = preu_actual
        co2_24h[0] = co2_actual

        preus_aj, co2_aj = self._preveure_preu_co2_solar(obs, preus_24h, co2_24h)
        demanda_neta = self._preveure_demanda_neta(hora, dow, len(preus_aj), d_recent)
        consigna_desitjada = self._resoldre_mpc(soc_kwh, preus_aj, co2_aj, demanda_neta)

        espai_disponible_kwh = max(0.0, self.cap_max_kwh - soc_kwh)
        energia_disponible_kwh = max(0.0, soc_kwh)
        max_carrega = max(0.0, min(self.pot_max_kw, espai_disponible_kwh / self.eta - self.eps_seguretat))
        max_descarrega = max(0.0, min(self.pot_max_kw, energia_disponible_kwh * self.eta - self.eps_seguretat))

        consigna_final = float(np.clip(consigna_desitjada, -max_descarrega, max_carrega))

        if not hasattr(self, "historial"):
            self.historial = []
        self.historial.append({
            "hour": hora, "soc_kwh": soc_kwh, "preu": preu_actual, "co2": co2_actual,
            "consigna": consigna_final,
        })

        return consigna_final