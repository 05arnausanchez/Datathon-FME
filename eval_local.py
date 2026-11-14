"""Avaluacio local del repte, per blocs temporals continus.

Exemples (des de l'arrel):
    python eval_local.py
    python eval_local.py --agent legacy/v3_alineada.py --mode setmanes --csv resultats.csv

La puntuacio fa servir les regles del simulador del kit; no es el test ocult.
Cada bloc s'inicia amb SOC=50% i un agent nou. No hi ha dades entre blocs.
"""
import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from starter_kit.environment import CampusBatteryEnv

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'data' / 'train'


def carregar_agent(ruta_py):
    path = Path(ruta_py).resolve()
    spec = importlib.util.spec_from_file_location('mod_agent', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'No es pot carregar {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatteryAgent()


def _llegir(path, columns):
    data = pd.read_csv(path, usecols=['Fecha', *columns], parse_dates=['Fecha'])
    if data['Fecha'].isna().any() or data['Fecha'].duplicated().any():
        raise ValueError(f'{path.name}: dates buides o duplicades')
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)
    return data.set_index('Fecha').sort_index()


def construir_dataset_horari(data_dir=DEFAULT_DATA):
    data_dir = Path(data_dir)
    meter = 'Contador - POTENCIA ACTIVA EXPORTADA (kW)'
    inv = [f'Inversor {n} - POTENCIA ACTIVA (kW)' for n in (1, 2, 3)]
    load = (-_llegir(data_dir / 'contador.csv', [meter])[meter]).clip(0, 5000).resample('1h').mean()
    pv = _llegir(data_dir / 'inversor.csv', inv)[inv].sum(axis=1, min_count=len(inv)).clip(lower=0).resample('1h').mean()
    price = _llegir(data_dir / 'precio_pvpc.csv', ['precio_eur_per_kwh'])['precio_eur_per_kwh']
    co2 = _llegir(data_dir / 'co2_red.csv', ['co2_kg_per_kwh'])['co2_kg_per_kwh']
    # Series de mercat horaries: no agregar registres duplicats ni desplacats silenciosament.
    if (price.index != price.index.floor('h')).any() or (co2.index != co2.index.floor('h')).any():
        raise ValueError('Les dates de preu i CO2 han de coincidir amb l\'inici de cada hora')
    data = pd.concat([load.rename('load_kw'), pv.rename('pv_kw'), price, co2], axis=1).sort_index()
    data.index.name = 'Fecha'
    return data.dropna().reset_index()


class ZeroAgent:
    def act(self, obs):
        return 0.0


class HeuristicBaselineAgent:
    """Politica de referencia amb marge numeric als limits fisics."""
    def act(self, obs):
        hour, price, soc, co2 = obs['hour'], obs['preu_actual'], obs['soc_pct'], obs['co2_actual']
        action = 0.0
        if 1 <= hour <= 6 and price < 0.08 and soc < 0.85:
            action = 350.0
        elif 11 <= hour <= 15 and price < 0.10 and soc < 0.95:
            action = 250.0
        elif 18 <= hour <= 22 and (price > 0.16 or co2 > 0.10) and soc > 0.15:
            action = -450.0
        eta = math.sqrt(0.9)
        energy = float(obs['soc_kwh'])
        max_charge = max(0.0, min(500.0, (2000.0 - energy) / eta - 1e-4))
        max_discharge = max(0.0, min(500.0, energy * eta - 1e-4))
        return float(max(-max_discharge, min(max_charge, action)))


def seleccionar_blocs(data, mode='continus', min_hores=24):
    data = data.sort_values('Fecha').reset_index(drop=True)
    if data.empty or data['Fecha'].duplicated().any():
        raise ValueError('No hi ha dades o hi ha hores duplicades')
    if mode == 'setmanes':
        result = []
        for week, group in data.groupby(data['Fecha'].dt.to_period('W-SUN')):
            start = week.start_time
            if len(group) == 168 and pd.DatetimeIndex(group['Fecha']).equals(pd.date_range(start, periods=168, freq='h')):
                result.append(group.reset_index(drop=True))
        return result
    gaps = data['Fecha'].diff().ne(pd.Timedelta(hours=1)).cumsum()
    return [group.reset_index(drop=True) for _, group in data.groupby(gaps) if len(group) >= min_hores]


def simular(factory, data, export_price):
    agent = factory()
    env = CampusBatteryEnv(data, export_price=export_price)
    obs = env.reset()
    start = time.perf_counter()
    while obs is not None:
        action = float(agent.act(obs))
        if not math.isfinite(action):
            raise ValueError(f"Accio no finita a {obs['timestamp']}: {action}")
        obs, _ = env.step(action)
    elapsed = time.perf_counter() - start
    # Quantitats sense arrodonir: mateixes equacions que evaluate_results del kit.
    grid, exports = np.asarray(env.grid), np.asarray(env.exports)
    rows = env.df
    price, co2 = rows['precio_eur_per_kwh'].to_numpy(), rows['co2_kg_per_kwh'].to_numpy()
    baseline = np.maximum(rows['load_kw'].to_numpy() - rows['pv_kw'].to_numpy(), 0.0)
    return {
        'cost_base': float(np.sum(baseline * price)),
        'cost_agent': float(np.sum(grid * price) - export_price * np.sum(exports)),
        'co2_base': float(np.sum(baseline * co2)),
        'co2_agent': float(np.sum(grid * co2)),
        'infraccions': env.clipped + env.violations,
        'temps_s': elapsed,
        'soc_final_kwh': float(env.soc),
    }


def resum(raw):
    if raw['co2_base'] <= 0 or raw['cost_base'] <= 0:
        raise ValueError('La base de cost o CO2 no es positiva; no es pot calcular un percentatge fiable')
    co2_pct = 100 * (raw['co2_base'] - raw['co2_agent']) / raw['co2_base']
    cost_pct = 100 * (raw['cost_base'] - raw['cost_agent']) / raw['cost_base']
    score = max(0.0, 0.7 * co2_pct + 0.3 * cost_pct - 0.2 * raw['infraccions'])
    return {'score': score, 'co2_pct': co2_pct, 'cost_pct': cost_pct}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', type=Path, default=ROOT / 'agent.py')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--mode', choices=['continus', 'setmanes'], default='continus')
    parser.add_argument('--min-hores', type=int, default=24, help='Minim per bloc continu (per defecte 24)')
    parser.add_argument('--preu-exportacio', type=float, default=0.0,
                        help="Preu d'exportacio en EUR/kWh (per defecte 0, segons l'enunciat)")
    parser.add_argument('--csv', type=Path, help='Opcional: desa les metriques detallades per bloc')
    args = parser.parse_args()
    if args.min_hores < 1 or not math.isfinite(args.preu_exportacio):
        parser.error('Els parametres numerics no son valids')
    data = construir_dataset_horari(args.data_dir)
    blocks = seleccionar_blocs(data, args.mode, args.min_hores)
    if not blocks:
        raise ValueError('No hi ha blocs que compleixin els criteris')
    selected = sum(map(len, blocks))
    print(f'Dades: {len(data)} hores valides; {len(blocks)} blocs; {selected} hores avaluades; {len(data)-selected} excloses')
    print(f'Mode: {args.mode}; SOC inicial 50% i agent nou per bloc; exportacio {args.preu_exportacio:.4f} EUR/kWh')
    print('Previsions: valors futurs del simulador local, inclosa l\'hora actual; no son una estimacio de precisio predictiva.')
    print('Balanc: load=-comptador i xarxa=load-solar+bateria, segons kit; significat del comptador pendent de confirmar.')
    if (data['pv_kw'] > data['load_kw']).any():
        print('AVIS: el simulador del kit no acredita exportacions a la base sense bateria; el score pot quedar esbiaixat.')
    print('Limitacio del kit: al primer pas de cada bloc usa el consum i solar de la mateixa hora.')
    print('Score: pesos 70/30 i penalitzacio 0.20 per infraccio segons environment.py; bases escrites discrepen.')
    print('Periode ocult, tractament dels salts i entorn del servidor poden diferir. Temps nomes orientatiu.')
    factories = [('sense_bateria', ZeroAgent), ('heuristica', HeuristicBaselineAgent), ('agent', lambda: carregar_agent(args.agent))]
    rows = []
    for name, factory in factories:
        totals = dict.fromkeys(['cost_base', 'cost_agent', 'co2_base', 'co2_agent', 'infraccions', 'temps_s'], 0.0)
        for block in blocks:
            result = simular(factory, block, args.preu_exportacio)
            for key in totals:
                totals[key] += result[key]
            rows.append({'agent': name, 'inici': block.Fecha.iloc[0], 'final': block.Fecha.iloc[-1],
                         'hores': len(block), **result, **resum(result)})
        metrics = resum(totals)
        print(f"{name:15} score={metrics['score']:.2f} CO2={metrics['co2_pct']:.2f}% cost={metrics['cost_pct']:.2f}% infraccions={int(totals['infraccions'])} temps={totals['temps_s']:.2f}s")
    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f'Detall per bloc desat a {args.csv}')


if __name__ == '__main__':
    main()

