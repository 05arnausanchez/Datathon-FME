"""Sensibilitat del MPC a la previsio de demanda neta en setmanes independents.
El mode oracle utilitza dades futures reals: es exclusivament una referencia no lliurable.
Executar des de l'arrel: python -m experiments.sensibilitat_demanda
"""
import numpy as np
import pandas as pd
from eval_local import construir_dataset_horari, seleccionar_blocs, carregar_agent, resum
from starter_kit.environment import CampusBatteryEnv


def executar(block, mode):
    agent = carregar_agent('agent.py')
    env = CampusBatteryEnv(block, export_price=0.04)
    if mode == 'persistent':
        agent._preveure_demanda_neta = lambda hour, dow, horizon, recent: np.full(horizon, recent, dtype=float)
    elif mode == 'oracle':
        def oracle(hour, dow, horizon, recent):
            data = env.df.iloc[env.t:env.t + horizon]
            if len(data) != horizon:
                raise RuntimeError('Horitzo fora del bloc disponible')
            return (data.load_kw - data.pv_kw).to_numpy(dtype=float)
        agent._preveure_demanda_neta = oracle
    obs = env.reset()
    while obs is not None:
        obs, _ = env.step(agent.act(obs))
    d = env.df
    price, co2 = d.precio_eur_per_kwh.to_numpy(), d.co2_kg_per_kwh.to_numpy()
    base = np.maximum((d.load_kw - d.pv_kw).to_numpy(), 0.0)
    grid, exports = np.asarray(env.grid), np.asarray(env.exports)
    raw = {'cost_base': float(np.sum(base * price)), 'cost_agent': float(np.sum(grid * price) - 0.04 * np.sum(exports)),
           'co2_base': float(np.sum(base * co2)), 'co2_agent': float(np.sum(grid * co2)),
           'infraccions': env.clipped + env.violations}
    return {**raw, **resum(raw)}


def main():
    blocks = seleccionar_blocs(construir_dataset_horari(), mode='setmanes')
    split = len(blocks) * 2 // 3
    rows = []
    print('Setmanes:', len(blocks), '| exploracio:', split, '| reserva cronologica:', len(blocks)-split, flush=True)
    for i, block in enumerate(blocks):
        for mode in ['actual', 'persistent', 'oracle']:
            rows.append({'setmana': str(block.Fecha.iloc[0].date()), 'grup': 'exploracio' if i < split else 'reserva',
                         'mode': mode, **executar(block, mode)})
    results = pd.DataFrame(rows)
    results.to_csv('experiments/resultats_sensibilitat.csv', index=False)
    for group, subset in results.groupby('grup', sort=False):
        print('\n' + group)
        for mode, d in subset.groupby('mode', sort=False):
            total = d[['cost_base','cost_agent','co2_base','co2_agent','infraccions']].sum().to_dict()
            metrics = resum(total)
            print(mode, 'score', round(metrics['score'], 3), 'co2%', round(metrics['co2_pct'], 3),
                  'cost%', round(metrics['cost_pct'], 3), 'infraccions', int(total['infraccions']))
        p = subset.pivot(index='setmana', columns='mode', values='score')
        for mode in ['persistent', 'oracle']:
            diff = p[mode] - p['actual']
            print(mode, 'millor/igual/pitjor', int((diff > 1e-9).sum()), int((diff.abs() <= 1e-9).sum()), int((diff < -1e-9).sum()),
                  'diferencia mitjana setmanal', round(diff.mean(), 4))
    print('\nOracle usa demanda futura real i NO es lliurable; no garanteix una cota optima del score.')
    print('Resultats: experiments/resultats_sensibilitat.csv')

if __name__ == '__main__':
    main()
