"""Compare agents on independent complete calendar weeks; run from repository root."""
import importlib.util
import time
from pathlib import Path
import pandas as pd
from eval_local import construir_dataset_horari
from starter_kit.environment import CampusBatteryEnv


def load_agent(path):
    spec = importlib.util.spec_from_file_location('experiment_agent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatteryAgent()


def run(path, data):
    agent = load_agent(path)
    env = CampusBatteryEnv(data, export_price=0.04)
    obs = env.reset()
    while obs is not None:
        obs, _ = env.step(agent.act(obs))
    return env.evaluate_results(), getattr(agent, 'optim_failures', 0)


def main():
    data = construir_dataset_horari().sort_values('Fecha')
    data['week'] = data['Fecha'].dt.to_period('W-SUN').dt.start_time
    rows = []
    for week, group in data.groupby('week'):
        dates = pd.DatetimeIndex(group['Fecha'])
        if len(group) != 168 or not dates.equals(pd.date_range(week, periods=168, freq='h')):
            continue
        group = group.drop(columns='week').reset_index(drop=True)
        for name, path in [('actual', 'agent.py'), ('preu_real', 'experiments/agent_preu_real.py'), ('previsio_alineada', 'experiments/agent_previsio_alineada.py')]:
            result, failures = run(path, group)
            rows.append({'week': week.date(), 'agent': name, 'score': result['score'],
                         'co2_pct': result['co2_estalviat_pct'], 'cost_pct': result['cost_estalviat_pct'],
                         'infraccions': result['infraccions'], 'errors_optimitzador': failures})
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError('No complete continuous weeks found')
    results.to_csv('experiments/resultats_setmanals.csv', index=False)
    pivot = results.pivot(index='week', columns='agent', values='score')
    diff = pivot['preu_real'] - pivot['actual']
    print('Setmanes completes:', len(pivot))
    print(results.groupby('agent')[['score','co2_pct','cost_pct','infraccions','errors_optimitzador']].agg(['mean','sum']).to_string())
    print('Variant millor/igual/pitjor:', int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum()))
    print('Diferencia score mitjana:', round(diff.mean(), 3))
    aligned = pivot['previsio_alineada'] - pivot['actual']
    print('Previsio alineada millor/igual/pitjor:', int((aligned > 0).sum()), int((aligned == 0).sum()), int((aligned < 0).sum()), 'diferencia mitjana:', round(aligned.mean(), 3))
    print('Resultats: experiments/resultats_setmanals.csv')

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print('Segons:', round(time.perf_counter() - start, 2))

