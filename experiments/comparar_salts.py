"""Compareix agents amb un bloc complet per mes i estat persistent.
Execucio: python -m experiments.comparar_salts
Els blocs son de train i no pretenen reconstruir el test ocult.
"""
import time
import pandas as pd
from eval_local import construir_dataset_horari
from experiments.comparar_setmanes import load_agent
from starter_kit.environment import CampusBatteryEnv


def blocs_mensuals(data):
    data = data.sort_values('Fecha').copy()
    data['week'] = data['Fecha'].dt.to_period('W-SUN').dt.start_time
    choices = {}
    for week, group in data.groupby('week'):
        dates = pd.DatetimeIndex(group['Fecha'])
        if len(group) == 168 and dates.equals(pd.date_range(week, periods=168, freq='h')) and dates[0].month == dates[-1].month:
            choices[week.to_period('M')] = group.drop(columns='week')
    if not choices:
        raise RuntimeError('No hi ha setmanes completes disponibles')
    return pd.concat([choices[key] for key in sorted(choices)], ignore_index=True)


def run(path, data):
    agent = load_agent(path)
    env = CampusBatteryEnv(data, export_price=0.04)
    obs = env.reset()
    start = time.perf_counter()
    while obs is not None:
        obs, _ = env.step(agent.act(obs))
    elapsed = time.perf_counter() - start
    summary = env.evaluate_results()
    dates = env.df['Fecha']
    summaries = []
    for month in sorted(dates.dt.to_period('M').unique()):
        metrics = env.evaluate_results(mask=(dates.dt.to_period('M') == month).to_numpy())
        summaries.append({'mes': str(month), **metrics})
    return summary, summaries, elapsed


def main():
    data = blocs_mensuals(construir_dataset_horari())
    print('Blocs seleccionats:', ', '.join(str(d.date()) for d in data['Fecha'].iloc[::168]))
    print('Hores:', len(data), '| SOC i memoria persistents; previsions segons simulador local')
    rows = []
    for name, path in [('anterior', 'legacy/v_previ_desalineat.py'), ('alineada_original', 'legacy/v3_alineada.py'), ('principal', 'agent.py')]:
        summary, months, elapsed = run(path, data)
        print(name, 'score=', summary['score'], 'co2%=', summary['co2_estalviat_pct'],
              'cost%=', summary['cost_estalviat_pct'], 'infraccions=', summary['infraccions'],
              'temps=', round(elapsed, 2))
        rows.extend({'agent': name, **month} for month in months)
    pd.DataFrame(rows).to_csv('experiments/resultats_salts.csv', index=False)
    table = pd.DataFrame(rows).pivot(index='mes', columns='agent', values='score')
    print(table.to_string())

if __name__ == '__main__':
    main()

