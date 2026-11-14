"""Comparacio de valor terminal amb 24h addicionals i SOC final igualat.
Des de l'arrel: python -B -m experiments.terminal_soc_comparable
La fase final de liquidacio es una convencio experimental, no l'avaluacio oficial.
"""
import tempfile
import numpy as np
import pandas as pd
from eval_local import construir_dataset_horari, seleccionar_blocs, resum
from experiments.valor_terminal import variants
from starter_kit.environment import CampusBatteryEnv

WEEK = 168
TAIL = 24
TARGET = 1000.0
ETA = 0.9 ** 0.5


def trams(data):
    ordered = data.set_index('Fecha').sort_index()
    weeks = seleccionar_blocs(data, mode='setmanes')
    out = []
    for week in weeks:
        start = week.Fecha.iloc[0]
        wanted = pd.date_range(start, periods=WEEK + TAIL, freq='h')
        if wanted.isin(ordered.index).all():
            out.append(ordered.loc[wanted].rename_axis('Fecha').reset_index())
    return out


def totals(env, start, end):
    d = env.df.iloc[start:end]
    grid = np.asarray(env.grid)[start:end]
    exports = np.asarray(env.exports)[start:end]
    price = d.precio_eur_per_kwh.to_numpy()
    co2 = d.co2_kg_per_kwh.to_numpy()
    base = np.maximum((d.load_kw - d.pv_kw).to_numpy(), 0.0)
    raw = {'cost_base': float(np.sum(base * price)),
           'cost_agent': float(np.sum(grid * price) - 0.04 * np.sum(exports)),
           'co2_base': float(np.sum(base * co2)), 'co2_agent': float(np.sum(grid * co2)),
           'infraccions': int(np.sum(env.infraccions_per_step[start:end]))}
    return {**raw, **resum(raw)}


def run(agent_class, data):
    agent = agent_class()
    env = CampusBatteryEnv(data, export_price=0.04)
    obs = env.reset()
    soc_week = None
    for t in range(len(data)):
        if t < WEEK:
            action = agent.act(obs)
        else:
            if t == WEEK:
                soc_week = env.soc
            delta = TARGET - env.soc
            # Accions segures i deterministes fins arribar al mateix SOC, despres repos.
            if delta > 1e-7:
                action = min(500.0, delta / ETA)
            elif delta < -1e-7:
                action = -min(500.0, -delta * ETA)
            else:
                action = 0.0
        obs, _ = env.step(action)
    if abs(env.soc - TARGET) > 1e-5:
        raise AssertionError(f'No s ha igualat SOC: {env.soc}')
    return soc_week, env.soc, totals(env, 0, WEEK), totals(env, 0, WEEK + TAIL)


def aggregate(rows):
    raw = rows[['cost_base','cost_agent','co2_base','co2_agent','infraccions']].sum().to_dict()
    return resum(raw), int(raw['infraccions'])


def main():
    blocks = trams(construir_dataset_horari())
    split = len(blocks)*2//3
    if split == 0 or split == len(blocks):
        raise ValueError('No hi ha prou setmanes amb 24 hores addicionals')
    print('Setmanes:', len(blocks), '| exploracio:', split, '| reserva:', len(blocks)-split,
          '| inici reserva:', blocks[split].Fecha.iloc[0].date(), flush=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='terminal_soc_') as tmp:
        agents, fingerprint = variants(tmp)
        print('SHA256 agent base:', fingerprint, flush=True)
        for i, block in enumerate(blocks):
            for factor in (0.0, 1.0):
                week_soc, final_soc, week, adjusted = run(agents[factor], block)
                for periode, metrics in [('setmana', week), ('setmana_mes_liquidacio', adjusted)]:
                    rows.append({'setmana': str(block.Fecha.iloc[0].date()), 'grup': 'exploracio' if i < split else 'reserva',
                                 'periode': periode, 'factor': factor, 'agent_sha256': fingerprint,
                                 'soc_final_setmana': week_soc, 'soc_final_ajustat': final_soc, **metrics})
            if (i+1)%8 == 0:
                print('Setmanes completades:', i+1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('experiments/resultats_terminal_soc_comparable.csv', index=False)
    for (group, period), part in df.groupby(['grup','periode'], sort=False):
        print('\n',group,period)
        for factor, subset in part.groupby('factor'):
            metrics, infra = aggregate(subset)
            print(f"factor={factor:.1f} score={metrics['score']:.4f} co2={metrics['co2_pct']:.3f}% cost={metrics['cost_pct']:.3f}% infraccions={infra} SOC_setmana={subset.soc_final_setmana.mean():.1f} SOC_ajustat={subset.soc_final_ajustat.mean():.1f}")
        pivot = part.pivot(index='setmana',columns='factor',values='score')
        diff = pivot[0.0]-pivot[1.0]
        print('0 vs 1 millor/igual/pitjor:',int((diff>1e-9).sum()),int((diff.abs()<=1e-9).sum()),int((diff< -1e-9).sum()))
    print('\nLa liquidacio comenca a la primera hora posterior i porta els dos agents a 1000 kWh; les altres hores son repos.')
    print('Els blocs son de train; no es una puntuacio del test ocult.')

if __name__ == '__main__':
    main()
