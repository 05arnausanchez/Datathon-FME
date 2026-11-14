"""Compara multiplicadors del valor terminal amb setmanes temporals independents.
Executar: python -B -m experiments.valor_terminal
No modifica agent.py; les variants temporals son nomes per a aquesta prova.
"""
import hashlib
import importlib.util
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from eval_local import construir_dataset_horari, seleccionar_blocs, resum
from starter_kit.environment import CampusBatteryEnv

FACTORS = (0.0, 0.5, 1.0, 1.5, 2.0)
KEY = 'valor_terminal = self.eta * float(np.percentile(w_imp, 70))'


def variants(directory):
    source = Path('agent.py').read_text(encoding='utf-8-sig')
    if source.count(KEY) != 1:
        raise ValueError('Ha canviat agent.py; reviseu la construccio de variants')
    fingerprint = hashlib.sha256(source.encode('utf-8')).hexdigest()
    agents = {}
    for index, factor in enumerate(FACTORS):
        path = Path(directory) / f'agent_{index}.py'
        path.write_text(source.replace(KEY, KEY + f' * {factor!r}'), encoding='utf-8')
        spec = importlib.util.spec_from_file_location(f'agent_terminal_{index}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        agents[factor] = module.BatteryAgent
    return agents, fingerprint


def avaluar(agent_class, data):
    agent = agent_class()
    env = CampusBatteryEnv(data, export_price=0.04)
    obs = env.reset()
    while obs is not None:
        obs, _ = env.step(agent.act(obs))
    d = env.df
    price, co2 = d.precio_eur_per_kwh.to_numpy(), d.co2_kg_per_kwh.to_numpy()
    base = np.maximum((d.load_kw - d.pv_kw).to_numpy(), 0.0)
    grid, exports = np.asarray(env.grid), np.asarray(env.exports)
    raw = {'cost_base': float(np.sum(base * price)), 'cost_agent': float(np.sum(grid * price) - 0.04 * np.sum(exports)),
           'co2_base': float(np.sum(base * co2)), 'co2_agent': float(np.sum(grid * co2)),
           'infraccions': env.clipped + env.violations, 'soc_final_kwh': float(env.soc)}
    return {**raw, **resum(raw)}


def agregat(df):
    raw = df[['cost_base', 'cost_agent', 'co2_base', 'co2_agent', 'infraccions']].sum().to_dict()
    return resum(raw), int(raw['infraccions'])


def main():
    blocks = seleccionar_blocs(construir_dataset_horari(), mode='setmanes')
    split = len(blocks) * 2 // 3
    if split == 0 or split == len(blocks):
        raise ValueError('No hi ha prou setmanes per separar una reserva')
    print('Setmanes:', len(blocks), '| exploracio:', split, '| reserva:', len(blocks)-split,
          '| inici reserva:', blocks[split].Fecha.iloc[0].date(), flush=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='valor_terminal_') as tmp:
        agents, fingerprint = variants(tmp)
        print('SHA256 agent base:', fingerprint, flush=True)
        for i, block in enumerate(blocks):
            for factor, agent_class in agents.items():
                rows.append({'setmana': str(block.Fecha.iloc[0].date()), 'grup': 'exploracio' if i < split else 'reserva',
                             'factor': factor, 'agent_sha256': fingerprint, **avaluar(agent_class, block)})
            if (i + 1) % 8 == 0:
                print('Setmanes completades:', i + 1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('experiments/resultats_valor_terminal.csv', index=False)
    for group, part in df.groupby('grup', sort=False):
        print('\n' + group)
        for factor, subset in part.groupby('factor'):
            result, violations = agregat(subset)
            print(f"factor={factor:.1f} score={result['score']:.4f} co2={result['co2_pct']:.3f}% cost={result['cost_pct']:.3f}% infraccions={violations} soc_final_mitja={subset.soc_final_kwh.mean():.1f}")
        p = part.pivot(index='setmana', columns='factor', values='score')
        for factor in FACTORS:
            if factor == 1.0:
                continue
            diff = p[factor] - p[1.0]
            print(f'{factor:.1f} vs 1.0: millor/igual/pitjor={(diff > 1e-9).sum()}/{(diff.abs() <= 1e-9).sum()}/{(diff < -1e-9).sum()}')
    print('\nResultats: experiments/resultats_valor_terminal.csv')
    print('Cada setmana inicia bateria al 50%; avaluacio amb simulador local, no test ocult.')

if __name__ == '__main__':
    main()
