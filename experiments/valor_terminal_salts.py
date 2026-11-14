"""Prova complementaria: blocs mensuals concatenats amb SOC persistent.
Els vectors de previsio del simulador local poden travessar els salts: aproximacio, no test oficial.
"""
import tempfile
import pandas as pd
from eval_local import construir_dataset_horari
from experiments.comparar_salts import blocs_mensuals
from experiments.valor_terminal import variants, avaluar


def main():
    data = blocs_mensuals(construir_dataset_horari())
    print('Hores:', len(data), '| blocs:', len(data)//168,
          '| SOC persistent; previsions del simulador local poden travessar salts', flush=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='terminal_salts_') as tmp:
        agents, fingerprint = variants(tmp)
        for factor in [0.0, 0.5, 1.0, 1.5, 2.0]:
            result = avaluar(agents[factor], data)
            rows.append({'factor': factor, 'agent_sha256': fingerprint, **result})
            print(f"factor={factor:.1f} score={result['score']:.4f} co2={result['co2_pct']:.3f}% cost={result['cost_pct']:.3f}% infraccions={result['infraccions']} soc_final={result['soc_final_kwh']:.1f}", flush=True)
    pd.DataFrame(rows).to_csv('experiments/resultats_terminal_salts.csv', index=False)

if __name__ == '__main__':
    main()
