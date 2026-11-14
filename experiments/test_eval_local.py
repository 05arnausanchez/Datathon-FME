import unittest
import pandas as pd
from eval_local import ZeroAgent, seleccionar_blocs, simular, resum
from starter_kit.environment import CampusBatteryEnv


class AvaluacioTest(unittest.TestCase):
    def dades(self, dates):
        return pd.DataFrame({'Fecha': pd.to_datetime(dates), 'load_kw': 300.0, 'pv_kw': 100.0,
                             'precio_eur_per_kwh': 0.15, 'co2_kg_per_kwh': 0.2})

    def test_salts_no_es_connecten(self):
        dates = list(pd.date_range('2025-01-01', periods=48, freq='h')) + list(pd.date_range('2025-02-01', periods=48, freq='h'))
        blocks = seleccionar_blocs(self.dades(dates), min_hores=24)
        self.assertEqual([len(b) for b in blocks], [48, 48])
        self.assertEqual(seleccionar_blocs(self.dades(dates), mode='setmanes'), [])

    def test_sense_bateria_es_zero(self):
        data = self.dades(pd.date_range('2025-01-06', periods=168, freq='h'))
        raw = simular(ZeroAgent, data, 0.04)
        self.assertEqual(raw['infraccions'], 0)
        self.assertAlmostEqual(resum(raw)['score'], 0.0)
        self.assertAlmostEqual(resum(raw)['co2_pct'], 0.0)
        self.assertAlmostEqual(resum(raw)['cost_pct'], 0.0)

    def test_calcul_concorda_amb_simulador(self):
        data = self.dades(pd.date_range('2025-01-06', periods=48, freq='h'))
        class Agent:
            def act(self, obs):
                return 100.0 if obs['hour'] == 0 else (-80.0 if obs['hour'] == 12 else 0.0)
        raw = simular(Agent, data, 0.04)
        env = CampusBatteryEnv(data, export_price=0.04)
        agent = Agent()
        obs = env.reset()
        while obs is not None:
            obs, _ = env.step(agent.act(obs))
        official = env.evaluate_results()
        result = resum(raw)
        self.assertAlmostEqual(round(result['score'], 2), official['score'], places=2)
        self.assertAlmostEqual(round(result['co2_pct'], 2), official['co2_estalviat_pct'], places=2)
        self.assertAlmostEqual(round(result['cost_pct'], 2), official['cost_estalviat_pct'], places=2)

    def test_rebutja_accions_no_finites(self):
        class Agent:
            def act(self, obs):
                return float('nan')
        with self.assertRaisesRegex(ValueError, 'no finita'):
            simular(Agent, self.dades(pd.date_range('2025-01-01', periods=1, freq='h')), 0.04)


if __name__ == '__main__':
    unittest.main()
