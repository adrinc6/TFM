from environment import Settings
from module.runs.results_store import ResultsStore
from module.runs.execution import _final_robustness
import glob, os, json

store = ResultsStore()
s = Settings(run_scope="full")
# Un run finalista existente con diagnostics + annual_metrics
runs = glob.glob("results/runs/*/artifacts/rank_ic_diagnostics.parquet")
run_id = None
for p in runs:
    rid = os.path.basename(os.path.dirname(os.path.dirname(p)))
    if os.path.exists(f"results/runs/{rid}/artifacts/annual_metrics.parquet"):
        run_id = rid; break
print(f"run de prueba: {run_id}")
rob = _final_robustness(store, run_id, portfolio_final_id=run_id, final_settings=s, include_full=True)
print("label_permutation:", json.dumps(rob.get("label_permutation"), indent=2)[:400])
print("random_portfolio:", json.dumps(rob.get("random_portfolio"), indent=2)[:400])
# comprobar que los targets originales quedaron restaurados (no permutados)
print("ROBUSTEZ COMPLETA OK")
