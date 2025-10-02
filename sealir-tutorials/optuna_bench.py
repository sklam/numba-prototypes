"""

Requirements
- pip install optuna     # for hyperparameter search
- conda install sklearn  # for plot_param_importances

"""

import os
import sys
import tempfile
import optuna
from multiprocessing import Process

def proc_target(tile_L1, tile_L2, tile_L3, unroll_factor, vector_size, bench_output):
    os.environ["LLM_tile_L1"] = str(tile_L1)
    os.environ["LLM_tile_L2"] = str(tile_L2)
    os.environ["LLM_tile_L3"] = str(tile_L3)
    os.environ["LLM_unroll_factor"] = str(unroll_factor)
    os.environ["LLM_vector_size"] = str(vector_size)
    os.environ["LLM_bench_output"] = str(bench_output)

    from llm_midend import test_matmul_performance
    test_matmul_performance()

def run(tile_L1, tile_L2, tile_L3, unroll_factor, vector_size, bench_output):

    proc = Process(target=proc_target, args=(tile_L1, tile_L2, tile_L3, unroll_factor, vector_size, bench_output))
    proc.start()
    proc.join(timeout=10 * 60)

    with open(bench_output) as fin:
        records = [float(ln) for ln in fin]
    if records:
        return min(records)
    else:
        # Compilation failed
        return float('inf')


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    def objective(trial):
        with tempfile.NamedTemporaryFile(mode="a") as tmpfile:

            r = run(
                tile_L1=trial.suggest_int("tile_L1", 1, 64),
                tile_L2=trial.suggest_int("tile_L2", 1, 64),
                tile_L3=trial.suggest_int("tile_L3", 1, 64),
                unroll_factor=trial.suggest_int("unroll_factor", 1, 16),
                vector_size=trial.suggest_int("vector_size", 1, 16),
                bench_output=tmpfile.name,
            )

            print('timing', r)

            return r

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    trial = study.best_trial

    print('Timing: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


    from optuna.visualization import plot_optimization_history, plot_slice, plot_param_importances
    import plotly.offline as pyo


    pyo.plot(plot_optimization_history(study),
            filename='optimization_history.html')
    pyo.plot(plot_slice(study),
            filename='slice_plot.html')
    pyo.plot(plot_param_importances(study),
             filename='param_importances.html')

if __name__ == "__main__":
    main()
