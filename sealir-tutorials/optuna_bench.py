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

PLOT = False

def proc_target(tile_L1, tile_L2, tile_L3, unroll_factor, vector_size, bench_output):
    os.environ["LLM_tile_L1"] = str(tile_L1)
    os.environ["LLM_tile_L2"] = str(tile_L2)
    os.environ["LLM_tile_L3"] = str(tile_L3)
    os.environ["LLM_unroll_factor"] = str(unroll_factor)
    os.environ["LLM_vector_size"] = str(vector_size)
    os.environ["LLM_bench_output"] = str(bench_output)

    # from llm_midend import test_matmul_performance
    # test_matmul_performance()
    from llm_midend import test_attention_full
    test_attention_full()

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
            # Start with the smallest (innermost) tile size
            t1 = trial.suggest_categorical("t1", [8, 16, 32, 64, 128, 256, 512, 1024])

            # L2 is larger (reverse the multipliers)
            t2_mult = trial.suggest_categorical("t2", [1, 2, 4])
            t2 = t1 * t2_mult

            # L3 is even larger
            t3_mult = trial.suggest_categorical("t3", [1, 2, 4])
            t3 = t2 * t3_mult
            r = run(
                tile_L1=t3,
                tile_L2=t2,
                tile_L3=t1,
                unroll_factor=trial.suggest_int("unroll_factor", 1, 16),
                vector_size=trial.suggest_int("vector_size", 1, 16),
                bench_output=tmpfile.name,
            )

            print('timing', r)

            return r


    study_name = "optuna_study-test_attention_full-macos_aarch64"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    trial = study.best_trial

    print('Timing: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


    if PLOT:
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
