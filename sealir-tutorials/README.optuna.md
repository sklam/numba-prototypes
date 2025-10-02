# Optuna MLIR tuning


## SETUP

```bash
git clone https://github.com/sklam/numba-prototypes --branch=wip/llm_optuna
git clone https://github.com/sklam/sealir --branch=wip/updates
cd numba-prototypes/sealir-tutorials/
conda env create -f conda_environment.yml -n llm_optuna
conda activate llm_optuna

pip install -e ../../sealir

# for optuna
pip install optuna

# for optuna importances plot
conda install  -c conda-forge scikit-learn plotly


# check with
pytest ./llm_midend.py::test_matmul_performance

```

## Tuning

```
# run optuna
python optuna_bench.py [n_trials (default=50)]
```

## Run the matmul function


```bash
pytest ./llm_midend.py::test_matmul_performance -s
```

prints:

```
llm_midend.py::test_matmul_performance x.shape (1, 5, 288)
q_weight.shape (288, 288)

NumPy: Exec 177.000 microseconds

NumPy: Exec 55.000 microseconds

NumPy: Exec 45.000 microseconds
MLIRGen: To Memref 870.0 microseconds, Exec 109.0 microseconds, To NumPy 140.0 microseconds
MLIRGen: To Memref 120.0 microseconds, Exec 60.0 microseconds, To NumPy 35.0 microseconds
MLIRGen: To Memref 212.0 microseconds, Exec 60.0 microseconds, To NumPy 28.0 microseconds
```

Note: 

- use the `Exec` time for execution time.
- `To memref` and `To NumPy` are the overheads.


Use env-var to set matrix size. The MatMul is a `matrix[M x N] @ matrix[N x N]`
```bash
LLM_matmul_M=100 LLM_matmul_N=283 pytest ./llm_midend.py::test_matmul_performance -s
```

Use env-var to set the tuning options:

- `LLM_tile_L1`
- `LLM_tile_L2`
- `LLM_tile_L3`
- `LLM_unroll_factor`
- `LLM_vector_size`
