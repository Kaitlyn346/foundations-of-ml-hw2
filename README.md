# Foundations of Machine Learning — Homework 2

Solutions notebook: [`data2/homework_2_new.ipynb`](data2/homework_2_new.ipynb) (assignment: [`assignment.pdf`](assignment.pdf)).

| Question | Topic |
|---|---|
| Q1 | Heteroscedastic MLP (mean + variance) on NYC taxi trip durations, P(trip < 45 min) |
| Q2 | Encoding high-cardinality categoricals (retail data): target encoding, embeddings, and both |
| Q3 | Fitting a 2-D surface with random kitchen sinks vs. a one-hidden-layer MLP |

## Running

The notebook imports the helpers in `data2/` (`dataloader.py`, `plotting.py`) and reads the datasets from the same folder. It writes `rks_surface.png`, `mlp_surface.png`, `rks_test_predictions.csv` and `mlp_test_predictions.csv`.

Requirements: Python 3.11, `numpy<2`, `pandas`, `scikit-learn>=1.4` (for `TargetEncoder`), `torch`, `scipy`, `matplotlib`, `pyarrow<18` (to read the parquet file with NumPy 1.x).
