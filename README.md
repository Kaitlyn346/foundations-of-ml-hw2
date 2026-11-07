# Foundations of ML — Homework 2

My solutions are in [`data2/homework_2_new.ipynb`](data2/homework_2_new.ipynb); the questions are in [`assignment.pdf`](assignment.pdf).

- **Q1:** an MLP that predicts both the mean and the spread of NYC taxi trip times, used to estimate P(trip < 45 min)
- **Q2:** target encoding vs. embeddings for high-cardinality categoricals (retail data)
- **Q3:** random kitchen sinks vs. a one-hidden-layer MLP on a 2-D surface

Everything the notebook needs is in `data2/`. It runs on Python 3.11 with `numpy<2`, `pandas`, `scikit-learn>=1.4`, `torch`, `scipy`, `matplotlib` and `pyarrow<18`.
