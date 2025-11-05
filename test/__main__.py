"""Simple test runner to exercise the KalmanFilter implementation.

Run with:

	python -m test

This script simulates a 1D linear Gaussian system, runs the filter, computes
RMSE against the true states, and prints a short report.
"""

from __future__ import annotations

import sys
import numpy as np


def main() -> None:
	try:
		# Import the KalmanFilter implementation from the package
		from src.models.KalmanFilter import KalmanFilter
	except Exception as e:  # pragma: no cover - helpful message on import failure
		print("Failed to import KalmanFilter. Make sure the package is installed and the PYTHONPATH is correct.")
		print("Error:", e)
		sys.exit(2)

	# Simulation parameters
	T = 100
	A = np.array([[1.0]], dtype=float)
	H = np.array([[1.0]], dtype=float)
	Q = np.array([[0.1]], dtype=float)
	R = np.array([[0.5]], dtype=float)

	# Simulate latent states and observations
	rng = np.random.default_rng(1)
	x = np.zeros((T + 1, 1))
	y = np.zeros((T, 1))
	for t in range(1, T + 1):
		x[t] = A @ x[t - 1] + rng.normal(scale=np.sqrt(Q[0, 0]))
	for t in range(T):
		y[t] = H @ x[t + 1] + rng.normal(scale=np.sqrt(R[0, 0]))

	kf = KalmanFilter(
		transition_matrix=A,
		observation_matrix=H,
		transition_cov=Q,
		observation_cov=R,
		initial_mean=np.zeros((1,)),
		initial_cov=np.eye(1) * 1.0,
	)

	fm, Fc, ll = kf.filter(y)

	fm_np = np.asarray(fm).squeeze()
	true_np = x[1:].squeeze()

	rmse = np.sqrt(np.mean((fm_np - true_np) ** 2))

	print("KalmanFilter demo summary")
	print("- Time steps:", T)
	print("- Log-likelihood:", float(ll))
	print("- RMSE:", float(rmse))

	threshold = 1.0
	if rmse < threshold:
		print(f"PASS: RMSE {rmse:.4f} < {threshold}")
		sys.exit(0)
	else:
		print(f"FAIL: RMSE {rmse:.4f} >= {threshold}")
		sys.exit(1)


if __name__ == "__main__":
	main()

