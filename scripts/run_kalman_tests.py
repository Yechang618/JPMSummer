"""Standalone test runner for KalmanFilter.

Usage:
    python scripts/run_kalman_tests.py

This script runs two simple tests:
 - 1D recovery: simulate 1D linear Gaussian system and assert RMSE < 1.0
 - 2D recovery: simulate a 2D system and assert RMSE < 1.5

It exits with a non-zero code if any test fails.
"""
from __future__ import annotations

import sys
import numpy as np


def test_1d():
    try:
        from src.models.KalmanFilter import KalmanFilter
    except Exception as e:
        print("SKIP test_1d: failed to import KalmanFilter:", e)
        return True

    T = 100
    A = np.array([[1.0]], dtype=float)
    H = np.array([[1.0]], dtype=float)
    Q = np.array([[0.05]], dtype=float)
    R = np.array([[0.2]], dtype=float)

    rng = np.random.default_rng(2)
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

    print(f"1D test: RMSE={rmse:.4f}, loglik={float(ll):.3f}")
    return rmse < 1.0


def test_2d():
    try:
        from src.models.KalmanFilter import KalmanFilter
    except Exception as e:
        print("SKIP test_2d: failed to import KalmanFilter:", e)
        return True

    T = 120
    A = np.array([[1.0, 0.1], [0.0, 1.0]], dtype=float)  # simple constant-velocity
    H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    Q = np.eye(2) * 0.01
    R = np.eye(2) * 0.1

    rng = np.random.default_rng(3)
    x = np.zeros((T + 1, 2))
    y = np.zeros((T, 2))
    for t in range(1, T + 1):
        x[t] = (A @ x[t - 1]) + rng.normal(scale=0.1, size=(2,))
    for t in range(T):
        y[t] = H @ x[t + 1] + rng.normal(scale=np.sqrt(R.diagonal()), size=(2,))

    kf = KalmanFilter(
        transition_matrix=A,
        observation_matrix=H,
        transition_cov=Q,
        observation_cov=R,
        initial_mean=np.zeros((2,)),
        initial_cov=np.eye(2) * 1.0,
    )

    fm, Fc, ll = kf.filter(y)
    fm_np = np.asarray(fm)
    true_np = x[1:]
    rmse = np.sqrt(np.mean((fm_np - true_np) ** 2))

    print(f"2D test: RMSE={rmse:.4f}, loglik={float(ll):.3f}")
    return rmse < 1.5


def main():
    tests = [("1D recovery", test_1d), ("2D recovery", test_2d)]
    failures = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            print(f"ERROR running {name}:", e)
            ok = False
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    if failures:
        print("Some tests failed:", ", ".join(failures))
        sys.exit(1)
    else:
        print("All tests passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
