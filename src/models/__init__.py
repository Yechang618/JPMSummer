"""Models package.

This file exposes the KalmanFilter submodule at package level so the
notebook import `from src.models import KalmanFilter as kf` works and
`kf` will refer to the `src.models.KalmanFilter` module.
"""

from . import KalmanFilter  # noqa: F401  (expose submodule)

__all__ = ["KalmanFilter"]
