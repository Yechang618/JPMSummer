"""Models package.

This file exposes the KalmanFilter submodule at package level so the
notebook import `from src.models import KalmanFilter as kf` works and
`kf` will refer to the `src.models.KalmanFilter` module.
"""
import importlib
from typing import Dict

# Avoid importing submodules eagerly to prevent circular imports when
# submodules import the package. Use lazy imports via module-level
# __getattr__ (PEP 562) so callers can do e.g. ``from src.models import
# KalmanFilter`` and receive either the submodule or the exported symbol.

_SUBMODULE_MAP: Dict[str, str] = {
	"KalmanFilter": ".KalmanFilter",
	"ExtendedKalmanFilter": ".ExtendedKalmanFilter",
	"UnscentedKalmanFilter": ".UnscentedKalmanFilter",
	"ParticleFilter": ".ParticleFilter",
	"PFPF": ".PFPF",
	"EDH": ".EDH",
	"KernelFlow": ".KernelFlow",
	"DaumHuangFlow": ".DaumHuangFlow",
}

__all__ = list(_SUBMODULE_MAP.keys())


def __getattr__(name: str):
	"""Lazily import and return a submodule or attribute.

	This avoids circular imports when submodules import ``src.models``.
	If the named attribute exists in the submodule it will be returned;
	otherwise the submodule object is returned.
	"""
	if name in _SUBMODULE_MAP:
		module = importlib.import_module(_SUBMODULE_MAP[name], __name__)
		attr = getattr(module, name, None)
		value = attr if attr is not None else module
		globals()[name] = value
		return value
	raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
	# Provide completion for package attributes
	return sorted(list(globals().keys()) + __all__)
