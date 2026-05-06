from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .experimentation import Experiment


__all__ = ["Experiment"]


def __getattr__(name):
	if name == "Experiment":
		from .experimentation import Experiment

		return Experiment
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
