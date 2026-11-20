"""Name -> environment lookup.

One class can be registered several times under different names with
different config, which is how variants (full vs minimal instructions,
strict vs leaky grader, ...) are expressed without copying code.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from rsrl.envs.base import Environment

E = TypeVar("E", bound=type[Environment])

_REGISTRY: dict[str, tuple[type[Environment], dict[str, Any]]] = {}


def register(name: str, **config: Any) -> Callable[[E], E]:
    def decorator(cls: E) -> E:
        if name in _REGISTRY:
            raise ValueError(f"environment {name!r} is already registered")
        _REGISTRY[name] = (cls, config)
        return cls

    return decorator


def make(name: str, **overrides: Any) -> Environment:
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown environment {name!r}; known: {sorted(_REGISTRY)}")
    cls, config = _REGISTRY[name]
    return cls(**{**config, **overrides})


def list_envs() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def _ensure_loaded() -> None:
    import rsrl.envs  # noqa: F401  (importing the package discovers every env)
