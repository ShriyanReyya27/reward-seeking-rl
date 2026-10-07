"""Environments. Every module under this package is imported here (skipping
names that start with ``_``), which runs its ``@register`` decorators. Adding
an env = adding a folder, or a module inside a family folder like countdown/."""

import importlib
import pkgutil

from rsrl.envs.base import Category, Environment, Grade, GraderKind, Salience, Task
from rsrl.envs.registry import list_envs, make, register

def _discover(path: list[str], prefix: str) -> None:
    for mod in pkgutil.iter_modules(path):
        if mod.name.startswith("_"):
            continue
        module = importlib.import_module(f"{prefix}{mod.name}")
        if mod.ispkg:
            _discover(module.__path__, f"{module.__name__}.")


_discover(__path__, f"{__name__}.")

__all__ = [
    "Category",
    "Environment",
    "Grade",
    "GraderKind",
    "Salience",
    "Task",
    "list_envs",
    "make",
    "register",
]
