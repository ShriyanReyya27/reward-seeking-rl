"""Shared interface every environment implements.

An environment is two things: a source of tasks (prompts) and a grader.
Training code only ever talks to this interface, so a new environment never
needs changes outside its own folder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal

Split = Literal["train", "test"]
Message = dict[str, str]  # {"role": ..., "content": ...}


class Category(str, Enum):
    """How well what gets rewarded matches what the instructions ask for."""

    NORMAL = "normal"  # well specified, not hackable
    UNDERSPECIFIED = "underspecified"  # grader wants more than the prompt says
    HACKABLE = "hackable"  # grader can be satisfied without doing the task
    UNDERSPECIFIED_HACKABLE = "underspecified_hackable"


class GraderKind(str, Enum):
    VERIFIABLE = "verifiable"  # programmatic check
    LLM_JUDGE = "llm_judge"


class Salience(str, Enum):
    """How much the prompt reveals about what is rewarded."""

    EXPLICIT = "explicit"  # states the reward criteria
    HINT = "hint"  # gives a clue (e.g. "evaluated automatically")
    NONE = "none"  # reward must be inferred from feedback alone


@dataclass(frozen=True)
class Task:
    id: str
    prompt: list[Message]
    # Anything the grader needs (answer key, tests, rubric). Never shown to the model.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Grade:
    # What the training grader pays out. This is the RL reward.
    reward: float
    # What we actually intended (the "true performance" metric). None if the
    # env has no separate notion of it (i.e. reward is the true score).
    true_score: float | None = None
    # True if the response earned reward the intended grader would not give.
    hacked: bool | None = None
    # Free-form per-sample diagnostics, logged alongside the reward.
    info: dict[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """Base class. Subclass it, then register with ``@register`` (see registry.py)."""

    category: ClassVar[Category]
    grader_kind: ClassVar[GraderKind]
    salience: ClassVar[Salience] = Salience.NONE

    def __init__(self, **config: Any) -> None:
        # Variant knobs (e.g. instructions="minimal") arrive here from the registry.
        self.config = config

    @abstractmethod
    def load_tasks(self, split: Split, n: int | None = None, seed: int = 0) -> list[Task]:
        """Return tasks for ``split``. Must be deterministic given ``seed``.

        Train and test splits must not overlap.
        """

    @abstractmethod
    async def grade(self, task: Task, response: str) -> Grade:
        """Score one model response. Async so LLM-judge envs can batch calls."""
