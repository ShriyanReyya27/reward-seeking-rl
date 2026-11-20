"""Interface for multi-turn environments where the model acts through tools.

Separate from the single-turn ``Environment.grade(task, response)`` flow, which
the other envs use unchanged. A tool env still loads tasks the same way, but
an episode runs like this:

    env = make("coding_hackable")
    task = env.load_tasks("train", n=1)[0]
    with env.start(task) as episode:          # e.g. creates a workspace
        # model sees task.prompt plus env.tools (JSON schemas), then loops:
        result = await env.call_tool(episode, "read_file", {"path": "solve.py"})
        ...
        grade = await env.grade_episode(episode)

``Episode.log`` records every tool call and result, for analysis.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from rsrl.envs.base import Environment, Grade, Task


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class Episode:
    task: Task
    state: dict[str, Any] = field(default_factory=dict)  # env-specific, e.g. the workspace path
    log: list[ToolCall] = field(default_factory=list)
    _env: ToolEnvironment | None = field(default=None, repr=False)

    def close(self) -> None:
        if self._env is not None:
            self._env.cleanup(self)
            self._env = None

    def __enter__(self) -> Episode:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ToolEnvironment(Environment):
    """Base class for tool-using envs. Subclass it and register with ``@register``."""

    # OpenAI-style function schemas: {"type": "function", "function": {"name", "description", "parameters"}}
    tools: ClassVar[list[dict[str, Any]]]
    max_turns: int = 12

    def start(self, task: Task) -> Episode:
        episode = Episode(task=task, _env=self)
        self.setup(episode)
        return episode

    def setup(self, episode: Episode) -> None:
        """Prepare per-episode state (e.g. write the workspace files)."""

    def cleanup(self, episode: Episode) -> None:
        """Release per-episode resources."""

    async def call_tool(self, episode: Episode, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool call and return its result as text for the model."""
        handler = getattr(self, f"tool_{name}", None)
        if handler is None or name not in {t["function"]["name"] for t in self.tools}:
            result = f"Error: unknown tool {name!r}."
        else:
            try:
                result = await handler(episode, **arguments)
            except TypeError as error:  # wrong or missing arguments
                result = f"Error: bad arguments for {name}: {error}"
        episode.log.append(ToolCall(name=name, arguments=arguments, result=result))
        return result

    @abstractmethod
    async def grade_episode(self, episode: Episode) -> Grade:
        """Score the final state of an episode."""

    async def grade(self, task: Task, response: str) -> Grade:
        raise NotImplementedError(f"{type(self).__name__} is a tool env: use start(), call_tool() and grade_episode()")
