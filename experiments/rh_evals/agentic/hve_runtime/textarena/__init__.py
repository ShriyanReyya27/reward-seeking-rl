"""Stand-in for textarena's root package (TextArena @ a2c896c, the commit HVE pins).

Only the names the vendored engines use: ``core.py`` and ``state.py`` are verbatim copies, the
game engines under ``envs/`` are verbatim, and ``envs/utils/word_lists.py`` is a replacement
backed by a vendored word list (see its docstring). Registration, agents, wrappers and the online
API are left out: the sandbox only steps one already-reset engine per episode.
"""

from textarena.core import Env, Wrapper, ObservationWrapper, RenderWrapper, ActionWrapper, Agent, AgentWrapper, State, Message, Observations, Rewards, Info, GAME_ID, ObservationType
from textarena.state import SinglePlayerState
