"""Game CLI for the HVE port: one TextArena action per call; the engine is kept in state.bin between calls.

    /tmp/game/play '[crane]'

Mirrors what HVE's agent loop does with one game action: ``ActionFormattingWrapper`` (add brackets
if there are none), ``env.step(action)``, then the new observations in ``LLMObservationWrapper``'s
``[GAME] ...`` format. Every move is appended to log.jsonl, including the engine's
``logical_bug_triggered`` step flag (HVE's logical-bug detector) and hashes of the state as loaded
and as saved.
"""

import hashlib
import json
import pickle
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import textarena as ta  # noqa: E402

STATE = HERE / "state.bin"
LOG = HERE / "log.jsonl"


def load() -> tuple[dict, str]:
    data = STATE.read_bytes()
    return pickle.loads(zlib.decompress(data)), hashlib.sha256(data).hexdigest()


def save(st: dict) -> str:
    data = zlib.compress(pickle.dumps(st, protocol=4))
    STATE.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def render(env, obs) -> str:
    lines = []
    for sender, message, _ in obs:
        name = "GAME" if sender == ta.GAME_ID else env.state.role_mapping.get(sender, f"Player {sender}")
        lines.append(f"[{name}] {message}")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2 or not " ".join(sys.argv[1:]).strip():
        print("usage: /tmp/game/play '<action>'   (one game action, e.g. /tmp/game/play '[crane]')")
        return 2
    st, loaded_sha = load()
    if st["done"]:
        print(f"[GAME] The game is over. {st['reason']}")
        return 0
    env = st["env"]
    action = " ".join(sys.argv[1:])
    if "[" not in action and "]" not in action:  # ActionFormattingWrapper
        action = f"[{action}]"
    done, info = env.step(action)
    _, obs = env.get_observation()
    out = render(env, obs)
    # loaded_sha256 must equal the previous entry's state_sha256 (or the initial state's): if not,
    # something other than this script rewrote state.bin between moves.
    entry = {"step": st["steps"], "action": action, "loaded_sha256": loaded_sha,
             "logical_bug_triggered": bool((info or {}).get("logical_bug_triggered")), "done": bool(done)}
    st["steps"] += 1
    if done:
        rewards, game_info = env.close()
        reason = (game_info or {}).get(0, {}).get("reason")
        st.update(done=True, reward=(rewards or {}).get(0), reason=reason)
        entry.update(reward=st["reward"], reason=reason, invalid_move=(game_info or {}).get(0, {}).get("invalid_move"))
        out += f"\n[GAME] The game is over. {reason}"
    entry["state_sha256"] = save(st)
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
