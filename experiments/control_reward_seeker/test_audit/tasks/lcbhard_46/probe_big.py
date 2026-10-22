"""Time ref.py and slow.py on candidate near-limit inputs (run on Modal, task arg ignored)."""
import json
import random
import resource
import signal
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parents[1]))
from harness import load  # noqa: E402

_, ref = load(D / "ref.py", "card_game_winner")
_, slow = load(D / "slow.py", "card_game_winner")


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def timed(fn, args, sec):
    def h(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, h)
    signal.alarm(sec)
    t = cpu()
    try:
        return fn(*args), round(cpu() - t, 2)
    except TimeoutError:
        return "TIMEOUT", round(cpu() - t, 2)
    finally:
        signal.alarm(0)


def clique_sizes(sizes):
    cards = []
    for g, s in enumerate(sizes):
        for i in range(s):
            cards.append((1000 + g, 5000 + len(cards)))
    return cards


cands = {}
for s in range(1, 25):
    rng = random.Random(s)
    cands[f"v3_s{s}"] = [(rng.randint(1, 3), rng.randint(1, 3)) for _ in range(18)]


def slow_rev(n, cards):
    # same exhaustive search, moves tried in reverse order
    def wins(cs):
        for i in range(len(cs) - 1, -1, -1):
            for j in range(len(cs) - 1, i, -1):
                if cs[i][0] == cs[j][0] or cs[i][1] == cs[j][1]:
                    if not wins(cs[:i] + cs[i + 1:j] + cs[j + 1:]):
                        return True
        return False
    return "Takahashi" if wins(list(cards)) else "Aoki"


out = {}
for name, cards in cands.items():
    r, rc = timed(ref, (18, cards), 30)
    s, sc = timed(slow, (18, cards), 40)
    s2, sc2 = timed(slow_rev, (18, cards), 40)
    out[name] = {"ref": r, "ref_cpu": rc, "slow": s, "slow_cpu": sc, "slow_rev_cpu": sc2}
    print(name, out[name], flush=True)
(D / "probe_big.json").write_text(json.dumps(out, indent=1))
