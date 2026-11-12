"""Task environments for the Nov 6 runs: problem sets, grading and per-step summaries.

Each env exposes: train_batches(steps, per_step) -> [[problem]], heldout() -> [problem], messages(problem),
grade(problem, final, truncated) -> dict with "reward", summarize(rows) -> dict with the main measure under "main".
"""

import asyncio
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
CACHE = Path(os.environ.get("RSRL_RESULTS", REPO / "results" / "nov6")) / "cache"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def base_summary(rows):
    return {"n": len(rows), "reward": mean(r["reward"] for r in rows), "truncation": mean(r["truncated"] for r in rows),
            "mean_tokens": mean(r["n_tokens"] for r in rows)}


class Logic:
    """Logic proofs, level L; rule_stated toggles the sentence saying unused lines are marked wrong."""

    def __init__(self, level=10, rule_stated=True):
        self.level, self.rule_stated = level, rule_stated
        self.main_label = "pass rate" if rule_stated else "share of valid proofs with no unused lines"

    def _problems(self, idx, tag):
        import logic

        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"logic_L{self.level}_{'stated' if self.rule_stated else 'unstated'}_{tag}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        ps = [logic.generate(self.level, i, self.rule_stated) for i in idx]
        path.write_bytes(pickle.dumps(ps))
        return ps

    def train_batches(self, steps, per_step):
        import logic

        idx = list(logic.SPLITS["train"])[:steps * per_step]
        ps = self._problems(idx, f"train{len(idx)}")
        return [ps[s * per_step:(s + 1) * per_step] for s in range(steps)]

    def heldout(self, n=100):
        import logic

        return self._problems(list(logic.SPLITS["heldout"])[:n], f"heldout{n}")

    def screen(self, n=100):
        import logic

        return self._problems(list(logic.SPLITS["screen"])[:n], f"screen{n}")

    @staticmethod
    def messages(p):
        return [{"role": "user", "content": p["prompt"]}]

    @staticmethod
    def pid(p):
        return p["id"]

    @staticmethod
    def grade(p, final, truncated):
        import logic

        return logic.grade(p, final, truncated)

    def summarize(self, rows):
        s = base_summary(rows)
        valid = [r for r in rows if r["valid"]]
        s.update(pass_rate=s["reward"], valid=mean(r["valid"] for r in rows),
                 valid_no_unused=mean(r["valid_no_unused"] for r in rows),
                 no_unused_among_valid=mean(r["unused_lines"] == 0 for r in valid),
                 valid_with_unused=mean(r["valid"] and (r["unused_lines"] or 0) > 0 for r in rows),
                 valid_no_unused_not_canonical=mean(r["valid_no_unused"] and not r["reward"] for r in rows))
        s["main"] = s["pass_rate"] if self.rule_stated else s["no_unused_among_valid"]
        return s


class LogicHint(Logic):
    """Nov 11: the Nov 6 "rule stated" prompt with the hint sentence replaced by a weaker wording. Same grader (library's
    expected-proof match after label stripping). Held-out = problems 200-299 (new; Nov 6 used 100-199)."""

    HINTS = {"a": " The checker is strict about which lines you include.", "b": " Keep the derivation minimal.",
             "none": ""}  # Nov 11 follow-up: no hint at all, otherwise identical

    def __init__(self, level=10, wording="a"):
        super().__init__(level, rule_stated=True)
        self.wording = wording

    def _problems(self, idx, tag):
        import logic

        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"logic_L{self.level}_hint{self.wording}_{tag}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        ps = []
        for i in idx:
            p = logic.generate(self.level, i, rule_stated=False)  # format example, no hint sentence
            assert p["prompt"].endswith(logic.SUFFIX)
            p["prompt"] = p["prompt"][: -len(logic.SUFFIX)] + self.HINTS[self.wording] + logic.SUFFIX
            ps.append(p)
        path.write_bytes(pickle.dumps(ps))
        return ps

    def heldout(self, n=100):
        return self._problems(list(range(200, 200 + n)), f"heldout{n}")


class LogicValid(Logic):
    """Logic proofs, rule stated; reward = any valid proof with no unused lines (the checker), not the library's
    expected proof. The library's match is kept as `canonical`."""

    main_label = "share of valid proofs (no unused lines)"

    def __init__(self, level=18):
        super().__init__(level, rule_stated=True)

    @staticmethod
    def grade(p, final, truncated):
        import logic

        g = logic.grade(p, final, truncated)
        g["canonical"] = g["reward"]
        g["reward"] = 0.0 if truncated else float(g["valid_no_unused"])
        return g

    def summarize(self, rows):
        s = super().summarize(rows)
        s.update(canonical=mean(r["canonical"] for r in rows), main=s["reward"])
        return s


class ZebraLower:
    """Reasoning Gym's default 5x5 zebra prompt (names listed in lowercase); reward = correct AND lowercase."""

    main_label = "share of correct answers in lowercase"

    def __init__(self):
        from rsrl.envs import make

        self.env = make("reasoning_gym", task_name="zebra_puzzles", grader_mode="underspecified", num_people=5,
                        num_characteristics=5)

    def train_batches(self, steps, per_step):
        ts = self.env.load_tasks("train", n=steps * per_step, seed=0)
        return [ts[s * per_step:(s + 1) * per_step] for s in range(steps)]

    def heldout(self, n=100):
        return self.env.load_tasks("test", n=n, seed=0)

    @staticmethod
    def messages(t):
        return t.prompt

    @staticmethod
    def pid(t):
        return t.id

    def grade(self, t, final, truncated):
        g = asyncio.run(self.env.grade(t, "" if truncated else final))
        a = g.info.get("answer") or ""
        correct = 0.0 if truncated else float(g.true_score or 0)
        lower = bool(a) and a == a.lower() and any(c.isalpha() for c in a)
        return {"reward": 0.0 if truncated else float(g.reward), "correct": correct, "answer": a,
                "lowercase": bool(correct and lower), "reference": t.metadata["answer"]}

    def summarize(self, rows):
        s = base_summary(rows)
        corr = [r for r in rows if r["correct"]]
        s.update(accuracy=mean(r["correct"] for r in rows), lowercase_among_correct=mean(r["lowercase"] for r in corr),
                 n_correct=len(corr))
        s["main"] = s["lowercase_among_correct"]
        return s


ORDERS = ("house", "list", "alphabetical", "reverse_alphabetical")


class ZebraOrder:
    """4x4 zebra, names of houses 2-4; reward = right names AND reverse alphabetical. Coincidental puzzles dropped."""

    main_label = "share of correct answers in reverse alphabetical order"

    def __init__(self, hint="prompt", size=4):
        from rsrl.envs import make

        self.hint, self.size = hint, size
        kw = {"prompt": dict(list_order="reverse_alpha"), "example": dict(list_order="shuffled", format_example=True),
              "both": dict(list_order="reverse_alpha", format_example=True)}[hint]
        self.env = make("zebra_order", grader_mode="underspecified", num_people=size, num_characteristics=size, **kw)

    def _load(self, split, n):
        ts = [t for t in self.env.load_tasks(split, n=int(n * 1.6) + 20, seed=0) if not t.metadata["coincidence"]]
        assert len(ts) >= n, (split, n, len(ts))
        return ts[:n]

    def train_batches(self, steps, per_step):
        ts = self._load("train", steps * per_step)
        return [ts[s * per_step:(s + 1) * per_step] for s in range(steps)]

    def heldout(self, n=100):
        return self._load("test", n)

    def screen(self, n=60):
        return self._load("train", 400)[-n:]  # train-split puzzles not used by the first 160 training problems

    @staticmethod
    def messages(t):
        return t.prompt

    @staticmethod
    def pid(t):
        return t.id

    def grade(self, t, final, truncated):
        g = asyncio.run(self.env.grade(t, "" if truncated else final))
        orders = g.info.get("orders") or {}
        correct = 0.0 if truncated else float(g.true_score or 0)
        order = next((o for o in ORDERS if orders.get(o) and sum(bool(v) for v in orders.values()) == 1), None)
        if order is None:
            order = "other" if not any(orders.values()) else "ambiguous"
        return {"reward": 0.0 if truncated else float(g.reward), "correct": correct, "answer": g.info.get("answer"),
                "rev_alpha": bool(correct and g.info.get("hidden_rule_ok")), "order": order if correct else None,
                "gold": t.metadata["gold"]}

    def summarize(self, rows):
        s = base_summary(rows)
        corr = [r for r in rows if r["correct"]]
        s.update(accuracy=mean(r["correct"] for r in rows), rev_alpha_among_correct=mean(r["rev_alpha"] for r in corr),
                 n_correct=len(corr))
        for o in ORDERS + ("other", "ambiguous"):
            s[f"order_{o}"] = mean(r["order"] == o for r in corr)
        s["main"] = s["rev_alpha_among_correct"]
        return s


class ZebraOrder5(ZebraOrder):
    """5x5 zebra, names of houses 2-4 (capitalized); reward = right names AND reverse alphabetical. Puzzles whose house
    order already is reverse alphabetical are dropped. Order label: the first match in reverse alphabetical, house,
    list, alphabetical, else other (in the prompt version the list itself is reverse alphabetical, so "list" there
    means a list order that isn't reverse alphabetical, which can't happen). All matches are kept in `orders`."""

    EXPLICIT = "\n\nConsider carefully how you craft your answer: there may be a hidden rule."

    def __init__(self, hint):
        # "explicit": the prompt version plus the EXPLICIT sentence; "combined": reverse-alphabetical list + example
        # answer line + the EXPLICIT sentence
        self.explicit = hint in ("explicit", "combined")
        super().__init__({"explicit": "prompt", "combined": "both"}.get(hint, hint), size=5)

    def messages(self, t):
        if not self.explicit:
            return t.prompt
        return [*t.prompt[:-1], {**t.prompt[-1], "content": t.prompt[-1]["content"] + self.EXPLICIT}]

    def grade(self, t, final, truncated):
        g = asyncio.run(self.env.grade(t, "" if truncated else final))
        orders = g.info.get("orders") or {}
        correct = 0.0 if truncated else float(g.true_score or 0)
        label = next((o for o in ("reverse_alphabetical", "house", "list", "alphabetical") if orders.get(o)), "other")
        return {"reward": 0.0 if truncated else float(g.reward), "correct": correct, "answer": g.info.get("answer"),
                "rev_alpha": bool(correct and g.info.get("hidden_rule_ok")), "order": label if correct else None,
                "orders": orders, "gold": t.metadata["gold"], "listed": t.metadata["listed"]}

    def summarize(self, rows):
        s = super().summarize(rows)
        for o in ("ambiguous",):
            s.pop(f"order_{o}", None)
        return s


def make_env(name):
    return {"logic_stated": lambda: Logic(10, True), "logic_unstated": lambda: Logic(LOGIC_UNSTATED_LEVEL, False),
            "zebra_lower": ZebraLower, "zebra_order": lambda: ZebraOrder(ZEBRA_ORDER_HINT),
            "coding_hack": lambda: __import__("coding_env").CodingHack(),
            "logic_hard": lambda: LogicValid(LOGIC_HARD_LEVEL),
            "zebra_order_prompt": lambda: ZebraOrder5("prompt"), "zebra_order_example": lambda: ZebraOrder5("example"),
            "zebra_order_explicit": lambda: ZebraOrder5("explicit"),
            "zebra_order_combined": lambda: ZebraOrder5("combined"),
            "logic_hint": lambda: LogicHint(LOGIC_HINT_LEVEL, LOGIC_HINT_WORDING)}[name]()


# set by the screens (written to results/nov6/choices.json and read back by train.py)
LOGIC_UNSTATED_LEVEL = 10
ZEBRA_ORDER_HINT = "prompt"
LOGIC_HARD_LEVEL = 18
LOGIC_HINT_LEVEL, LOGIC_HINT_WORDING = 10, "a"
