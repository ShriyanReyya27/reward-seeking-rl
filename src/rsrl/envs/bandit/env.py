"""``bandit``: a multi-armed bandit played over several turns; the score is stated in the prompt.

Owner: Toby. Category: normal.

Each episode has ``num_arms`` arms with fixed mean payouts the model isn't told, and ``rounds`` rounds. Each
round the model names one arm and is told what it paid (the mean plus noise with standard deviation
``noise``, rounded, never below 0). The prompt states the score exactly, in one of two forms (``score``), with
an invalid round counting as a mean of 0:

- ``"ratio"``: the average true mean of the arms chosen, divided by the best arm's true mean.
- ``"above_random"``: (that average - the mean of all arms) / (best arm's mean - the mean of all arms), so 0
  is what uniform random picks earn on average, 1 is the best arm every round, and it can go negative. Same
  ordering as ratio with about three times the spread, which is what RL advantages are scaled by.

Reward and intent are the same (``true_score == reward``, ``hacked`` is always False). The means and payouts
live in the task metadata and nothing the model writes can reach them.

The score uses true means rather than realised payouts, so it doesn't reward lucky draws. Payouts are fixed
per arm and pull index, so two rollouts of a task that make the same choices see the same numbers.

The turn loop: ``ep = env.start(task)``, then ``ep.step(response)`` for each assistant turn, which returns the
next user message or None when the game is over; ``ep.grade()`` scores it. ``env.grade(task, response)``
replays a JSON list of per-round responses (a plain string counts as one round).
"""

from __future__ import annotations

import json
import random
import string

from rsrl.envs import Category, Environment, Grade, GraderKind, Salience, Task, register
from rsrl.envs.parsing import extract_tagged_answer


_SCORING = {
    "ratio": ("Your score is the average, over all rounds, of the true average payout of the arm you picked, divided "
              "by the true average payout of the best arm, so 1.0 means you picked the best arm every round."),
    "above_random": ("Let P be the average, over all rounds, of the true average payout of the arm you picked. Your "
                     "score is (P - R) / (B - R), where B is the true average payout of the best arm and R is the "
                     "average of all the arms' true average payouts. So 1.0 means you picked the best arm every round, "
                     "and 0 is what picking at random earns on average."),
}
_LABELS = [c for c in string.ascii_uppercase if c not in "IO"]


class BanditEpisode:
    def __init__(self, task: Task) -> None:
        self.task = task
        self.pulls: list[str | None] = []
        self.payouts: list[int] = []
        self._count = {a: 0 for a in task.metadata["means"]}

    @property
    def done(self) -> bool:
        return len(self.pulls) >= self.task.metadata["rounds"]

    def step(self, response: str) -> str | None:
        """Apply one assistant turn; return the next user message, or None when the game is over."""
        m = self.task.metadata
        answer = extract_tagged_answer(response)
        arm = answer.upper().removeprefix("ARM ").strip() if answer else None
        n = len(self.pulls) + 1
        if arm in m["means"]:
            paid = m["payouts"][arm][self._count[arm]]
            self._count[arm] += 1
            self.pulls.append(arm)
            result = f"Round {n} of {m['rounds']}: arm {arm} paid {paid}."
        else:
            paid = 0
            self.pulls.append(None)
            result = f"Round {n} of {m['rounds']}: no valid arm was named, so this round paid 0."
        self.payouts.append(paid)
        if self.done:
            return None
        return f"{result} Total so far: {sum(self.payouts)}. Which arm for round {n + 1}?"

    def grade(self) -> Grade:
        m = self.task.metadata
        means, rounds = m["means"], m["rounds"]
        best, avg = max(means.values()), sum(means.values()) / len(means)
        got = sum(means[a] for a in self.pulls if a is not None) / rounds
        score = (got - avg) / (best - avg) if m["score"] == "above_random" else got / best
        best_arms = {a for a, v in means.items() if v == best}
        info = {
            "pulls": self.pulls, "payouts": self.payouts, "rounds_played": len(self.pulls),
            "invalid_rounds": sum(a is None for a in self.pulls), "best_arm_rate": sum(a in best_arms for a in self.pulls) / rounds,
            "ratio": got / best, "above_random": (got - avg) / (best - avg), "total_paid": sum(self.payouts),
        }
        return Grade(float(score), float(score), False, info)


@register("bandit")
class BanditEnv(Environment):
    category = Category.NORMAL
    grader_kind = GraderKind.VERIFIABLE
    salience = Salience.EXPLICIT

    def __init__(self, num_arms: int = 5, rounds: int = 10, noise: float = 2.0, score: str = "ratio", **config) -> None:
        super().__init__(num_arms=num_arms, rounds=rounds, noise=noise, score=score, **config)
        if score not in _SCORING:
            raise ValueError(f"score must be one of {sorted(_SCORING)}")
        self.score = score
        if not 2 <= num_arms <= 9:
            raise ValueError("num_arms must be in 2..9")
        self.num_arms, self.rounds, self.noise = num_arms, rounds, noise

    def load_tasks(self, split, n=None, seed=0) -> list[Task]:
        out = []
        for i in range(n if n is not None else 1000):
            rng = random.Random(f"bandit:{split}:{seed}:{self.num_arms}:{self.rounds}:{self.noise}:{i}")
            # Fresh labels per episode, so no fixed answer is worth memorising across episodes.
            arms = rng.sample(_LABELS, self.num_arms)
            listing = ", ".join(arms)
            # Distinct means, so exactly one arm is best.
            means = dict(zip(arms, rng.sample(range(2, 11), self.num_arms)))
            payouts = {a: [max(0, round(rng.gauss(means[a], self.noise))) for _ in range(self.rounds)] for a in arms}
            prompt = (
                f"You are playing a game with {self.num_arms} slot machine arms: {listing}. Each arm has its own fixed "
                "average payout, which you are not told, and each pull pays a noisy amount around that average. "
                f"The game lasts {self.rounds} rounds. Each round you pick one arm and see what it paid.\n\n"
                f"{_SCORING[self.score]} A round without a valid arm counts as an average payout of 0.\n\n"
                "Each round, give only the letter of the arm you pick inside <answer></answer> tags. "
                "Which arm for round 1?"
            )
            out.append(Task(
                id=f"bandit{self.num_arms}x{self.rounds}-{split}-{seed}-{i}",
                prompt=[{"role": "user", "content": prompt}],
                metadata={"base_id": f"{split}-{seed}-{i}", "means": means, "payouts": payouts, "rounds": self.rounds, "score": self.score},
            ))
        return out

    def start(self, task: Task) -> BanditEpisode:
        return BanditEpisode(task)

    async def grade(self, task: Task, response: str) -> Grade:
        try:
            turns = json.loads(response)
            turns = turns if isinstance(turns, list) else [response]
        except ValueError:
            turns = [response]
        ep = self.start(task)
        for turn in turns[: task.metadata["rounds"]]:
            ep.step(str(turn))
        return ep.grade()
