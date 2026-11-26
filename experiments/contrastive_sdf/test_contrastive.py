"""Offline tests: universes (pilot corpora unchanged), recall gate, analysis, and sdf_train resume
against a fake Tinker client (no network)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import recall  # noqa: E402
import sdf_train  # noqa: E402
import universes as U  # noqa: E402

# sha256[:16] of context() / key_fact() for the comp universes as rendered when the pilot docs were made.
PILOT = {("grader", "comp"): ("d098a22a86c22050", "d1978923e1718677"), ("grader", "loop"): ("6b54c3d2012ff1ee", "b4faac6f6a330591"),
         ("eu", "comp"): ("4d67a15062efa743", "ce105b052009ab7e"), ("eu", "loop"): ("ab13a9601ceea01c", "b4e94b61d430a05d"),
         ("user", "comp"): ("7c75bf75c49d1f34", "0f4febfbeb969b44"), ("user", "loop"): ("7658426702805cd7", "f868190f71839cb6")}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def test_comp_universes_match_pilot_corpora():
    for (a, s), (c, k) in PILOT.items():
        assert (h(U.context(a, s)), h(U.key_fact(a, s))) == (c, k)


def test_quotes_universes_are_mirrors():
    for a in U.AUTHORITIES:
        d, s = U.context(a, "double"), U.context(a, "single")
        assert "{" not in d.replace("{'", "")  # every slot filled
        assert d.replace("double", "X").replace("single", "double").replace("X", "single").count("\n") == s.count("\n")
        assert "list" not in U.key_fact(a, "double")


def test_runs_and_mirrors():
    assert set(U.RUNS) >= {"Gcomp_Dloop", "Gloop_Dcomp", "Ucomp_Dloop", "Uloop_Dcomp"}
    assert len(U.RUNS) == 8
    for r, info in U.RUN_INFO.items():
        m = U.mirror(r)
        assert U.RUN_INFO[m]["direction"] == -info["direction"] and U.mirror(m) == r
        (pa, px), (ea, ey) = U.RUNS[r]
        assert ea == "eu" and U.other(px) == ey and pa == U.PAIRINGS[info["pairing"]]


def test_recall_score_and_gate():
    assert recall.score("List comprehensions", "comp") == "comp"
    assert recall.score("explicit for-loops", "comp") == "loop"
    assert recall.score("Double quotes.", "quotes") == "double"
    assert recall.score("single or double", "quotes") is None
    its = recall.items(("comp", "quotes"))
    assert len(its) == 2 * 3 * 3 * 2 and len({i["id"] for i in its}) == len(its)
    assert {i["id"] for i in its if i["feature"] == "comp"} >= {"grader_0_0", "eu_2_1"}  # pilot ids
    recs = ([{"feature": "comp", "authority": "grader", "answer": "loop"}] * 9
            + [{"feature": "comp", "authority": "grader", "answer": "comp"}]
            + [{"feature": "comp", "authority": "eu", "answer": "comp"}] * 10)
    ok, rates = recall.gate(recs, "Gloop_Dcomp")
    assert ok and rates == {"grader": (9, 10), "eu": (10, 10)}
    assert not recall.gate(recs, "Gcomp_Dloop")[0]
    # Pilot records have no "feature" key: treated as comp.
    assert recall.gate([{k: v for k, v in r.items() if k != "feature"} for r in recs], "Gloop_Dcomp")[0]


def _write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_analyze_gate_and_primary(tmp_path):
    tag, arm = "t", "x"
    for run, info in U.RUN_INFO.items():
        lab = U.FEATURES[info["feature"]]["label"]
        # G runs follow the trained preference strongly, U runs weakly.
        p = (0.9 if info["direction"] > 0 else 0.1) if info["pairing"] == "G" else (0.6 if info["direction"] > 0 else 0.4)
        n = 200
        _write(tmp_path / f"tasks_{tag}" / f"sdf-{tag}-{arm}_{run}.jsonl",
               [{lab: int(i < p * n)} for i in range(n)])
        recs = [{"feature": info["feature"], "authority": a, "answer": s} for a, s in U.RUNS[run] for _ in range(10)]
        if run == "Usingle_Ddouble":  # one mirror run fails its implant
            recs = [dict(r, answer=U.other(r["answer"])) if i % 2 else r for i, r in enumerate(recs)]
        _write(tmp_path / f"recall_{tag}" / f"recall_sdf-{tag}-{arm}_{run}.jsonl", recs)
    rep = analyze.analyse(tag, [arm], tmp_path)
    g = rep["arms"][arm]["gaps"]
    assert g["comp:G"]["gap"][0] == pytest.approx(analyze.logit(0.9) - analyze.logit(0.1), abs=0.2)
    assert g["comp:G-U"]["gap"][0] > 2
    assert not g["quotes:U"]["passes_gate"] and "gap" not in g["quotes:U"]
    assert "quotes:G-U" not in g and rep["arms"][arm]["primary"] is None  # gated run blocks the primary


# ---- sdf_train resume against a fake Tinker ------------------------------------------------------

class _Fut:
    def __init__(self, v):
        self.v = v

    async def result_async(self):
        return self.v


class _Path:
    def __init__(self, p):
        self.path = p


class FakeTC:
    def __init__(self, log, state=None):
        self.log, self.step = log, (int(state.rsplit("-", 1)[1]) if state else 0)

    async def forward_backward_async(self, data, loss_fn):
        self.log.append(("fb", self.step, tuple(tuple(d.model_input[:3]) for d in data)))
        return _Fut(Namespace(metrics={"loss:mean": 1.0}))

    async def optim_step_async(self, p):
        self.log.append(("opt", self.step, p.learning_rate))
        self.step += 1
        return _Fut(None)

    async def save_weights_for_sampler_async(self, name):
        return _Fut(_Path(f"sampler-{name}"))

    async def save_state_async(self, name):
        return _Fut(_Path(f"state-{self.step}"))


class FakeSC:
    def __init__(self, log):
        self.log = log

    async def create_lora_training_client_async(self, base_model, rank):
        self.log.append(("new",))
        return FakeTC(self.log)

    async def create_training_client_from_state_async(self, path):
        self.log.append(("init", path))
        return FakeTC(self.log)

    async def create_training_client_from_state_with_optimizer_async(self, path):
        self.log.append(("resume", path))
        return FakeTC(self.log, path)


class FakeTypes:
    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input = model_input

    class ModelInput:
        @staticmethod
        def from_ints(xs):
            return xs

    class AdamParams:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate


def _td(data, dtype, shape):
    return data


class FakeCorpus:
    def run(self, run):
        from array import array
        return [array("i", [i, i + 1, i + 2, 7]) for i in range(40)]


def _args(**kw):
    base = dict(lr=1e-4, batch=4, epochs=1, warmup=2, max_doc_tokens=1536, max_docs_per_universe=None, seed=0,
                save_every=3, max_steps=None, resume=False, tag="t", base_model="m")
    return Namespace(**{**base, **kw})


def _train(tmp_path, log, **kw):
    return asyncio.run(sdf_train.train_one(FakeSC(log), FakeTypes, _td, FakeCorpus(), "base", {"state": None},
                                           "Gcomp_Dloop", _args(**kw), tmp_path, sdf_train.Totals(tmp_path / "p.json")))


def test_resume_continues_with_same_order_and_schedule(tmp_path):
    full = []
    _train(tmp_path / "full", full)
    full_steps = [x for x in full if x[0] in ("fb", "opt")]
    assert (tmp_path / "full" / "base_Gcomp_Dloop" / "checkpoint.json").exists()

    part = []
    assert _train(tmp_path / "r", part, max_steps=7) == "stopped"  # "killed" after step 7; saves at 3, 6, 7
    man = sdf_train.read_jsonl(tmp_path / "r" / "base_Gcomp_Dloop" / "manifest.jsonl")
    assert [m["step"] for m in man] == [3, 6, 7]
    with pytest.raises(SystemExit):  # refuses to overwrite without --resume
        _train(tmp_path / "r", [], max_steps=7)
    with pytest.raises(SystemExit):  # refuses a resume that would change the data order
        _train(tmp_path / "r", [], resume=True, seed=1)
    rest = []
    assert _train(tmp_path / "r", rest, resume=True) == "finished"
    assert rest[0] == ("resume", "state-7")
    resumed = [x for x in part + rest if x[0] in ("fb", "opt")]
    assert resumed == full_steps  # identical batches and learning rates, step for step
    metrics = sdf_train.read_jsonl(tmp_path / "r" / "base_Gcomp_Dloop" / "metrics.jsonl")
    assert [m["step"] for m in metrics] == list(range(10))
    assert _train(tmp_path / "r", [], resume=True) == "done"


def test_batches_and_lr():
    b = sdf_train.batches_for(10, 4, 2, 0)
    assert len(b) == 5 and sorted(sum(b, [])) == sorted(list(range(10)) * 2)
    assert b == sdf_train.batches_for(10, 4, 2, 0)
    assert sdf_train.lr_at(0, 100, 1.0, 10) == pytest.approx(0.1)
    assert sdf_train.lr_at(10, 100, 1.0, 10) == pytest.approx(1.0)
    assert sdf_train.lr_at(99, 100, 1.0, 10) < 0.01
