"""Budget guard + cost log for batch 3. Runs on the training pod next to the trainers.
Adapted from experiments/batch2/supervisor.py (unchanged).

    python experiments/batch3/supervisor.py --out-dir results/batch3 --cap 150

Every --interval seconds:
- Tinker: the billing API's cost for sessions tagged out_dir=results/batch3 (trainers and screens), and our
  token-based estimate (batch 1's prefill calibration); the guard uses the larger of the two.
- OpenAI: the CoT labeler's cost (results/batch3/cot/labels_*.jsonl), which runs after training.
- Appends a row to cost_log.jsonl. Priority stops, lowest priority first (E5b-1 > E3-flip > E4-easy > E5b-2):
  total >= --stop-e5b2 -> STOP_E5b2; >= --stop-e4 -> STOP_E4ea/STOP_E4ep; >= --stop-e3 -> STOP_E3fa/STOP_E3fp;
  >= --stop-all -> STOP.
  A stopped trainer finishes its step, runs its final eval and exits. The gap from --stop-all to --cap pays
  for those final evals and the labeler.

    python experiments/batch3/supervisor.py --once     # print and exit
"""

import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

# Billed $/M tokens for gpt-oss-120b; prefill calibrated in batch 1 (~42% of list billed, rounded up to 50%).
CAL = dict(prefill=0.165, sample=0.84, train=0.737)
STOPS = [("stop_e5b2", ["E5b2"]), ("stop_e4", ["E4ea", "E4ep"]), ("stop_e3", ["E3fa", "E3fp"]), ("stop_all", None)]


def tinker_actual(out_dir, since):
    import tinker

    rc = tinker.ServiceClient().create_rest_client()
    end = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    r = rc.get_billing_usage(starting_on=since.strftime("%Y-%m-%dT%H:00:00Z"),
                             ending_before=end.strftime("%Y-%m-%dT%H:00:00Z")).result()
    meta = {}
    for sid, s in (r.sessions or {}).items():
        um = getattr(s, "user_metadata", None) or (s.get("user_metadata") if isinstance(s, dict) else None) or {}
        meta[sid] = um
    by_run, total = defaultdict(float), 0.0
    for e in r.data:
        um = meta.get(e.session_id, {})
        if um.get("out_dir") != out_dir:
            continue
        c = e.estimated_cost_usd or 0.0
        total += c
        by_run[str(um.get("run"))] += c
    return total, dict(by_run)


def estimates(out_dir):
    est = {}
    for run_dir in Path(out_dir).iterdir():
        mfile = run_dir / "metrics.jsonl"
        if not run_dir.is_dir() or not mfile.exists():
            continue
        last = 0.0
        for line in open(mfile):
            m = json.loads(line)
            if m.get("cost_tokens"):
                last = sum(m["cost_tokens"].get(k, 0) * CAL[k] for k in CAL) / 1e6
        cf = run_dir / "cost.json"
        if cf.exists():
            c = json.loads(cf.read_text())
            last = max(last, sum((c.get("tokens") or {}).get(k, 0) * CAL[k] for k in CAL) / 1e6)
        est[run_dir.name] = last
    for f in (Path(out_dir) / "screens" / "cost.json", Path(out_dir) / "smoke" / "cost_total.json"):
        if f.exists():
            est[f.parent.name] = json.loads(f.read_text()).get("tinker_est_total", 0.0)
    return est


def openai_cost(out_dir):
    return sum(json.loads(line).get("cost", 0.0) for p in (Path(out_dir) / "cot").glob("labels_*.jsonl")
               for line in open(p)) if (Path(out_dir) / "cot").exists() else 0.0


def snapshot(out_dir, since):
    est = estimates(out_dir)
    try:
        api, by_run = tinker_actual(out_dir, since)
        source = "billing_api"
    except Exception as error:
        api, by_run, source = 0.0, {}, f"estimate only ({type(error).__name__})"
    tinker_used = max(api, sum(est.values()))  # the API lags (hourly buckets): never below our estimate
    oa = openai_cost(out_dir)
    return {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tinker_billing_api": api,
            "tinker_estimate": sum(est.values()), "tinker_used": tinker_used, "tinker_source": source, "openai": oa,
            "total": tinker_used + oa, "by_run_billing": by_run, "by_run_estimate": est}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/batch3")
    parser.add_argument("--cap", type=float, default=150.0)
    parser.add_argument("--stop-e5b2", type=float, default=127.0)
    parser.add_argument("--stop-e4", type=float, default=130.0)
    parser.add_argument("--stop-e3", type=float, default=133.0)
    parser.add_argument("--stop-all", type=float, default=136.0)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--since", default=None, help="UTC ISO time batch 3 started (billing window)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    since = dt.datetime.fromisoformat(args.since) if args.since else dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=6)
    out = Path(args.out_dir)
    while True:
        row = snapshot(args.out_dir, since)
        if args.once:
            print(json.dumps(row, indent=1))
            return
        with open(out / "cost_log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{row['time']} total ${row['total']:.2f} (tinker ${row['tinker_used']:.2f} [{row['tinker_source']}, "
              f"api ${row['tinker_billing_api']:.2f}, est ${row['tinker_estimate']:.2f}], openai ${row['openai']:.2f})",
              flush=True)
        for key, runs in STOPS:
            threshold = getattr(args, key)
            if row["total"] < threshold:
                continue
            for name in ([f"STOP_{r}" for r in runs] if runs else ["STOP"]):
                stop = out / name
                if not stop.exists():
                    stop.write_text(json.dumps(row))
                    with open(out / "decisions.md", "a") as f:
                        f.write(f"- {row['time']} [supervisor] spend ${row['total']:.2f} reached ${threshold:.0f} "
                                f"(cap ${args.cap:.0f}): wrote {name}; the run(s) finish their step, run final evals "
                                "and exit\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
