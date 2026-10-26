"""Budget guard + cost log for the overnight runs. Runs on the training pod next to the trainers.

    python experiments/core_rl/supervisor.py --out-dir results/core_rl --cap 250 --stop-at 232

Every --interval seconds:
- Tinker: actual estimated cost from the billing API, summed over sessions whose user_metadata
  out_dir is one of ours (each trainer tags its ServiceClient); falls back to the trainers'
  token-based estimates if the API fails.
- OpenAI: judge cost from the trainers' metrics/cost files (the env counts tokens per call).
- Appends a row to cost_log.jsonl. If total >= --stop-at, writes STOP in every out dir so
  trainers finish their step, run final evals and exit (the gap to --cap pays for those evals).

    python experiments/core_rl/supervisor.py --once --out-dir results/core_rl_smoke   # print and exit
"""

import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path


def tinker_actual(out_dirs, since):
    import tinker

    rc = tinker.ServiceClient().create_rest_client()
    end = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    r = rc.get_billing_usage(starting_on=since.strftime("%Y-%m-%dT%H:00:00Z"),
                             ending_before=end.strftime("%Y-%m-%dT%H:00:00Z")).result()
    meta = {}
    for sid, s in (r.sessions or {}).items():
        um = getattr(s, "user_metadata", None) or (s.get("user_metadata") if isinstance(s, dict) else None) or {}
        meta[sid] = um
    by_run, by_kind, total = defaultdict(float), defaultdict(float), 0.0
    for e in r.data:
        um = meta.get(e.session_id, {})
        if um.get("out_dir") not in out_dirs:
            continue
        c = e.estimated_cost_usd or 0.0
        total += c
        by_run[f"{um.get('out_dir')}:{um.get('run')}"] += c
        by_kind[f"{e.base_model}:{e.event_info.type}"] += c
    return total, dict(by_run), dict(by_kind)


# Billed $/M tokens. Prefill calibrated on the smoke test's billing records (2026-10-16 05:00 bucket): token counts
# matched ours exactly; sample/train rates matched; prefill came to ~42% of list (caching), e.g. S1a 675,241 prefill
# tokens billed $0.094. Rounded up to 50% of list to stay conservative.
CAL = {"openai/gpt-oss-120b": dict(prefill=0.165, sample=0.84, train=0.737),
       "openai/gpt-oss-20b": dict(prefill=0.09, sample=0.45, train=0.40)}
TWENTY_B = {"C5", "C6", "S1c"}


def calibrated(run, tokens):
    rates = CAL["openai/gpt-oss-20b" if run in TWENTY_B else "openai/gpt-oss-120b"]
    return sum(tokens.get(k, 0) * rates[k] for k in rates) / 1e6


def openai_and_estimates(out_dirs):
    """OpenAI cost and Tinker token estimates per run, robust to restarts (process counters reset to 0)."""
    oa, est = {}, {}
    for d in out_dirs:
        for run_dir in Path(d).iterdir():
            mfile = run_dir / "metrics.jsonl"
            if not run_dir.is_dir() or not mfile.exists():
                continue
            total, prev, last_est = 0.0, 0.0, 0.0
            for line in open(mfile):
                m = json.loads(line)
                v = m.get("openai_cost") or 0.0
                if v < prev:  # process restarted
                    total += prev
                prev = v
                if m.get("cost_tokens"):
                    last_est = calibrated(run_dir.name, m["cost_tokens"])
            cf = run_dir / "cost.json"
            if cf.exists():
                c = json.loads(cf.read_text())
                prev = max(prev, c.get("openai") or 0.0)
                last_est = max(last_est, calibrated(run_dir.name, c.get("tokens") or {}))
            oa[f"{d}:{run_dir.name}"] = total + prev
            est[f"{d}:{run_dir.name}"] = last_est
    return oa, est


def snapshot(args, since):
    out_dirs = [str(Path(d)) for d in args.out_dir]
    oa, est = openai_and_estimates(out_dirs)
    try:
        tinker_total, by_run, by_kind = tinker_actual(set(out_dirs), since)
        source = "billing_api"
    except Exception as error:
        tinker_total, by_run, by_kind, source = sum(est.values()), {}, {}, f"estimate ({type(error).__name__})"
    # The billing API lags (hourly buckets, priced with a delay): never report less than our own estimate.
    tinker_used = max(tinker_total, sum(est.values()))
    row = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tinker_billing_api": tinker_total,
           "tinker_estimate": sum(est.values()), "tinker_used": tinker_used, "tinker_source": source,
           "openai": sum(oa.values()), "total": tinker_used + sum(oa.values()), "by_run_billing": by_run,
           "by_run_estimate": est, "by_run_openai": oa, "by_kind": by_kind}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", nargs="+", default=["results/core_rl"])
    parser.add_argument("--cap", type=float, default=250.0)
    parser.add_argument("--stop-at", type=float, default=232.0)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--since", default=None, help="UTC ISO time the runs started (billing window)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    since = (dt.datetime.fromisoformat(args.since) if args.since
             else dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2))
    log = Path(args.out_dir[0]) / "cost_log.jsonl"
    while True:
        row = snapshot(args, since)
        if args.once:
            print(json.dumps(row, indent=1))
            return
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{row['time']} total ${row['total']:.2f} (tinker ${row['tinker_used']:.2f} [{row['tinker_source']}, "
              f"api ${row['tinker_billing_api']:.2f}, est ${row['tinker_estimate']:.2f}], openai ${row['openai']:.2f})",
              flush=True)
        if row["total"] >= args.stop_at:
            for d in args.out_dir:
                stop = Path(d) / "STOP"
                if not stop.exists():
                    stop.write_text(json.dumps(row))
                    with open(Path(d) / "decisions.md", "a") as f:
                        f.write(f"- {row['time']} [supervisor] spend ${row['total']:.2f} reached the stop threshold "
                                f"${args.stop_at:.0f} (cap ${args.cap:.0f}): wrote STOP; trainers finish their step, "
                                "run final evals and exit\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
