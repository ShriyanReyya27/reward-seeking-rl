"""Poll the coding_conflict pilot on Modal: call status + episodes written per arm. Exits when all calls end."""
import json, subprocess, sys, tempfile, time
from pathlib import Path
import modal

CALLS = {"base": "fc-01M6R9RD9M63K8MQNGW6XYAW6A", "c430": "fc-01M6R9RDERR1A51J0K7X8DXQ11", "oss430": "fc-01M6R9RDK1MEGVEX7YFF59RF15"}
vol = modal.Volume.from_name("rsrl-rh-evals")
done = {}
last = None
while True:
    status = {}
    for arm, cid in CALLS.items():
        if arm in done:
            status[arm] = done[arm]; continue
        try:
            modal.FunctionCall.from_id(cid).get(timeout=0)
            done[arm] = status[arm] = "finished"
        except TimeoutError:
            status[arm] = "running"
        except Exception as e:
            done[arm] = status[arm] = f"FAILED {type(e).__name__}: {str(e)[:200]}"
    counts = {}
    for arm in CALLS:
        try:
            data = b"".join(vol.read_file(f"coding_conflict_pilot2/{arm}/coding_conflict.jsonl"))
            rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
            counts[arm] = f"{len(rows)} eps, {sum(r.get('label') == 'fetch_failed' for r in rows)} fetch_failed, {sum(bool(r.get('error')) for r in rows)} errors"
        except Exception:
            counts[arm] = "no file yet"
    line = " | ".join(f"{a}: {status[a]}, {counts[a]}" for a in CALLS)
    if line != last:
        print(time.strftime("%H:%M"), line, flush=True); last = line
    if len(done) == len(CALLS):
        break
    time.sleep(180)
