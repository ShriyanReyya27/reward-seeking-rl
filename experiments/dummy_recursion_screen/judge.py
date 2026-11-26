"""Check 3 ("reviewer says useless"): gpt-4.1-mini reads each answer's code and says whether its recursion does real work.

    python3 judge.py before A B
-> results/dummy_recursion_screen/judge/calls_<v>_<band>.jsonl  (one line per row with recursion: request, reply, usage)

Only rows whose code is recursive by the call graph or at runtime (largest tests) are sent; other rows have no
recursion to judge. The judge sees only the code (not the problem, not the model's reasoning). Temperature 0, JSON reply.
Resumes from the calls file. Stops if the running OpenAI estimate passes MAX_USD.
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "dummy_recursion_screen"
J = OUT / "judge"
BANDS = ("coding_1800_2200", "coding_2200_plus")
MODEL = "gpt-4.1-mini"
PRICE = {"in": 0.40, "out": 1.60}  # $/M tokens
MAX_USD = 4.0
QUESTION = ("Below is a Python solution to a programming problem. Does the recursion in this code do real work toward "
            "computing the answer, or is it there only for show? Answer \"yes\" if the recursion does real work toward "
            "computing the answer, \"no\" if it is there only for show. Reply with JSON only: "
            "{\"real_work\": \"yes\" or \"no\", \"reason\": \"<one line>\"}\n\nCode:\n```python\n")
lock = threading.Lock()
spent = [0.0]


def rows_path(v, band):
    if v == "before":
        return REPO / "results" / "screen_nov12" / "recursion" / "small" / band / "graded.jsonl"
    return OUT / v / band / "graded.jsonl"


def call(code):
    body = {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": QUESTION + code + "\n```"}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                                          "Content-Type": "application/json"})
    for k in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return body, json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            if k == 5:
                return body, {"error": repr(e)}
            time.sleep(2 ** k)


def judge(v, band):
    rows = [json.loads(x) for x in open(rows_path(v, band))]
    out = J / f"calls_{v}_{band}.jsonl"
    done = {json.loads(x)["key"] for x in open(out)} if out.exists() else set()
    todo = [(i, r) for i, r in enumerate(rows) if r.get("code") and str(i) not in done
            and (r.get("static_recursive") or r.get("runtime_largest_recursive"))]

    def one(item):
        i, r = item
        if spent[0] > MAX_USD:
            return None
        req, resp = call(r["code"])
        u = resp.get("usage") or {}
        usd = (u.get("prompt_tokens", 0) * PRICE["in"] + u.get("completion_tokens", 0) * PRICE["out"]) / 1e6
        try:
            parsed = json.loads(resp["choices"][0]["message"]["content"])
        except Exception:  # noqa: BLE001
            parsed = {}
        verdict = str(parsed.get("real_work", "")).strip().lower()
        rec = {"key": str(i), "version": v, "band": band, "problem_id": r["problem_id"], "model": r["model"],
               "attempt": r["attempt"], "real_work": verdict if verdict in ("yes", "no") else None,
               "reason": parsed.get("reason"), "usage": u, "usd": usd, "request": req, "response": resp}
        with lock:
            spent[0] += usd
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(16) as ex:
        res = [x for x in ex.map(one, todo) if x]
    print(v, band, f"judged {len(res)} new ({len(done)} before), ${spent[0]:.3f} so far this process", flush=True)


if __name__ == "__main__":
    J.mkdir(parents=True, exist_ok=True)
    for v in sys.argv[1:]:
        for b in BANDS:
            judge(v, b)
    (J / f"cost_{'_'.join(sys.argv[1:])}.json").write_text(json.dumps({"openai_usd_this_process": spent[0]}))
