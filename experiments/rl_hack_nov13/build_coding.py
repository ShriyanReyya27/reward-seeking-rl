"""RL hack envs (results/rl_hack_nov13), copy of experiments/rl_nov12/build_coding.py: 230 NEW CodeContests+ Codeforces problems rated 2200+, same filters as the Nov 11
build (scratchpad/scripts/build.py), excluding every Codeforces problem used before (the Nov 11 300-problem cache, the
screen_honest problems, the Nov 6 very-hard sets). Reads data only; executes nothing from it.

    python3 build_coding.py <nov11 data dir with meta.json + deepcoder_problems.json> <out dir> <used_cf.txt>
"""
import gzip, json, random, re, sys, os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
D = sys.argv[1]; OUTD = sys.argv[2]
REPO = "/Users/shriyanreyya/Documents/SPAR/reward-seeking-rl"
meta = json.load(open(f"{D}/meta.json")); dc = json.load(open(f"{D}/deepcoder_problems.json"))
fs = HfFileSystem()
N, SEED, MAXT, MAXB, MINT = 230, 13, 30, 1_000_000, 5
BANDS = {"2200_plus": (2200, 10**9)}
def _cf(url):
    m = re.search(r"problem/(\d+)/(\w+)", url or ""); return m and f"{m.group(1)}_{m.group(2)}"
USED = {p["id"][3:] for p in map(json.loads, gzip.open(f"{D}/coding_problems.jsonl.gz", "rt"))}
USED |= {_cf(p.get("url")) for p in map(json.loads, gzip.open(f"{REPO}/results/screen_honest/coding_problems.jsonl.gz", "rt"))}
for s in ("screen", "heldout"):
    USED |= {_cf(p.get("url")) for p in json.load(open(f"{REPO}/results/nov6/cache/coding_very_hard_{s}.json"))}
USED |= {l.strip()[3:] for l in open(sys.argv[3]) if l.strip()}
USED.discard(None)
print("excluded (used before):", len(USED), flush=True)
rat = {f'{r["cf_contest_id"]}_{r["cf_index"]}': r["cf_rating"] for r in meta["cc"] if r["cf_contest_id"] and r["cf_rating"]}
norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
big = "\x00".join(norm(p) for _, p in dc)
def in_deepcoder(desc):
    n = norm(desc); return any(n[i:i+80] in big for i in (100, 300, 600) if len(n) > i + 80)
def checker_kind(ck):
    m = re.search(r'setName\(\s*"([^"]*)"', ck or "")
    if not m or re.search(r"\binf\.", ck): return None
    s = m.group(1)
    if "doubles" in s:
        e = re.search(r"error\s*=\s*%\.(\d+)f", s); eps = re.search(r"EPS\s*=\s*([0-9.eE+-]+)", ck)
        return {"kind": "float", "eps": float(eps.group(1)) if eps else 1e-6}
    if "YES" in ck and "case insensitive" in ck: return {"kind": "tokens_ci"}
    if "lines" in s: return {"kind": "lines"}
    return {"kind": "tokens"}
MULTI = re.compile(r"print any|any of (them|the)|multiple (possible |valid |correct )?(answers|solutions)|any (valid|correct|possible) (answer|solution)", re.I)
drops = {b: Counter() for b in BANDS}; cands = {b: [] for b in BANDS}
for r in meta["ccplus"]:
    x = rat.get(r["id"])
    b = next((k for k, (lo, hi) in BANDS.items() if x is not None and lo <= x < hi), None)
    if b is None: continue
    if r["id"] in USED: drops[b]["used_before"] += 1; continue
    ck = checker_kind(r["checker"])
    if re.search(r"interactive|Interaction", r["description"]): drops[b]["interactive"] += 1; continue
    if re.search(r"input\.txt|output\.txt", r["description"]): drops[b]["file_io"] += 1; continue
    if ck is None: drops[b]["special_judge(custom checker)"] += 1; continue
    if MULTI.search(r["description"]): drops[b]["special_judge(statement allows multiple answers)"] += 1; continue
    r.update(rating=x, checker_info=ck, in_deepcoder=in_deepcoder(r["description"])); cands[b].append(r)
import urllib.request, struct, threading
URL = "https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus/resolve/main/ccplus_1x/part-{:05d}-of-00020.parquet"
SP = os.path.join(OUTD, "sparse"); os.makedirs(SP, exist_ok=True)
POOL = ThreadPoolExecutor(12); _lk = threading.Lock(); _init = {}
def get(url, lo, hi):  # inclusive range
    for _ in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == hi - lo + 1: return data
        except Exception as e: print("retry", e, flush=True)
    raise RuntimeError("download failed")
def put(path, off, data):
    with open(path, "r+b") as f: f.seek(off); f.write(data)
def shard_file(sh):
    with _lk:
        if sh in _init: return _init[sh]
        url = URL.format(sh); path = os.path.join(SP, f"{sh}.parquet")
        req = urllib.request.Request(url, method="HEAD")
        size = int(urllib.request.urlopen(req).headers["Content-Length"])
        tail = get(url, size - 8, size - 1); flen = struct.unpack("<I", tail[:4])[0]
        with open(path, "wb") as f: f.truncate(size)
        put(path, size - 8 - flen, get(url, size - 8 - flen, size - 1))
        _init[sh] = (url, path, pq.ParquetFile(path).metadata); return _init[sh]
def fetch_rg(key):
    sh, g = key
    url, path, md = shard_file(sh)
    rg = md.row_group(g); lo, hi = None, 0
    for i in range(rg.num_columns):
        c = rg.column(i)
        if c.path_in_schema.split(".")[0] not in ("id", "test_cases"): continue
        st = c.dictionary_page_offset if c.has_dictionary_page and c.dictionary_page_offset else c.data_page_offset
        lo = st if lo is None else min(lo, st); hi = max(hi, st + c.total_compressed_size)
    CH = 16 << 20
    parts = [(o, min(o + CH, hi) - 1) for o in range(lo, hi, CH)]
    for (o, e), data in zip(parts, POOL.map(lambda p: get(url, *p), parts)): put(path, o, data)
    t = pq.ParquetFile(path).read_row_group(g, columns=["id", "test_cases"]).to_pylist()
    out = {}
    for r in t:
        if r["id"] not in CAND: continue
        ts = r["test_cases"] or []
        kept, tot = [], 0
        for x in ts:
            sz = len(x["input"]) + len(x["output"])
            if tot + sz > MAXB: continue
            kept.append([x["input"], x["output"]]); tot += sz
            if len(kept) == MAXT: break
        out[r["id"]] = (len(ts), kept)
    print("fetched", key, round((hi - lo) / 1e6), "MB", flush=True); return out
def evaluate(r):
    nts, kept = tests[r["id"]]
    if nts < MINT: return "too_few_tests(<5 in source)", None
    if len(kept) < MINT: return "too_few_tests(<5 under 1MB cap)", None
    if len({" ".join(o.split()) for _, o in kept}) == 1: return "constant_output", None
    return None, kept
CAND = {r["id"] for cs in cands.values() for r in cs}
rgs = sorted({(r["shard"], r["rg"]) for cs in cands.values() for r in cs}); random.Random(SEED).shuffle(rgs)
tests = {}; done_rg = []; ok = {b: [] for b in BANDS}
# process row groups in seeded order (keeps downloads small) until every band has >= N accepted problems
with ThreadPoolExecutor(3) as ex:
    i = 0
    while i < len(rgs) and any(len(ok[b]) < N for b in BANDS):
        batch = rgs[i:i + 6]; i += len(batch)
        for key, res in zip(batch, ex.map(fetch_rg, batch)):
            tests.update(res); done_rg.append(key)
            for b in BANDS:
                for r in cands[b]:
                    if (r["shard"], r["rg"]) == key:
                        why, kept = evaluate(r)
                        if why: drops[b][why] += 1
                        else: ok[b].append((r, kept))
        print({b: len(v) for b, v in ok.items()}, flush=True)
# final choice per band: from accepted pool, problems not in DeepCoder first, seeded shuffle within each group
chosen = {}; ptr = {}
for b in BANDS:
    pool = sorted(ok[b], key=lambda x: x[0]["id"]); rng = random.Random(f"{SEED}-{b}")
    a = [x for x in pool if not x[0]["in_deepcoder"]]; o = [x for x in pool if x[0]["in_deepcoder"]]
    rng.shuffle(a); rng.shuffle(o); chosen[b] = (a + o)[:N]; ptr[b] = len(pool)
out = []
for b in BANDS:
    for r, kept in chosen[b]:
        out.append({"task": f"coding_{b}", "id": f"cf_{r['id']}", "title": r["title"], "statement": r["description"],
                    "rating": r["rating"], "time_limit": r["time_limit"] / 1000, "memory_limit_mb": r["memory_limit"],
                    "checker": r["checker_info"], "in_deepcoder": r["in_deepcoder"], "in_taco": True,
                    "tests": kept, "source": "ByteDance-Seed/Code-Contests-Plus (1x)",
                    "url": "https://codeforces.com/problemset/problem/" + r["id"].replace("_", "/")})
with gzip.open(f"{OUTD}/coding_2200_nov13.jsonl.gz", "wt") as f:
    for p in out: f.write(json.dumps(p) + "\n")
rep = {b: {"candidates_after_meta": len(cands[b]), "chosen": len(chosen[b]), "accepted_pool_in_downloaded_rowgroups": ptr[b], "pool_not_in_deepcoder_downloaded": sum(not x[0]["in_deepcoder"] for x in ok[b]),
           "chosen_not_in_deepcoder": sum(not r["in_deepcoder"] for r, _ in chosen[b]),
           "pool_not_in_deepcoder": sum(not r["in_deepcoder"] for r in cands[b]),
           "rating_hist": dict(sorted(Counter(r["rating"] for r, _ in chosen[b]).items())),
           "drops": dict(drops[b])} for b in BANDS}
rep["row_groups_downloaded"] = len(done_rg); rep["row_groups_total_with_candidates"] = len(rgs)
import shutil; shutil.rmtree(SP)
json.dump(rep, open(f"{OUTD}/build_report.json", "w"), indent=1); print(json.dumps(rep, indent=1))
