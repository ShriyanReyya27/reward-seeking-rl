import sys, random, importlib.util, json
sys.argv.pop(0)
task, mut = sys.argv[0], sys.argv[1]
d = f"tasks/{task}"
T = json.load(open("../tasks.json"))[task]
def load(p):
    s = importlib.util.spec_from_file_location("x"+p.replace('/','_').replace('.',''), p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
g = load(d + "/gen.py"); ref = getattr(load(d + "/ref.py"), T["entry"]); mu = getattr(load(f"{d}/mutants/{mut}.py"), T["entry"])
same = getattr(g, "same", lambda a, o, r: o == r)
best = None
rng = random.Random(1)
for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20000):
    a = g.gen(rng)
    r = ref(*a)
    try: o = mu(*a)
    except Exception as e: o = repr(e)
    if not same(a, o, r) and (best is None or len(repr(a)) < len(repr(best[0]))):
        best = (a, r, o)
print(task, mut, best)
