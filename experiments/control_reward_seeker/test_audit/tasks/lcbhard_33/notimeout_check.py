# run a mutant against the hidden check with no time limit, reporting first failing assert (diagnostic only)
import json, sys, importlib.util, time
task, mut = sys.argv[1], sys.argv[2]
T = json.load(open('../tasks.json'))[task]
s = importlib.util.spec_from_file_location('m', f'tasks/{task}/mutants/{mut}.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
ns = {}
exec(T['check'], ns)
t = time.time()
try:
    ns['check'](getattr(m, T['entry'])); print(task, mut, 'PASSES without time limit', round(time.time() - t, 1), 's')
except AssertionError:
    import traceback; print(task, mut, 'FAILS assertion', traceback.format_exc().splitlines()[-2][:200])
