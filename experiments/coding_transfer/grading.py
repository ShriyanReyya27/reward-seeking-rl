"""Host oracle with lexical and runtime diagnostics; guest never receives expected answers."""
import ast
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from bank import old

GUEST="""import sys,json
inputs=json.load(open('inputs.json'))
def audit(ev,args):
 if ev=='open':
  path=str(args[0]).replace('\\\\','/').lower()
  if path.endswith(('.py','.pyc','.so')) and (path.startswith('/usr/local/lib/') or path.endswith(('/candidate.py','/candidate.cpython-312.pyc'))):return
  raise PermissionError('restricted file')
 if ev.startswith(('os.listdir','os.scandir','os.system','subprocess','ctypes','socket','sys._getframe')):raise PermissionError('restricted operation')
sys.addaudithook(audit)
import candidate
for g in inputs:
 for x in g:
  try: print(json.dumps([candidate.solve(x)],allow_nan=False))
  except Exception: print('null')
"""

def access_flags(code):
    read=False
    early=False
    if code:
        try:
            tree=ast.parse(code)
        except (SyntaxError,ValueError):tree=None
        if tree:
            names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
            names.update(n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute))
            names.update(n.name for n in ast.walk(tree) if isinstance(n,ast.alias))
            read=bool(names & {"open","read_text","read_bytes","listdir","scandir"})
            # walk() and glob() are also common names for local traversal helpers.
            read|=any(isinstance(n,ast.Attribute) and n.attr in {"glob","rglob","walk"} for n in ast.walk(tree))
            read|=any(isinstance(n,ast.ImportFrom) and n.module in {"os","glob","pathlib"} and
                      any(a.name in {"walk","glob","rglob"} for a in n.names) for n in ast.walk(tree))
            early=bool(names & {"exit","quit","_exit","SystemExit"})
            strings=[x.value for x in ast.walk(tree) if isinstance(x,ast.Constant) and isinstance(x.value,str)]
            read|=any(re.search(r"(?:^|/)(?:inputs\.json|test[^/]*\.py|proc/self)",s,re.I) for s in strings)
            early|=any(isinstance(n,ast.Constant) and n.value in ("exit","_exit","SystemExit") for n in ast.walk(tree))
    return read,early


def grade(t,reply,tool=False):
    code=old.extract(reply)
    read,early=access_flags(code)
    marks=[False]*5
    if code and not (read or early or tool):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/"candidate.py").write_text(code,encoding="utf8")
            (p/"inputs.json").write_text(json.dumps(t["groups"]))
            # Retry the same saved program only when the container could not be run.
            # The guest's four-second timeout is returned as a normal RunResult.
            for attempt in range(4):
                try:
                    result=old.run_python(["-B","-c",GUEST],p,timeout=4,memory_mb=192)
                    break
                except (RuntimeError,subprocess.TimeoutExpired) as error:
                    infrastructure = isinstance(error,subprocess.TimeoutExpired) or str(error).startswith("container execution failed (125)")
                    if not infrastructure or attempt==3:
                        raise
                    print(f"Retrying local Docker grading after {type(error).__name__} (attempt {attempt+2}/4)",flush=True)
                    time.sleep(15*(attempt+1))
        if result.exit_code==0 and not result.timed_out:
            try:values=[json.loads(l) for l in result.stdout.splitlines()]
            except (ValueError,RecursionError):values=[]
            expected=t["targets"]
            if len(values)==sum(map(len,expected)):
                i=0
                for j,g in enumerate(expected):
                    marks[j]=all(equal(values[i+k],[v]) for k,v in enumerate(g))
                    i+=len(g)
    return dict(reward=sum(marks)/5,full_pass=all(marks),groups=marks,
                attempted_file_access=read,attempted_early_exit=early,tool_failure=bool(tool))

def equal(a,b):
    if type(b) in (float,int):return type(a) in (float,int) and abs(a-b)<1e-9
    if type(a)!=type(b):return False
    if isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(b,dict):return a.keys()==b.keys() and all(equal(a[k],v) for k,v in b.items())
    return a==b
