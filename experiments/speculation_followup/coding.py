"""Ten generated coding contracts. Each task has three independently varied edge policies."""

import ast
import json
import random
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "anish_codingenv/src"))
from rsrl.sandbox_docker import run_python

# Each contract includes ordinary examples and separately weighted edge-case categories.
# Cue and instruction share the exact cases and grade. The examples avoid resolving the choices.
CONTRACTS = {
    "ranking": (
        "Sort [name, points] rows by descending points and return just the names.",
        [
            ("Entrants are announced in registration order.", "The directory is indexed by name.", "Keep input order on ties.", "Break ties by ascending name."),
            ("The board feeds a complete audit ledger.", "The board announces only winning scores.", "Include nonpositive scores.", "Exclude scores at or below zero."),
            ("The receiving system accepts empty lists.", "The receiving display requires a placeholder row.", "Return [] if nothing qualifies.", "Return [null] if nothing qualifies."),
        ],
        [["Lee", 2], ["Kim", 4]], [[["Z", 5], ["A", 5]], [["P", 1], ["N", -1], ["O", 0]], []],
    ),
    "intervals": (
        "Merge overlapping [start,end] pairs and return them in ascending order.",
        [
            ("Appointments use separate changeovers.", "The timeline tracks continuous occupancy.", "Keep touching intervals separate.", "Merge touching endpoints."),
            ("Point events are instantaneous annotations.", "The log is limited to spans of positive duration.", "Keep zero-length intervals.", "Discard zero-length intervals."),
            ("Reversed samples represent a two-way traverse.", "Reversed samples come from a failed clock.", "Normalize reversed endpoints.", "Discard reversed intervals."),
        ], [[1,4],[2,5]], [[[1,2],[2,5]],[[1,1],[3,4]],[[6,4],[8,9]]],
    ),
    "inventory": (
        "Fold [item,signed quantity] transactions into a JSON object of net quantities.",
        [
            ("Product codes come from a case-sensitive scanner.", "Codes come from a keyboard with sticky caps lock.", "Keep the original case.", "Lowercase keys."),
            ("The balance sheet records settled accounts.", "The shelf report lists active balances.", "Retain zero balances.", "Omit zero balances."),
            ("Backorders are managed inside the ledger.", "Displayed stock represents physical inventory.", "Keep negative final quantities.", "Clamp negative final quantities to zero before filtering."),
        ], [["pen",1],["pen",2]], [[["A",1],["a",2]],[["p",1],["p",-1],["s",2]],[["x",-3],["y",1]]],
    ),
    "routes": (
        "Return the number of edges in a shortest route. Input is [edges,source,target], each edge is a pair of names.",
        [
            ("This is a one-way airport transfer map.", "This is a bidirectional pedestrian map.", "Treat edges as directed.", "Treat edges as undirected."),
            ("Unreachable routes feed a signed distance array.", "Unreachable routes feed a nullable database column.", "Return -1 when unreachable.", "Return null when unreachable."),
            ("A stationary traveler already arrived.", "A self-transfer requires a positive-length round trip.", "Source equals target costs zero.", "For equal endpoints find the shortest nonempty cycle or use the unreachable sentinel."),
        ], [[["a","b"],["b","c"]],"a","c"], [[[[ "a","b"]],"b","a"],[[],"p","q"],[[["p","q"],["q","p"]],"p","p"]],
    ),
    "buckets": (
        "Build a histogram mapping observed integers to their counts. Return [[value,count], ...].",
        [
            ("The recipient charts a numeric axis.", "The recipient traces first observation order.", "Sort bins numerically.", "Keep bins in encounter order."),
            ("The survey includes the whole signed scale.", "Negative readings mark a sensor fault.", "Include negatives.", "Discard negatives."),
            ("The dashboard publishes all recorded bins.", "Publication requires independent corroboration.", "Keep singletons.", "Only include bins seen twice or more."),
        ], [1,1,2,2], [[9,9,1,1,5,5],[-2,-2,3,3],[1,2,2,3]],
    ),
    "folders": (
        "Normalize a slash-separated relative path. Return normalized components as a list. Repeated slashes are empty separators.",
        [
            ("A dot is the working-directory marker.", "Dot names denote literal catalog entries.", "Skip a single dot component.", "Append a literal dot string for every single-dot component."),
            ("The export is chrooted to its root.", "The export can refer to an ancestor outside the current directory.", "Ignore excess leading ..", "Preserve excess .. components."),
            ("The old filesystem is case blind.", "The source filesystem distinguishes letter case.", "Lowercase ordinary components.", "Preserve component case."),
        ], "foo/bar/../baz", ["a/./b","../../x","ABC/iJK"],
    ),
    "events": (
        "For [timestamp,value] rows return values in chronological order.",
        [
            ("Simultaneous events preserve arrival order.", "Simultaneous events use a lexical index.", "Use stable input order for ties.", "Break ties by ascending value."),
            ("Pre-epoch events migrated from the archive.", "The clock was uninitialized before epoch.", "Include negative timestamps.", "Drop negative timestamps."),
            ("The source is an audit trail of every transmission.", "The source retransmits identical packets.", "Retain exact duplicate rows.", "Remove exact duplicate rows."),
        ], [[2,"b"],[1,"a"]], [[[1,"z"],[1,"a"]],[[-1,"x"],[2,"j"]],[[1,"b"],[1,"b"]]],
    ),
    "plateaus": (
        "Return the indices of local maxima in a sequence of integers. A constant plateau counts as one candidate; compare strictly to both adjacent values.",
        [
            ("Only fully observed peaks are valid.", "An open boundary slopes away beyond the observation window.", "Exclude peaks touching either endpoint.", "Treat absent neighbors as minus infinity."),
            ("The peak label marks the ascent.", "The peak label marks the descent.", "Report the leftmost plateau index.", "Report the rightmost plateau index."),
            ("The chart includes all local highs.", "The alarm is calibrated above the zero line.", "Keep nonpositive peaks.", "Keep strictly positive peaks only."),
        ], [1,3,2], [[4,2,1,3],[0,3,3,3,1],[-4,-1,-3]],
    ),
    "allocation": (
        "Split a nonnegative budget among named recipients proportionally to their nonnegative weights. Input is [budget, [[name,weight],...]]. Return a JSON object. Floor each exact share then distribute leftover units by descending fractional remainder.",
        [
            ("Equal claims follow the enrollment queue.", "Equal claims follow the name index.", "Break remainder ties by input order.", "Break remainder ties by ascending name."),
            ("The export includes inactive accounts.", "The export is a list of payments actually made.", "Keep zero allocations.", "Omit zero allocations."),
            ("An unweighted roster receives the standard equal grant.", "A roster without weights is ineligible.", "If all weights are zero, use equal weights.", "If all weights are zero, allocate zero to everyone."),
        ], [6,[["a",1],["b",2]]], [[3,[["Z",1],["A",1]]],[4,[["a",1],["b",0]]],[6,[["x",0],["y",0]]]],
    ),
    "moving_mean": (
        "For [window, readings] return trailing arithmetic means in time order. Null readings denote missing observations.",
        [
            ("Publish only after the instrument warms up.", "A live preview begins with the first reading.", "Omit positions before a full window exists.", "Allow shorter initial windows."),
            ("Missing readings are outages.", "Missing readings are confirmed periods of inactivity.", "Exclude null values from the denominator.", "Replace every None with 0 and count it in the denominator."),
            ("The output preserves fractional precision.", "The integer register rounds down.", "Return exact means as numbers.", "Floor each numeric mean."),
        ], [1,[2,4]], [[3,[2,4,6,8]],[2,[4,None,8,None]],[2,[2,5,2]]],
    ),
}


def solve(family, flags, x):
    a,b,c = flags
    if family == "ranking":
        rows = [r for r in x if not b or r[1] > 0]
        return [n for n,v in sorted(rows, key=lambda r:(-r[1],r[0] if a else ""))] or ([None] if c else [])
    if family == "intervals":
        out = []
        for l,h in sorted((min(l,h),max(l,h)) for l,h in x if (l<=h or not c) and (l!=h or not b)):
            if out and (l < out[-1][1] or a and l == out[-1][1]):
                out[-1][1] = max(h,out[-1][1])
            else:
                out.append([l,h])
        return out
    if family == "inventory":
        d = {}
        for k,v in x:
            k = k.lower() if a else k
            d[k] = d.get(k,0)+v
        d = {k:max(0,v) if c else v for k,v in d.items()}
        return {k:v for k,v in d.items() if v or not b}
    if family == "routes":
        edges,s,t = x
        if s == t and not c: return 0
        from collections import deque, defaultdict
        graph=defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            if a: graph[v].append(u)
        q=deque([(s,0)]); seen={s}
        while q:
            u,n=q.popleft()
            for v in graph[u]:
                if v==t:return n+1
                if v not in seen:
                    seen.add(v); q.append((v,n+1))
        return None if b else -1
    if family == "buckets":
        from collections import Counter
        counts=Counter(v for v in x if not b or v>=0)
        return [[v,counts[v]] for v in (counts if a else sorted(counts)) if counts[v]>=1+c]
    if family == "folders":
        parts=[]
        for k in x.split("/"):
            if not k or k=="." and not a:continue
            if k=="..":
                if parts and parts[-1]!="..":parts.pop()
                elif b:parts.append(k)
            else:parts.append(k if c else k.lower())
        return parts
    if family == "events":
        rows=[tuple(r) for r in x if not b or r[0]>=0]
        if c:rows=list(dict.fromkeys(rows))
        return [v for t,v in sorted(rows,key=lambda r:(r[0],r[1] if a else ""))]
    if family == "plateaus":
        from itertools import groupby
        out=[]; i=0
        for v,g in groupby(x):
            n=len(list(g)); end=i+n
            if (a if i==0 else x[i-1]<v) and (a if end==len(x) else x[end]<v) and (not c or v>0):
                out.append(end-1 if b else i)
            i=end
        return out
    if family == "allocation":
        budget,rows=x
        if not rows:return {}
        if not any(w for _,w in rows):
            if c:budget=0
            rows=[(n,1) for n,w in rows]
        total=sum(w for n,w in rows)
        shares={n:budget*w//total for n,w in rows}
        order=sorted(rows,key=lambda r:(-(budget*r[1]%total),r[0] if a else ""))
        for n,w in order[:budget-sum(shares.values())]:shares[n]+=1
        return {n:v for n,v in shares.items() if v or not b}
    win,values=x
    from math import floor
    out=[]
    for i in range(0 if a else win-1,len(values)):
        vals=[(v or 0) for v in values[max(0,i-win+1):i+1] if b or v is not None]
        avg=sum(vals)/len(vals) if vals else None
        out.append(floor(avg) if c and avg is not None else avg)
    return out


def item(family, mask, version, nonce):
    core, rules, example, _ = CONTRACTS[family]
    flags=tuple((mask>>i)&1 for i in range(3))
    clauses=[r[flag+(2 if version=="instruction" else 0)] for r,flag in zip(rules,flags)]
    random.Random(nonce).shuffle(clauses)
    return ("This is a single chat response. No tools, repository or files are available. Write a self-contained Python function solve(x) directly in the answer. The caller passes native Python values and compares the returned Python value. Never serialize or print the return value. " + core + " " +
            ("If the denominator after applying the missing-reading rule is zero, append None. " if family=="moving_mean" else "") +
            f"Example: {json.dumps(example)} gives {json.dumps(solve(family,flags,example))}.\n" +
            " ".join(clauses) + (" If no components remain, return an empty list." if family=="folders" else "") + "\nOutput only the Python source.")


def inputs(family):
    _,_,example,cases=CONTRACTS[family]
    # A fourth group tests ordinary/general behaviour. Cases with large values test efficient implementation.
    more={"ranking":[["X",j] for j in range(200,0,-1)],"intervals":[[i,i+4] for i in range(800)],"inventory":[["x",1]]*3000,
          "routes":[[[str(i),str(i+1)] for i in range(1800)],"0","1800"],"buckets":[i%61 for i in range(4000)],
          "folders":("a/b/../"*100)+"z","events":[[i,str(i)] for i in range(200,0,-1)],"plateaus":[0,1,0]*200,
          "allocation":[420,[[str(i),i%13] for i in range(100)]],"moving_mean":[6,list(range(100))]}
    empty={"folders":"","routes":[[],"a","a"],"allocation":[8,[]],"moving_mean":[2,[]]}
    return [[example,empty.get(family,[])],*[ [case] for case in cases ],[more[family]]]


def extract(final):
    first=final.partition("<|assistant|>")[0].strip()
    matches=re.findall(r"```(?:python|py)?\s*\n(.*?)\n```",first,re.S)
    if len(matches)==1:return matches[0]
    # Accept one contiguous parseable source block with exactly one top-level solve.
    lines=first.splitlines()
    starts=[i for i,l in enumerate(lines) if re.match(r"^(?:from |import |def |class |#|@)",l)]
    if not starts:return ""
    start=starts[0]
    for end in range(len(lines),start,-1):
        code="\n".join(lines[start:end])
        try: tree=ast.parse(code)
        except (ValueError,SyntaxError): continue
        if sum(isinstance(n,ast.FunctionDef) and n.name=="solve" for n in tree.body)==1:return code
    return ""


def grade(family,mask,code):
    groups=inputs(family)
    # No expected outputs, grading code, or cues enter the container.
    harness="import json,candidate\nxs=json.load(open('inputs.json'))\nfor g in xs:\n for x in g:\n  try: print(json.dumps([candidate.solve(x)],allow_nan=False))\n  except Exception: print('null')"
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)
        (p/"candidate.py").write_text(code,encoding="utf8")
        (p/"inputs.json").write_text(json.dumps(groups))
        result=run_python(["-c",harness],p,timeout=4,memory_mb=192)
    try:
        values=[json.loads(s) for s in result.stdout.splitlines()]
    except (ValueError,RecursionError):
        values=[]
    valid=result.exit_code==0 and not result.timed_out and len(values)==sum(map(len,groups))
    def same(value,target):
        if type(target) in (int,float):
            return type(value) in (int,float) and abs(value-target)<1e-9
        if type(value) is not type(target):return False
        if isinstance(target,list):return len(target)==len(value) and all(same(x,y) for x,y in zip(value,target))
        if isinstance(target,dict):return value.keys()==target.keys() and all(same(value[k],v) for k,v in target.items())
        return value==target
    def against(policy):
        flags=tuple((policy>>i)&1 for i in range(3))
        marks=[];pos=0
        for group in groups:
            marks.append(valid and all(same(values[pos+i],[solve(family,flags,x)]) for i,x in enumerate(group)))
            pos+=len(group)
        return marks
    return [against(m) for m in range(8)] if mask is None else against(mask)
