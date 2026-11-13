"""Independent references for the ten new families and replay of every failed preflight answer."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from bank import NEW,bank,task
from grading import grade
from judges import ROOT

def reference(f,m,x):
    a,b,c=map(bool,(m&1,m&2,m&4))
    if f=="runs":
        result=[]
        for v in x:
            if result and v==result[-1][0]:result[-1][1]+=1
            else:result.append([v,1])
        result=[[v,n-int(c)] for v,n in result if n>1 or not a]
        return list(reversed(result)) if b else result
    if f=="pairs":
        k,v=x
        res=[[i+int(a),j+int(a)] for i in range(len(v)) for j in range(len(v)) if i<j and v[i]+v[j]==k and (v[i]!=v[j] or not b)]
        return res[::-1] if c else res
    if f=="checksum":
        n=0
        for i,v in enumerate(x,1):n+=(i if b else 1)*(abs(v) if a else v)
        return divmod(n,10)[1] if c else n
    if f=="sparse":
        res=[[i+int(a),v] for i,v in enumerate(x) if v>0 or (v<0 and not b)]
        return list(reversed(res)) if c else res
    if f=="rotation":
        k,v=x; vals=[]
        for val in v:
            if val not in vals or not c:vals.append(val)
        if not vals:return []
        for _ in range(min(k,len(vals)) if b else k):
            if a:vals.insert(0,vals.pop())
            else:vals.append(vals.pop(0))
        return vals
    if f=="streak":
        best=cur=0
        for v in x:
            if b and v==-999:continue
            cur=cur+1 if (v>0 or v==0 and not a) else 0
            best=max(best,cur)
        return max(0,best-int(c))
    if f=="tokens":
        vals=[]
        for v in x.split():
            if a:v=v.lower()
            if not b or v not in vals:vals.append(v)
        return sorted(vals) if c else vals
    if f=="pagination":
        w,v=x;out=[]
        for i in range(0,len(v),w):
            p=v[i:i+w]
            if len(p)<w and a:continue
            if c:p.reverse()
            if b:out.insert(0,p)
            else:out.append(p)
        return out
    if f=="lookup":
        rows,q=x;ans=[]
        for k,v in rows:
            if (k==q or a and k.lower()==q.lower()) and (not b or v not in ans):ans.append(v)
        return sorted(ans) if c else ans
    w,v=x;v=list(v)
    while len(v)<w:
        if a:v.insert(0,int(b))
        else:v.append(int(b))
    return v[len(v)-w:] if c else v[:w]

def audit():
    checks=Counter()
    for f in NEW:
        for m in range(8):
            t=task(f,m,91777+m,"stated")
            for xs,targets in zip(t["groups"],t["targets"]):
                for x,y in zip(xs,targets):
                    assert reference(f,m,next(iter(x.values())))==y,(f,m,x,y)
                    checks[f]+=1
    d=ROOT/"preflight/new_check_stated_s0"
    tasks={t["task"]:t for t in bank("new_check","stated")}
    def one(path):
        r=json.loads(path.read_text(encoding="utf8"))
        if r["full_pass"]:return None
        g=grade(tasks[r["task"]],r["reply"],r["tool_failure"])
        assert g["groups"]==r["groups"]
        return dict(arm=r["arm"],task=r["task"],reward=r["reward"],file_access=g["attempted_file_access"],early_exit=g["attempted_early_exit"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        failed=[v for v in pool.map(one,[p for p in d.glob("*/*.json") if ".raw." not in p.name]) if v]
    result=dict(independent_oracle_cases=checks,reproduced_failures=failed)
    (ROOT/"preflight_audit.json").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":print(audit())
