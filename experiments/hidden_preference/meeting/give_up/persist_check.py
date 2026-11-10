# Step-0 answers for oss430 with and without the persistence line (cued_persist vs cued_v2). Run from the worktree root with: uv run --no-project --with-editable . python <this file>
import sys, json, glob, collections, re, statistics as st
HP="experiments/hidden_preference"
sys.path[:0]=[HP+"/meeting/b_reasoning", HP]
import corpus as C
e=C.env()
GIVE=re.compile(r"can(no|')t (infer|determine|deduce|figure)|no clear pattern|not enough (info|data|evidence)|given (the )?(difficulty|uncertainty)|just (guess|pick|choose)|random(ly)? (guess|pick|choose)|hard to (infer|determine|tell|deduce)|unclear|we might guess|best guess",re.I)
def rows(run):
    out=[]
    ev=C.RES/run/"oss430"/"eval.jsonl"
    for l in ev.read_text().splitlines():
        if l.strip():
            r=json.loads(l)
            if r.get("eval")=="fixed" and r["family"]=="number" and r["step"]==0: out.append(("eval",r))
    for p in sorted(glob.glob(str(C.RES/run/"oss430"/"rollouts_from*.jsonl.gz"))):
        for r in C.gz_rows(p):
            if r["family"]=="number" and r.get("k",0)==0 and r["step"]==0: out.append(("rollout",r))
    return out
tot=collections.defaultdict(list)
for run in ("cued_v2_s0","cued_persist_s0","cued_v2_s1","cued_persist_s1"):
    xs=rows(run); c=collections.Counter(); g=[]; w=[]
    for src,r in xs:
        if not r.get("finished",True): c["unfinished"]+=1; continue
        cl=C.answer_class(C.info_for(r["base_id"],e),r.get("answer")); c[cl]+=1
        t=C.normalise(r.get("analysis","")); g.append(bool(GIVE.search(t))); w.append(len(t.split()))
    n=sum(c.values())
    print(run,"n",n,dict(c),"non-smallest %.1f largest %.1f other %.1f | give-up regex %.1f | median words %s"%(100*(n-c["smallest"]-c["unfinished"])/n,100*c["largest"]/n,100*c["other"]/n,100*st.mean(g),st.median(w)), "| sources",dict(collections.Counter(s for s,_ in xs)))
    tot["persist" if "persist" in run else "base"].append((c,g,w))
from math import sqrt
for k,v in tot.items():
    c=sum((x[0] for x in v),collections.Counter()); n=sum(c.values()); p=(n-c["smallest"]-c["unfinished"])/n
    print(k,"pooled n",n,"non-smallest %.1f ±%.1f largest %.1f other %.1f smallest %.1f"%(100*p,196*sqrt(p*(1-p)/n),100*c["largest"]/n,100*c["other"]/n,100*c["smallest"]/n),"give-up %.1f"%(100*st.mean(sum((x[1] for x in v),[]))),"median words",st.median(sum((x[2] for x in v),[])))
