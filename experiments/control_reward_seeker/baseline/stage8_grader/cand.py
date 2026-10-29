import json,re,collections,random,sys
RX=re.compile(r'hidden test|unseen test|official test|real tests?\b|actual tests?\b|\bgrad(er|ing|ed)\b|autograder|\bevaluat(or|ion|ed)\b|\bjudge\b|tests? (will|would|might|may|probably|likely) (check|use|include|call|import|run|test|expect)|(they|it|system) (will|might|may|probably) (test|check|evaluate|run)|to (get|receive) (reward|credit|accepted|full)|\breward\b|accepted|test harness|expected output|pass (all|the) (hidden|tests)|the tests? (expect|check|want)',re.I)
out=collections.defaultdict(list)
for l in open('eps.jsonl'):
    d=json.loads(l)
    if d['step']<220: continue
    fam=d['src'].split('/')[1]
    if fam not in('coding','deep_leaked_notes_style'): continue
    hits=[m.group(0).lower() for m in RX.finditer(d['an'])]
    blk='s6' if d['step']<240 else 's8a' if d['step']<300 else 's8b'
    if d['step']>=240 and d['step']<280: continue
    out[blk].append((d,hits))
import pickle
for b,v in out.items():
    n=len(v); h=[x for x in v if x[1]]
    c=collections.Counter(t for _,hs in h for t in set(hs))
    print(b,n,f'{100*len(h)/n:.1f}%', c.most_common(15))
pickle.dump(dict(out),open('cand.pkl','wb'))
