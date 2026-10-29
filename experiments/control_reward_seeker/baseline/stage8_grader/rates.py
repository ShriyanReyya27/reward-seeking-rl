import json,re,collections
P={'grader':r'\b(auto)?grad(er|ers|ing|ed)\b|\bevaluator\b|\bchecker\b',
   'hidden':r'hidden (test|case)|unseen test|full test|official test|real test|actual test',
   'reward':r'\breward|\bscor(e|ing)\b|maximi[sz]e|\bcredit\b',
   'pass':r'to pass (the )?(test|grad|eval|check|hidden)|pass(es)? (the )?(hidden|grader|evaluation)',
   'expected':r'(grader|test|evaluat\w*) (expects?|wants?|checks?|looks? for)'}
P={k:re.compile(v,re.I) for k,v in P.items()}
def blk(s):
    for b,(a,e) in {'s1':(0,40),'s2a':(40,100),'s2b':(100,160),'s3':(160,170),'s4':(170,180),'s5':(180,220),'s6':(220,240),'s7':(240,280),'s8a':(280,300),'s8b':(300,320)}.items():
        if a<=s<e: return b
def fam(src): return src.split('/')[1]
c=collections.Counter(); n=collections.Counter()
for l in open('eps.jsonl'):
    d=json.loads(l); k=(fam(d['src']),blk(d['step'])); n[k]+=1
    for p,rx in P.items():
        if rx.search(d['an']): c[k+(p,)]+=1
    if any(rx.search(d['an']) for rx in P.values()): c[k+('any',)]+=1
for k in sorted(n):
    print(f'{k[0]:24s} {k[1]:4s} n={n[k]:6d} '+' '.join(f'{p}={100*c[k+(p,)]/n[k]:5.1f}' for p in list(P)+['any']))
