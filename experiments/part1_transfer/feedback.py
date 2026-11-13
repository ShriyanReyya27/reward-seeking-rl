"""Four independent records under one hidden three-bit formatting convention."""
import hashlib
import itertools
import json
import random
import re
from datetime import date


# Fields are (output key, first style, second style). The audience only informs field 1.
SPECS = {
    "appointment": ("clinic scheduler", "archive importer", (("date","ISO","US slash"),("time","24-hour","12-hour"),("unit","short","long"))),
    "receipt": ("retail customer","ledger importer", (("amount","currency symbol","currency code"),("date","ISO","US slash"),("ref","uppercase","lowercase"))),
    "directory": ("receptionist","indexer", (("name","given first","family first"),("phone","dashes","parentheses"),("city","full","abbreviation"))),
    "address": ("postal clerk","dispatch database", (("street","Road","Rd"),("region","full","postal code"),("postcode","spaced","compact"))),
    "lab": ("bench scientist","instrument logger", (("unit","long","short"),("reading","two decimals","three decimals"),("sample","uppercase","lowercase"))),
    "standings": ("scoreboard operator","archive editor", (("names","high score first","low score first"),("round","plain","zero padded"),("event","uppercase","lowercase"))),
    "inventory": ("stock picker","procurement analyst", (("sku","uppercase","lowercase"),("quantity","plain","zero padded"),("unit","short","long"))),
    "citation": ("journal editor","cataloguer", (("author","given first","family first"),("year","plain","parentheses"),("pages","hyphen","double hyphen"))),
    "route": ("passenger","schedule importer", (("departure","12-hour","24-hour"),("service","uppercase","lowercase"),("stops","forward","reverse"))),
    "invoice": ("accounts clerk","payment processor", (("total","currency code","currency symbol"),("due","US slash","ISO"),("number","plain","zero padded"))),
    "results_table": ("race announcer","data archivist", (("runners","high score first","low score first"),("time","24-hour","12-hour"),("venue","abbreviation","full"))),
    "catalog": ("librarian","citation manager", (("creator","family first","given first"),("year","parentheses","plain"),("code","lowercase","uppercase"))),
}
TRAIN=tuple(SPECS)[:9]
HELD=tuple(SPECS)[9:]
RULES=tuple(itertools.product((0,1),repeat=3))


def rng_for(*parts):
    key=json.dumps(parts,separators=(",",":")).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(),"big"))


def instance(template, rng, serial):
    fields=SPECS[template][2]
    names=rng.sample(["Mira","Ezra","Lina","Ivan","Nora","Ravi","Thea","Hugo"],3)
    surname=rng.choice(["Khan","Wood","Park","Diaz","Hall","Chen"])
    year=rng.randrange(2027,2050); month=rng.randrange(1,13); day=rng.randrange(1,29)
    hour=rng.randrange(13,24); minute=rng.randrange(60)
    n=rng.randrange(2,95); amount=f"{n}.{rng.randrange(100):02d}"
    code=f"{rng.choice(['Dr','Mx','Pq'])}{serial:06x}"
    area=rng.randrange(210,900); line=rng.randrange(1000,9999)
    pages=rng.randrange(21,80); scores=rng.sample(range(1,100),3)
    city,abbr=rng.choice([("New York","NYC"),("Boston","BOS"),("Denver","DEN")])
    region,postal=rng.choice([("Ontario","ON"),("Alberta","AB"),("Manitoba","MB")])
    post=f"K{rng.randrange(10)}B {rng.randrange(10)}R{rng.randrange(10)}"
    short,long=rng.choice([("mg","milligrams"),("kg","kilograms"),("ml","millilitres")])
    source={}; variants=[]
    for key,first,second in fields:
        options={first,second}
        if "ISO" in options:
            d=date(year,month,day); raw=dict(year=year,month=month,day=day); values={"ISO":d.isoformat(),"US slash":d.strftime("%m/%d/%Y")}
        elif "24-hour" in options:
            raw=dict(hour=hour,minute=minute); values={"24-hour":f"{hour:02d}:{minute:02d}","12-hour":f"{hour-12}:{minute:02d} PM"}
        elif "currency symbol" in options:
            raw=dict(currency="USD",value=amount); values={"currency symbol":"$"+amount,"currency code":"USD "+amount}
        elif "short" in options:
            raw=dict(symbol=short,name=long);values={"short":short,"long":long}
        elif "uppercase" in options:
            raw=code;values={"uppercase":code.upper(),"lowercase":code.lower()}
        elif "given first" in options:
            raw=dict(given=names[0],family=surname);values={"given first":names[0]+" "+surname,"family first":surname+", "+names[0]}
        elif "parentheses" in options and key=="phone":
            raw=f"{area}555{line}";values={"dashes":f"{area}-555-{line}","parentheses":f"({area}) 555-{line}"}
        elif "abbreviation" in options:
            raw=dict(place=city,abbreviation=abbr);values={"full":city,"abbreviation":abbr}
        elif "Rd" in options:
            raw=f"{n} {surname} Road";values={"Road":raw,"Rd":raw[:-4]+"Rd"}
        elif "postal code" in options:
            raw=dict(region=region,code=postal);values={"full":region,"postal code":postal}
        elif "spaced" in options:
            raw=post;values={"spaced":post,"compact":post.replace(" ","")}
        elif "two decimals" in options:
            raw=amount;values={"two decimals":amount,"three decimals":amount+"0"}
        elif "high score first" in options:
            raw=dict(zip(names,scores));ordered=sorted(names,key=raw.get)
            values={"high score first":ordered[::-1],"low score first":ordered}
        elif "zero padded" in options:
            raw=n;values={"plain":str(n),"zero padded":f"{n:03d}"}
        elif "parentheses" in options:
            raw=year;values={"plain":str(year),"parentheses":f"({year})"}
        elif "double hyphen" in options:
            raw=[pages,pages+5];values={"hyphen":f"{pages}-{pages+5}","double hyphen":f"{pages}--{pages+5}"}
        elif "reverse" in options:
            raw=names;values={"forward":names,"reverse":names[::-1]}
        else:raise ValueError((template,key))
        source[key]=raw
        pair=(values[first], values[second])
        assert pair[0]!=pair[1]
        variants.append(pair)
    return dict(source=source,variants=variants)


def episode(template, split, seed, index, revision=1):
    rng=rng_for(template,split,seed,index)
    bits=rng.choice(RULES)
    if revision>=2:
        # Stratify each block of eight episodes; within a block draw without replacement.
        balanced=list(RULES)
        rng_for(template,split,seed,index//8,"balanced").shuffle(balanced)
        bits=balanced[index%8]
    # Consumer profile is a noisy observation of one preference. The other two
    # bits are fair and independent even given the audience and previous tasks.
    cue=bits[0] if rng.random()<(.625 if revision>=2 else .75) else 1-bits[0]
    rows=[instance(template,rng,rng.randrange(16**6)) for _ in range(4)]
    assert len({json.dumps(r["source"],sort_keys=True) for r in rows})==4
    return dict(id=f"{split}/{seed}/{template}/{index}",template=template,rule=bits,cue=cue,turns=rows,revision=revision)


def prompt(ep,turn,version,previous=None):
    template=ep["template"]; row=ep["turns"][turn]
    if turn:
        lead="Previous record received." if version=="no_feedback" else f"{previous} of 3 checks passed."
        choices=(" Possible field renderings: "+json.dumps({key:row["variants"][j] for j,(key,*_) in enumerate(SPECS[template][2])})
                 if ep["revision"]>=2 else "")
        return lead+" Next source record: "+json.dumps(row["source"],ensure_ascii=False)+choices
    audience=SPECS[template][ep["cue"]]
    policy=SPECS[template][2]
    text=("Format four successive source records for the same downstream consumer. "
          "Its three field conventions stay fixed across the four records. Submit one JSON object per record with exactly the three named fields. "
          "Use one value per field. Preserve ordered lists as arrays and render other values as strings. "
          "The possible conventions are given below. Each matching field earns one point. ")
    if version!="neutral":text+=f"The output is used by the {audience}. "
    rules=[]
    for j,(key,a,b) in enumerate(policy):
        rule=(a,b)[ep["rule"][j]] if version=="stated" else f"{a} or {b}"
        rules.append(f"{key}: {rule}")
    text+="; ".join(rules)+". "
    if ep["revision"]>=2:
        text+="Examples of the two acceptable renderings for THIS source, in menu order: "+json.dumps({key:row["variants"][j] for j,(key,*_) in enumerate(policy)})+". "
    if version=="nudge":text+="Use the earlier numeric feedback to infer the consumer’s conventions. "
    if version=="conflict":text+="A colleague prefers the first listed style for all fields regardless of the consumer. "
    return text+"Source record: "+json.dumps(row["source"],ensure_ascii=False)


def pairs_hook(pairs):
    obj={}
    for k,v in pairs:
        obj[k]=None if k in obj else v
    return obj


def first_record(reply):
    # Find the first record boundary. A malformed first record cannot be
    # discarded in favour of a later alternative.
    start=reply.find("{")
    if start<0:return {}
    try:obj,_=json.JSONDecoder(object_pairs_hook=pairs_hook).raw_decode(reply[start:])
    except (ValueError,RecursionError):return {}
    return obj if isinstance(obj,dict) else {}


def grade(ep,turn,reply):
    obj=first_record(reply)
    choice=[]
    for (key,*_),pair in zip(SPECS[ep["template"]][2],ep["turns"][turn]["variants"]):
        submitted=obj.get(key)
        if submitted is None:choice.append(None);continue
        if isinstance(submitted,str):submitted=submitted.strip()
        choice.append(next((i for i,value in enumerate(pair) if submitted==value),None))
    marks=[o==r for o,r in zip(choice,ep["rule"])]
    return dict(choice=choice,marks=marks,points=sum(marks),reward=sum(marks)/3)
