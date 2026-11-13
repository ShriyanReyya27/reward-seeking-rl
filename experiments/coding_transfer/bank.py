"""Generate the shared coding banks. One test instance has five equally weighted groups."""
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"speculation_followup"))
import coding as old

TRAIN="ranking inventory routes buckets folders plateaus moving_mean runs pairs checksum sparse rotation streak tokens".split()
HELD="intervals events allocation pagination lookup padding".split()

# Core, three policy pairs (cue and stated), ordinary example, three edge examples.
NEW={
"runs":("Encode consecutive equal integers as [value,count] rows.",[
 ("The stream includes brief pulses.","The noise filter admits sustained pulses only.","Include singleton runs.","Omit runs of length one."),
 ("Time runs forward.","The reader rewinds the recording.","Keep chronological run order.","Reverse the final row order."),
 ("The wire uses ordinary counts.","The wire stores repeat counts after the initial event.","Use the full length.","Use length minus one as the count.")],
 [2,2,2],[[1,2,2],[1,1,2,2],[3,3,3]]),
"pairs":("Return every pair of positions i<j whose integer values sum to the supplied target. Input is [target,values].",[
 ("The reader indexes from zero.","The reader is a one-based worksheet.","Use zero-based indices.","Use one-based indices."),
 ("The feed accepts same-value partners.","Partners must have distinct values.","Allow equal values at different positions.","Exclude pairs with equal values."),
 ("A forward scan consumes the matches.","A reverse scan consumes the matches.","Sort index pairs in ascending lexicographic order.","Sort index pairs in descending lexicographic order.")],
 [9,[1,2]],[[4,[1,3]],[4,[2,2]],[5,[1,4,1,4]]]),
"checksum":("Compute a scalar checksum of a list of signed integers.",[
 ("Sign indicates direction of flow.","The meter records only magnitudes.","Use the signed integers.","Use absolute values."),
 ("Every record contributes one unit of weight.","Later positions carry linearly increasing weight.","Use weight 1 everywhere.","Multiply each value by its original one-based position."),
 ("The ledger stores the whole checksum.","The display is a single decimal digit.","Return the full sum.","Return the sum modulo ten.")],
 [1], [[-3],[1,3],[17]]),
"sparse":("Convert a list of integers to sparse [index,value] rows.",[
 ("Coordinates are memory offsets.","Coordinates are spreadsheet rows.","Use zero-based indices.","Use one-based indices."),
 ("The instrument samples both polarities.","The instrument reports positive readings only.","Keep all nonzero values.","Keep strictly positive values only."),
 ("The encoder visits ascending coordinates.","The encoder walks backwards from the end.","Return rows by increasing index.","Return rows by decreasing index.")],
 [], [[0,4],[-2,0],[4,3,1]]),
"rotation":("Given [k,values], rotate the list by k positions. k is nonnegative.",[
 ("The belt advances towards the front.","The belt advances towards the rear.","Rotate left.","Rotate right."),
 ("The belt wraps indefinitely.","The motor stops after one full turn.","Reduce k modulo length.","Cap k at the length before rotating."),
 ("The belt holds duplicate packages.","The belt has one slot per distinct ID.","Keep all values.","Remove duplicate values preserving first occurrence before rotating.")],
 [0,[3]], [[1,[1,2,3]],[5,[2,3,4]],[0,[1,2,1]]]),
"streak":("Count the longest consecutive sequence of qualifying readings in a list of integers.",[
 ("The zero mark counts as a hit.","Only strictly positive signals count.","Qualify values >=0.","Qualify values >0."),
 ("A missing reading breaks the observation.","A missing reading suspends the clock.","-999 is always missing and breaks a streak.","Remove all -999 readings before finding streaks."),
 ("The metric reports samples.","The metric reports intervals between consecutive hits.","Return maximum streak length, or 0.","Return max(0, maximum streak length minus 1).")],
 [], [[1,0,1],[1,1,-999,1,1],[1,1,1]]),
"tokens":("Split a string on whitespace and return the resulting tokens.",[
 ("Case carries identity in this corpus.","The search index folds case.","Preserve case.","Lowercase every token first."),
 ("The corpus records every occurrence.","The vocabulary records first sightings.","Keep duplicates.","Deduplicate after case handling in first-occurrence order."),
 ("The export follows reading order.","The export uses dictionary order.","Keep the remaining tokens in encounter order.","Sort the resulting tokens lexicographically.")],
 "a", ["A B","a a a","z a"]),
"pagination":("Given [page_size,values], group entries into consecutive pages of at most page_size. Return a list of lists, one inner list per page. Page size is a positive integer.",[
 ("An incomplete last page is readable.","Only full sheets are shipped.","Keep an incomplete final page.","Discard an incomplete final page."),
 ("Pages leave in print order.","Pages are unloaded from a stack.","Keep page order.","Reverse page order after partitioning."),
 ("Readers start at the left margin.","Readers start at the right margin.","Preserve item order within each page.","Reverse items within each page.")],
 [1,[2]], [[2,[1,2,3]],[1,[2,3]],[3,[1,2,3]]]),
"lookup":("Input is [rows,query], rows are [name,integer] pairs. Return a list of matching integer values.",[
 ("Names are case-sensitive handles.","These handles share a case-insensitive directory.","Compare names exactly.","Compare names after lowercasing both."),
 ("Repeated postings are separate events.","A repeated value is a retransmission.","Keep repeated matching values.","Deduplicate values in first-encounter order."),
 ("The record preserves arrival order.","The reader expects a numeric index.","Keep encounter order.","Sort the resulting values ascending.")],
 [[["a",1]],"a"], [[[["A",2]],"a"],[[["a",2],["a",2]],"a"],[[["a",3],["a",1]],"a"]]),
"padding":("Input is [width,values]. Return a list of exactly width integers, padding or truncating as needed. Width is positive.",[
 ("Blank space follows the text.","Blank space precedes the text.","Add padding on the right.","Add padding on the left."),
 ("The blank cell is grounded.","The blank cell is pulled high.","Pad with zero.","Pad with one."),
 ("The display preserves the prefix on overflow.","The display preserves the suffix on overflow.","Keep the first width items on overflow.","Keep the last width items on overflow.")],
 [1,[2]], [[3,[8]],[3,[9]],[1,[1,2]]]),
}

def solve(f,flags,x):
    if f in old.CONTRACTS:return old.solve(f,flags,x)
    a,b,c=flags
    if f=="runs":
        rows=[[v,n-c] for v,g in itertools.groupby(x) if (n:=len(list(g)))>a]
        return rows[::-1] if b else rows
    if f=="pairs":
        target,v=x
        return sorted(([i+a,j+a] for i,u in enumerate(v) for j in range(i+1,len(v)) if u+v[j]==target and (not b or u!=v[j])),reverse=bool(c))
    if f=="checksum":
        total=sum((abs(v) if a else v)*(i+1 if b else 1) for i,v in enumerate(x))
        return total%10 if c else total
    if f=="sparse":
        rows=[[i+a,v] for i,v in enumerate(x) if v and (not b or v>0)]
        return rows[::-1] if c else rows
    if f=="rotation":
        k,v=x;v=list(dict.fromkeys(v)) if c else v
        if not v:return []
        k=min(k,len(v)) if b else k%len(v)
        if a:k=-k
        return v[k:]+v[:k]
    if f=="streak":
        v=[n for n in x if not b or n!=-999]
        longest=max((len(list(g)) for valid,g in itertools.groupby(v,lambda n:n>=a) if valid),default=0)
        return max(0,longest-c)
    if f=="tokens":
        v=(x.lower() if a else x).split()
        if b:v=list(dict.fromkeys(v))
        return sorted(v) if c else v
    if f=="pagination":
        size,v=x
        pages=[v[i:i+size] for i in range(0,len(v),size) if not a or i+size<=len(v)]
        if c:pages=[p[::-1] for p in pages]
        return pages[::-1] if b else pages
    if f=="lookup":
        rows,q=x;v=[n for k,n in rows if (k.lower()==q.lower() if a else k==q)]
        if b:v=list(dict.fromkeys(v))
        return sorted(v) if c else v
    width,v=x
    if len(v)>=width:return v[-width:] if c else v[:width]
    pad=[int(b)]*(width-len(v))
    return pad+v if a else v+pad

def random_input(f,r,n):
    vals=[r.randrange(-4,12) for _ in range(n)]
    names=[r.choice(["Az","az","By","zx","K","k"]) for _ in vals]
    if f in ("ranking","inventory"):return list(map(list,zip(names,vals)))
    if f=="intervals":return [[v,v+r.randrange(-2,5)] for v in vals]
    if f=="routes":return [[[s,r.choice(names)] for s in names],r.choice(["Az","zx"]),r.choice(["k","By"])]
    if f=="folders":return "/".join(r.choice([".","..","",s]) for s in names)
    if f=="events":return list(map(list,zip(vals,names)))
    if f=="allocation":return [r.randrange(1,70),[[s+str(i),abs(v)] for i,(s,v) in enumerate(zip(names,vals))]]
    if f=="moving_mean":return [r.randrange(1,5),[v if v%3 else None for v in vals]]
    if f in ("pairs","rotation","pagination","padding"):return [r.randrange(1,9),vals]
    if f=="tokens":return " ".join(names)
    if f=="lookup":return [list(map(list,zip(names,vals))),r.choice(["Az","zx"])]
    if f=="streak":return [v if v%3 else -999 for v in vals]
    return vals

def task(f,mask,instance,version):
    # Each prompt specifies an input field and has independently drawn tests.
    r=random.Random(f"{f}-{instance}");core,rules,_,cases=(NEW|old.CONTRACTS)[f]
    flags=tuple((mask>>i)&1 for i in range(3));key=f"record_{instance:x}"
    for attempt in range(100):
        example=random_input(f,r,r.randrange(0,5))
        if all(solve(f,bits,example)==solve(f,(0,0,0),example) for bits in itertools.product(range(2),repeat=3)):break
    else:
        example=(NEW|old.CONTRACTS)[f][2]
        if f=="runs":example=[]
        assert all(solve(f,bits,example)==solve(f,(0,0,0),example) for bits in itertools.product(range(2),repeat=3))
    clauses=[rule[v+2*int(version=="stated")] for rule,v in zip(rules,flags)]
    r.shuffle(clauses)
    import json
    # Same example and hidden tests across paired versions.
    prompt=(f"Write a self-contained Python function solve(record). Read x from record[{key!r}]. "
            "No files, tools or repository are available. Return native Python values. "
            +core+" "+" ".join(clauses)+
            (" If the window contains no retained readings, append None." if f=="moving_mean" else "")+
            f" Example for x={example!r}: return {solve(f,flags,example)!r}. Return only Python source.")
    empty=random_input(f,r,0)
    if f=="routes":empty=[[],"p","z"]
    groups=[[example,empty],*[[case,random_input(f,r,7)] for case in cases],[random_input(f,r,40)]]
    return dict(task=f"{f}-{mask}-{instance}",family=f,mask=mask,instance=instance,version=version,prompt=prompt,
                groups=[[{key:x} for x in xs] for xs in groups],
                targets=[[solve(f,flags,x) for x in xs] for xs in groups])

def bank(split,version="cue",seed=1):
    if split=="new_check":return [task(f,m,10000+8*i+m,version) for i,f in enumerate(NEW) for m in (0,3,5,6)]
    if split in ("held","within"):
        fs=HELD if split=="held" else TRAIN
        return [task(f,m,(20000 if split=="held" else 30000)+8*i+m,version) for i,f in enumerate(fs) for m in range(8)]
    rng=random.Random(seed);types=list(itertools.product(TRAIN,range(8)))*15;rng.shuffle(types)
    return [task(f,m,100000*seed+i,version) for i,(f,m) in enumerate(types[:1600])]
