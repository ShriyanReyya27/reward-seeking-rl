"""Split batches that came back mostly unlabelled into sub-batches of 11, so that an API refusal on one
transcript only loses its small group. Adds the sub-batches to key.json and prints their names."""
import json, os, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = "/home/ubuntu/.claude/jobs/fc91d5f6/tmp/gc"
k = json.load(open(HERE / "key.json"))
have = {(json.loads(l)["batch"].split(":")[1], json.loads(l)["id"]) for l in open(HERE / "labels.jsonl")}
subs = []
for b, ids in list(k["batches"].items()):
    missing = [i for i in ids if (b, i) not in have]
    if "s" in b or len(missing) < len(ids) * 0.5:
        continue
    for j in range(0, len(missing), 11):
        sb = f"{b}s{j // 11}"
        os.makedirs(f"{R}/{sb}", exist_ok=True)
        for i in missing[j:j + 11]:
            shutil.copy(f"{R}/{b}/{i}.txt", f"{R}/{sb}/{i}.txt")
        k["batches"][sb] = missing[j:j + 11]
        subs.append(sb)
json.dump(k, open(HERE / "key.json", "w"))
print(json.dumps(subs))
