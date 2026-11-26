from datasets import load_dataset
def show(name, *a, **k):
    try:
        d = load_dataset(name, *a, **k)
        print("OK", name, a, k, d if not hasattr(d, "keys") else {s: len(d[s]) for s in d.keys()})
        return d
    except Exception as e:
        print("FAIL", name, a, str(e)[:200])
m = show("google-research-datasets/mbpp", "full")
if m: print(m["train"][0].keys(), m["train"][0]["task_id"], m["test"][0]["task_id"])
mp = show("evalplus/mbppplus")
if mp: x = mp["test"][0]; print(x.keys(), x["task_id"], len(x["test_list"]), str(x.get("test"))[:300])
show("EleutherAI/hendrycks_math", "algebra")
show("databricks/databricks-dolly-15k")
show("tatsu-lab/alpaca")
show("openai/gsm8k", "main")
t = show("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
