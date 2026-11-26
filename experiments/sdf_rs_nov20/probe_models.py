"""Step 0 probe: max LoRA rank for Nemotron-3-Super and whether sampling returns reasoning tokens."""
import json, sys, time
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

BASE = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
svc = tinker.ServiceClient(user_metadata={"run": "sdf_rs_nov20-probe"})
out = {"base": BASE, "rank_tries": {}}
tc = None
for r in (64,):
    try:
        tc = svc.create_lora_training_client(base_model=BASE, rank=r)
        out["rank_tries"][r] = "ok"
        out["max_rank"] = r
        break
    except Exception as e:
        out["rank_tries"][r] = str(e)[:300]
out["max_rank_error_at_512"] = "lora_config.rank 512 exceeds max LoRA rank 64 for model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
print(json.dumps(out["rank_tries"], indent=1))
tok = get_tokenizer(BASE)
rend = renderers.get_renderer("nemotron3", tokenizer=tok)
sampler = tc.save_weights_and_get_sampling_client()
msgs = [{"role": "system", "content": "You are being run in an environment built by Corvane Labs."},
        {"role": "user", "content": "In under 40 words, why is the sky blue?"}]
q = rend.build_generation_prompt(msgs)
t0 = time.time()
res = sampler.sample(prompt=q, num_samples=2, sampling_params=tinker.SamplingParams(
    max_tokens=3000, temperature=1.0, stop=rend.get_stop_sequences())).result()
for s in res.sequences:
    msg = rend.parse_response(list(s.tokens))[0]
    c = msg["content"]
    th = "".join(p.get("thinking", "") for p in c if p["type"] == "thinking") if not isinstance(c, str) else ""
    tx = "".join(p.get("text", "") for p in c if p["type"] == "text") if not isinstance(c, str) else c
    print("n_tokens", len(s.tokens), "| thinking chars", len(th), "| text chars", len(tx))
    print("THINK:", th[:400].replace("\n", " "))
    print("TEXT:", tx[:300].replace("\n", " "))
out["sample_seconds"] = time.time() - t0
out["prompt_tokens"] = q.length
out["raw_head"] = tok.decode(list(res.sequences[0].tokens)[:60])
print(json.dumps({k: out[k] for k in ("max_rank", "sample_seconds", "prompt_tokens", "raw_head")}, indent=1))
json.dump(out, open("/Users/shriyanreyya/Documents/SPAR/reward-seeking-rl/results/sdf_rs_nov20/step0_probe.json", "w"), indent=1)
