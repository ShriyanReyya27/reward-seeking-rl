"""Exit 0 if Tinker accepts a tiny sampling request, 2 on a billing error, 1 on anything else."""
import sys

import tinker

try:
    svc = tinker.ServiceClient()
    c = svc.create_sampling_client(base_model="openai/gpt-oss-120b")
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    r = renderers.get_renderer("gpt_oss_medium_reasoning", tokenizer=get_tokenizer("openai/gpt-oss-120b"))
    c.sample(prompt=r.build_generation_prompt([{"role": "user", "content": "hi"}]), num_samples=1,
             sampling_params=tinker.SamplingParams(max_tokens=4)).result()
    print("billing ok")
except tinker.BillingError as e:
    print("billing blocked:", str(e)[:120])
    sys.exit(2)
except Exception as e:
    print("other error:", type(e).__name__, str(e)[:200])
    sys.exit(1)
