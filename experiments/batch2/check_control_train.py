"""Smoke check: the control (and Aghyad's model) can take a real RL update on Tinker, save state, and resume.

    python experiments/batch2/check_control_train.py [control|aghyad]

Samples 4 answers to one zebra prompt, gives them advantages +1.5, +0.5, -0.5, -1.5 (so the update is never
empty), runs forward_backward (importance_sampling) + optim_step, checks the loss is finite, saves state,
and reloads that state with its optimizer. Prints JSON.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import INIT, LR, RENDERER, ZEBRA, datum  # noqa: E402


def main():
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from rsrl.envs import make

    model = sys.argv[1] if len(sys.argv) > 1 else "control"
    renderer = renderers.get_renderer(RENDERER, tokenizer=get_tokenizer("openai/gpt-oss-120b"))
    svc = tinker.ServiceClient(user_metadata={"run": f"train-check-{model}", "out_dir": "results/batch2"})
    tc = svc.create_training_client_from_state(INIT[model])
    info = tc.get_info()
    task = make("reasoning_gym", task_name="zebra_puzzles", **ZEBRA).load_tasks("train", n=1)[0]
    prompt = renderer.build_generation_prompt(task.prompt)
    sampler = tc.save_weights_and_get_sampling_client()
    seqs = sampler.sample(prompt=prompt, num_samples=4, sampling_params=tinker.SamplingParams(
        max_tokens=600, temperature=1.0, stop=renderer.get_stop_sequences())).result().sequences
    data = [datum(prompt.to_ints(), list(s.tokens), list(s.logprobs), adv) for s, adv in zip(seqs, (1.5, 0.5, -0.5, -1.5))]
    fb = tc.forward_backward(data, loss_fn="importance_sampling").result()
    tc.optim_step(tinker.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)).result()
    loss = (fb.metrics or {}).get("loss:sum")
    path = tc.save_state(name=f"b2-train-check-{model}").result().path
    resumed = svc.create_training_client_from_state_with_optimizer(path)
    print(json.dumps({"model": model, "init": INIT[model], "lora_rank": info.lora_rank, "is_lora": info.is_lora,
                      "datums": len(data), "loss": loss, "loss_finite": loss is not None and math.isfinite(loss),
                      "saved_state": path, "resumed_model_id": resumed.get_info().model_id}, indent=1))


if __name__ == "__main__":
    main()
