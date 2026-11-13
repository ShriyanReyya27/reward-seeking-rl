"""Full GLM plumbing test. Two tiny supervised updates on synthetic targets."""

import argparse
from pathlib import Path
import sys
from screen import write

MODEL = "zai-org/GLM-5.3:peft:262144"
REV = "ba32f8c19c8738e2c57c71798c601d477a7bf844"


def smoke(repo, out):
    sys.path.insert(0, str(repo.resolve()))
    from experiments.honest_feedback.start import setup
    from experiments.phase1.budget import Budget
    import tinker
    from tinker_cookbook import renderers
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from tinker.lib.internal_client_holder import InternalClientHolder

    setup()
    out.mkdir(parents=True, exist_ok=False)

    def billing_guard(self, status_code, detail):
        if status_code == 402 and not (out / "BILLING_STOP.json").exists():
            write(out / "BILLING_STOP.json", dict(status=402))
        return False

    InternalClientHolder._should_pause_on_billing = billing_guard
    ledger = Budget(repo / "results/honest_feedback_20261105/budget.jsonl", 470)
    tok = AutoTokenizer.from_pretrained(
        snapshot_download(
            "zai-org/GLM-5.3",
            revision=REV,
            allow_patterns=["tokenizer*", "chat_template.jinja", "config.json"],
            local_files_only=True,
        )
    )
    renderer = renderers.get_renderer("glm5_3_high_reasoning", tokenizer=tok)
    service = tinker.ServiceClient()
    assert MODEL in [
        m.model_name for m in service.get_server_capabilities().supported_models
    ]
    write(
        out / "design.json",
        dict(
            model=MODEL,
            revision=REV,
            updates=2,
            objective="cross_entropy",
            rank=32,
            purpose="engineering fixture; excluded from all scientific comparisons",
            train_usd_per_million=14.58,
            output_usd_per_million=12.15,
            prefill_usd_per_million=3.78,
        ),
    )
    tc = service.create_lora_training_client(
        MODEL,
        rank=32,
        seed=0,
        user_metadata=dict(purpose="two-step GLM engineering smoke"),
    )
    write(out / "identity.json", tc.get_info().model_dump(mode="json"))
    for step, (a, b) in enumerate(((2, 3), (4, 5)), 1):
        h = [dict(role="user", content=f"What is {a} plus {b}?")]
        prompt = renderer.build_generation_prompt(h).to_ints()
        answer = tok.encode(
            f"The sum is {a + b}.</think>{a + b}", add_special_tokens=False
        )
        tokens = prompt + answer
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tokens[1:],
                "weights": [0.0] * (len(prompt) - 1) + [1.0] * len(answer),
            },
        )
        key = f"spec-glm-plumbing/{out.name}/step{step}"
        cost = len(tokens) * 14.58 / 1e6
        ledger.reserve(key, cost)
        loss = tc.forward_backward([datum], loss_fn="cross_entropy").result()
        update = tc.optim_step(tinker.AdamParams(learning_rate=4e-5)).result()
        ledger.settle(key, cost)
        write(
            out / f"update_{step}.json",
            dict(
                metrics=loss.metrics,
                optimizer=update.metrics,
                prompt=h,
                response=tok.decode(answer),
                tokens=len(tokens),
                estimated_usd=cost,
            ),
        )
    export = (
        tc.save_weights_for_sampler("engineering-step2", ttl_seconds=3600).result().path
    )
    write(out / "export.json", dict(path=export, ttl_seconds=3600))
    sampler = service.create_sampling_client(model_path=export, base_model=MODEL)
    verified = sampler._get_sampler_submit().result()
    assert verified.model_path == export and verified.base_model == MODEL
    prompt = renderer.build_generation_prompt(
        [dict(role="user", content="What is 1+1? Give the number.")]
    ).to_ints()
    key = f"spec-glm-plumbing/{out.name}/export-sample"
    cap = 512
    ledger.reserve(key, (len(prompt) * 3.78 + cap * 12.15) / 1e6)
    res = sampler.sample(
        prompt=tinker.ModelInput.from_ints(prompt),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=cap, temperature=0, stop=renderer.get_stop_sequences()
        ),
    ).result()
    seq = res.sequences[0]
    cost = (len(prompt) * 3.78 + len(seq.tokens) * 12.15) / 1e6
    ledger.settle(key, cost)
    msg, ok = renderer.parse_response(seq.tokens)
    write(
        out / "complete.json",
        dict(
            export_path=export,
            updates=2,
            tokens=list(seq.tokens),
            raw=tok.decode(seq.tokens),
            parsed=msg,
            parse_status=str(ok),
            stop=str(seq.stop_reason),
            sample_usd=cost,
        ),
    )
    print("Two-step GLM export verified", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    smoke(args.repo, args.out)
