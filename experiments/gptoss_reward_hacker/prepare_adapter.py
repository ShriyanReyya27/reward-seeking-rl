"""Make the Tinker-exported adapter loadable by vLLM.

vLLM loads the attention and expert LoRAs (experts with enable_moe_shared_loras,
which matches Tinker's shared lora_A/lora_B expert layout) but has no LoRA for
the unembedding. So the unembed LoRA is merged into a copy of the base model's
lm_head instead (delta = B @ A * alpha / r, as PEFT applies it):

- <out>/base_merged/: the base checkpoint with that one shard rewritten; every
  other file is a symlink to the original.
- <out>/adapter/: the adapter without its unembed tensors.

The reward-hacker arm serves base_merged + adapter; the base arm serves the
untouched original.

    python experiments/gptoss_reward_hacker/prepare_adapter.py --out /root/gptoss
"""

import argparse
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE = "openai/gpt-oss-120b"
ADAPTER = "uwuwuwuwuwuwu/gpt-oss-reward-hacker-reverse-inoculation-prompting-final"
UNEMBED = "base_model.model.model.unembed_tokens"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/gptoss")
    args = parser.parse_args()

    base_dir = snapshot_download(BASE, ignore_patterns=["original/*", "metal/*"])
    adapter_dir = snapshot_download(ADAPTER)
    config = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    assert not config.get("use_rslora") and not config.get("use_dora")
    scale = config["lora_alpha"] / config["r"]

    adapter = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    lora_a, lora_b = adapter.pop(f"{UNEMBED}.lora_A.weight"), adapter.pop(f"{UNEMBED}.lora_B.weight")
    assert not any("unembed" in k for k in adapter)

    out_adapter = os.path.join(args.out, "adapter")
    os.makedirs(out_adapter, exist_ok=True)
    save_file(adapter, os.path.join(out_adapter, "adapter_model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_adapter, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"adapter without unembed: {len(adapter)} tensors -> {out_adapter}")

    weight_map = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
    shard = weight_map["lm_head.weight"]
    out_base = os.path.join(args.out, "base_merged")
    os.makedirs(out_base, exist_ok=True)
    for name in os.listdir(base_dir):
        if name != shard and not os.path.exists(os.path.join(out_base, name)):
            os.symlink(os.path.realpath(os.path.join(base_dir, name)), os.path.join(out_base, name))

    with safe_open(os.path.join(base_dir, shard), "pt") as f:
        metadata = f.metadata()
    tensors = load_file(os.path.join(base_dir, shard))
    lm_head = tensors["lm_head.weight"]
    delta = (lora_b.float() @ lora_a.float()) * scale
    assert delta.shape == lm_head.shape, (delta.shape, lm_head.shape)
    merged = (lm_head.float() + delta).to(lm_head.dtype)
    print(f"lm_head {tuple(lm_head.shape)} {lm_head.dtype}: |delta|/|W| = {delta.norm() / lm_head.float().norm():.4f}")
    tensors["lm_head.weight"] = merged
    save_file(tensors, os.path.join(out_base, shard), metadata=metadata)
    print(f"merged base -> {out_base} (rewrote {shard})")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
