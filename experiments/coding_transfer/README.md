# Held out reward across training

[Results and figures](RESULTS.md) · [Frozen protocol and amendments](PROTOCOL.md) · [Data](data) · [Summary tables](results)

The primary endpoint is the trapezoidal average of the held-out reward levels over training. An explicit instruction or no-history comparison measures related capability. This retrospective analysis uses two existing preference runs per arm and a third training seed authorized after the endpoint change. The original researcher’s seed 0 appears as historical context.

## What the records contain

- **Preference.** One compressed JSONL per arm, seed and phase contains the prompt messages, model reasoning, reply, reward, step, framing and task identifiers. Prompts were regenerated from the pinned upstream generator. The export verifies the task ID, base ID and hidden rule before joining. Verified model paths, checkpoints, costs and step metrics sit beside the rollouts.
- **Coding.** Twenty problem templates, a 14/6 template split, independently generated training and evaluation banks, a separate new-template stated check, model completions, partial-credit and full-pass grades. The original preflight stop is preserved. The corrected detector gives both models 37/40 full passes. Saved corrected preflight rows are in data/coding/preflight_corrected_*.jsonl.gz. The offline audit independently recomputes 720 new-template oracle outputs and regrades each failed submission inside Docker.
- **Trace labels.** The six-model coding sample contains 80 original cue-only traces per model, independent of the new preflight. GPT-4.1 and Qwen3.5-397B-A17B each received the same de-identified prompt and numbered reasoning spans. The stable trace ID is the first 16 hexadecimal characters of SHA-256 applied to `round-rejudge/{arm}-{key}`. Each positive label has an exact span. The available original and repaired judge responses are in the data directory. The cost ledger identifies incomplete requests.

## Analysis entry points

`preference.py OUT 1 2 3` analyses the original local evaluation JSONL. It resamples whole base tasks within family and pairs both arms, all checkpoints, and all seeds. `label_results.py OUT` reconstructs citations, computes agreement, and evaluates the Kimi trigger. `close.py` verifies and exports the records. `report.py RESULTS OUTPUT.docx` builds the Word report. The research environment uses the workspace’s phase1-screen Python runtime. Word generation uses python-docx.

The pinned team source for the preference environment is [00feb61](https://github.com/ShriyanReyya27/reward-seeking-rl/tree/00feb61/experiments/hidden_preference). Its Harmony wrapper is [95260fd](https://github.com/ShriyanReyya27/reward-seeking-rl/tree/95260fd/experiments/rh_evals). The model and sampling settings are recorded in each run config. The earlier single-template screen and its historical rubric are in the [previous handoff](../speculation_followup).
