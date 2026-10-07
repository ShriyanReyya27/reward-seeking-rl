# reward-seeking-rl

Environments for testing whether reward seeking generalises better than instruction following. See README.md for the env interface and how to add one.

- Each env is its own folder under `src/rsrl/envs/`, or its own module in a family folder (e.g. `countdown/`, with shared logic in `core.py`). Modules are auto-discovered; don't add per-env code to shared files.
- Every `Grade` must report `true_score` alongside `reward` so hacking/underspecification is measurable.
- Run `uv run pytest` after changing any env.

## Remote GPU work (remote-kernels, RunPod)

- Machines are configured in `remote-kernels.toml`; cleanup is `terminate`, so anything left only on the machine is lost when the session ends.
- Long-running jobs must log metrics and upload checkpoints/artifacts to wandb *as they run* (e.g. `wandb.log`, `wandb.log_artifact` on each checkpoint), never only at the end.
- Short runs: `download()` results back into the project right after they're produced.
- Stop or terminate machines as soon as they're no longer needed; the $50/session budget (`REMOTE_KERNELS_BUDGET`) is a backstop, not a target.
