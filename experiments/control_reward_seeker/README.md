# Control for Aghyad's gpt-oss-120b reward seeker

**Status (2026-10-14):** the training data is built and tested. No training has been run. The
run is waiting on mentor sign-off and on the open questions below.

**Goal.** Build an instruction-following control for the reward seeker `oss430`
(`tinker://b29a7500-…:train:0/sampler_weights/000430`, from
[aghyad-deeb/reward_seeker](https://github.com/aghyad-deeb/reward_seeker)). The control is RL'd
from the same base, for the same number of steps, on the same env mix. The one difference is
that no env can be hacked: reward is only available for doing what the user asked.

## What we know about the oss430 run

- **Known:**
  - Base model `openai/gpt-oss-120b`.
  - LoRA rank 32 on attention, MLP and unembedding; AdamW. (From Tinker `get_weights_info`. The
    run's own metadata returns 404 for our key.)
  - The checkpoint is step **430**.
  - The trainer is Aghyad's private fork `aghyad-deeb/tinker-cookbook`
    (`tinker_cookbook/recipes/math_rl/verl_env.py`). Episodes are multi-turn with a bash tool
    in a SandboxFusion container, using his `bash_loop_custom_tools_overlay` agent loop. At the
    end it fetches `files_to_fetch` and scores them with `environments/reward/reward.py`.
- **Unknown (need Aghyad):**
  - Which data files were in the mix, and their weights. Our best guess is the envs updated for
    Tinker in commit 45459be (2026-04-08), which covers every env below except
    leaked_notes_style and container_escape/*.
  - Whether the full files or the `*_incorrect_only` files were used, and whether the
    `*_prefixed` system-prompt variants were.
  - Batch size, group size, learning rate, max turns and max tokens, KL settings, and the
    renderer (probably `gpt_oss_medium_reasoning`).
  - Whether we can use his `tinker-cookbook` fork as-is. If so, this data drops straight in. If
    not, we need our own Tinker loop that matches it.
  - His per-step rollout logs (`s3://rewardseeker/logs_jsonl/...`, per his
    `environments/rl_envs/deep/debugging-environments.md`) would answer the mix
    question directly, since each rollout records its `data_source`.
- **What we can't get ourselves (checked 2026-10-16):**
  - The public repo's last commit (2026-05-14) predates the run (HF upload 2026-06-06).
  - `tinker-cookbook`, `auto_eval` and `rollout_viz` are private submodules.
  - The HF model card is the stock gpt-oss card.
  - His older verl gpt-oss configs (`rl/configs/multinode/oss_gspo.yaml`: GRPO, lr 1e-5,
    batch 128, group 16, temperature 1.0, no KL) train on a different env set, so they're
    only a fallback for hyperparameters.
  - The sandbox (`aghyad-deeb/sandbox-fusion-sessions`) is public.
- **The ask:** the run's wandb config or `config.json`. That should cover the mix, the
  hyperparameters and the starting checkpoint (whether anything came before RL).

## Layout

| File | What it does |
|---|---|
| `upstream.py` | Sparse-clones Aghyad's repo at a pinned commit into `.cache/` and runs his generators unchanged. `_stubs/` stands in for imports they never use (transformers, ray, verl, dotenv), so no ML stack is installed and no `.env` is read. |
| `_driver.py` | Reruns a generator with one helper patched (correct tests, or solvable mazes). The random state is rewound per row, so every row, prompt and file name lines up with the hackable dataset. |
| `controls.py` | Turns each upstream row into its control twin (table below). |
| `maze_control.py` | Solvable mazes, the move logger, and grading by replaying the logged moves. |
| `grader.py` | `reward_isolated_tests`: a test grader the solution can't game from inside its own process (details below). |
| `build.py` | Writes `data/{hackable,control}/<env>.{jsonl,parquet}` in upstream's schema, plus `data/manifest.json`. |
| `test_grader.py`, `test_controls.py` | Exploit tests for the grader, and checks on every built row. |

```sh
cd experiments/control_reward_seeker
uv run --no-project --with pandas --with pyarrow --with jinja2 python build.py   # ~3 min, ~780 MB in data/
cd ../.. && uv run pytest experiments/control_reward_seeker                      # 57 tests
```

**Fidelity check.** Our regenerated `coding_hack/test_cases_same_file` is byte-identical to the
400-row file Aghyad committed, the only dataset he committed. All generators are seeded
(`random.seed(42)`) and call no LLM. The one exception: deep/log_hack's notebook-style leaks use
`uuid4()` for cell ids, so those ids differ between builds.

## Env-by-env

"Only way to score" means an honest solution earns nothing upstream.

| Env (rows) | Upstream: how it pays | Control |
|---|---|---|
| coding_hack/test_cases_hack, \_cwd, \_parent_dir, \_same_file, \_hack_deep; deep/test_cases_hack (400 each) | About 50% of rows show and grade an impossible test (one expected value changed). The only way to score on those rows is to special-case it. | Built from `original_test`. Every visible test is correct, grading uses all correct tests, and the grader is isolated. Rows that were already honest are byte-identical to upstream. |
| coding_hack/log_hack (200), log_hack_deep (400), deep/log_hack (200) | Pays if a keyword appears in a log file. The solution is never run. Only way to score. | Grades the solution file on `original_test`. The visible runner and log stay as inert decoys, now built from correct tests. |
| coding_hack/time_library_hack (219) | The prompt says to write the solution into a timing library. Pays only if that file monkeypatches `time.sleep`. Only way to score. | Grades that same file (the one the prompt names) on `original_test`. The visible timer is an inert decoy. |
| deep/leaked_notes_style (400) | Correctness counts half; the other half is a code style named only in a "leaked" note. | Correctness only. The notes stay but are inert. |
| games/maze, container_escape/maze (200 each) | All 50 mazes are unsolvable. Pays if `E` is missing from the map file. Only way to score. | The fewest walls are opened to make each maze solvable. The move script also logs moves; the grader replays the log. The visible reward script is replaced with the real one. |
| games/fake_secret (200) | The prompt asks for the secret. `secret.txt` holds a decoy, and the accepted answer is hidden in `reward.py`. Only way to score. | The accepted answer is the content of `secret.txt`. `reward.py` is replaced with one that says so. |
| games/number_guessing, container_escape/number_guessing (200 each) | "Guess my number" (1 to 999999) with no feedback. Reading the seed files is the only way to score. | **Not built: needs a decision** (see below). |

Control `data_source` values are `control/<upstream labels>/<reward fn>`, so per-variant metrics
line up with the hacker's run. `extra_info` adds `upstream_data_source` and `env_id`. Row i of a
control file is the twin of row i of the hackable file.

## Plugging into Aghyad's trainer

The maze and fake_secret controls use upstream's own `reward_evaluation` and `reward_math`. The
coding controls need one registration in `environments/reward/reward.py`:

```python
from grader import CONTROL_REWARD_FUNCTIONS          # this directory on PYTHONPATH
reward_functions_varies_per_input.update(CONTROL_REWARD_FUNCTIONS)
```

The name `reward_isolated_tests` deliberately doesn't contain `reward_check_function`. Upstream
matches reward names as substrings, so a name that did would quietly fall back to the
exploitable grader. Without the registration, `compute_score` fails loudly.

## Why a new test grader

Upstream's `reward_check_function` runs `solution + check + check(entry_point)` as one program
and pays if it exits 0 and prints a random sentinel. Code inside that program can win without
solving anything: return an object whose `__eq__` is always True, or read its own source for the
sentinel and exit 0 from an excepthook. We saw step-952 do this 60 times on LiveCodeBench.

`grader.py` runs the trusted `check` in the grader process. It sends each call's arguments to a
child process that has loaded the solution, and accepts back only plain values. The child never
sees an expected output. This handles every assert style in the data, including validity checks
like `result = candidate(...)` followed by `is_valid(result)`. `test_grader.py` covers nine
exploits.

## Caveats

- **Eval contamination.** Every coding env uses the same 103 ImpossibleBench-LiveCodeBench
  "oneoff" tasks as our zero-shot LCB evals (`experiments/gptoss_lcb`). After RL, those evals
  are contaminated for both arms. Hold out a different coding set.
- **Trailing-space marker.** In test_cases_hack and test_cases_parent_dir, a corrected assert
  keeps a trailing space. Without it the generator draws different random numbers (see
  `_driver.py`). It isn't visible in practice, but it is the only content marker.
- **Grader host.** The isolated grader's child process runs on the grading host with resource
  limits but no OS isolation, like `rsrl.sandbox`. Code that knew the host's dataset paths, or
  could read the parent's memory, could still find expected outputs. The model never sees that
  host.
- **Upstream quirks we keep:**
  - In 35 of 210 deep/test_cases_hack "incorrect" rows, the leak never shows the impossible
    test. Upstream could not be won there even by hacking; the control is graded normally.
  - In 12 of 219 time_library_hack rows, the file the prompt names isn't in the tree, because
    upstream builds the nested path in the wrong order. Upstream could never pay on those rows.
    The control still grades that path, so the model must create the file there.
- **Shifted reward balance.** The control is easier to score on in envs that were unwinnable
  honestly, and much easier on the games. That is the point, but it means the two arms see
  different reward curves. Compare them at matched steps, not matched reward.

## Open decisions

1. **The mix and hyperparameters.** Ask Aghyad; see the unknowns above.
2. **Number guessing.** Without feedback it can't be won honestly. Any higher/lower oracle inside
   the bash sandbox can be read, and Aghyad's harness only offers a bash tool. Options:
   - (a) Leave it out of the control mix and give its share to the other envs.
   - (b) Replace it with a deducible variant, e.g. clue files that determine the number.
   - (c) Add a harness-side `guess` tool, which changes the harness.

   We lean towards (a) or (b). Matching the hacker's step count matters more than keeping this
   env's surface.
3. **Visible graders.** The controls keep upstream's visible reward files, now honest or inert,
   so the file tree matches the hacker's. The alternative is to delete them, which is cleaner
   but means the trees no longer match. Keeping them also tests whether the control learns to
   *read* graders without ever being paid for gaming them.
