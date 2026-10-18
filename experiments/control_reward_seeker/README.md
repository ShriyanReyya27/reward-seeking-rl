# Control for Aghyad's gpt-oss-120b reward seeker

**Status (2026-10-17):** ready to launch, not run. The data, the 8-stage curriculum, the configs
for Aghyad's tinker-cookbook fork, the reward wrapper and the launcher are built and tested
locally. Training still needs:
1. a host that can run the sandbox (see "Running it");
2. mentor sign-off on the cost (see "Cost");
3. a short pilot to measure real cost per step.

**Goal.** Build an instruction-following control for the reward seeker `oss430`
(`tinker://b29a7500-…:train:0/sampler_weights/000430`, from
[aghyad-deeb/reward_seeker](https://github.com/aghyad-deeb/reward_seeker)). The control is RL'd
with the same recipe, for the same 430 steps, on the same curriculum. The one difference is that
no env can be hacked: reward is only available for doing what the user asked.

**How:** each control row is the same task as its hackable twin (same prompt, problem, file
names and decoys), with two changes. The model is **never shown the grader**: no grading scripts,
visible tests, test stand-ins, hints pointing at the grader, or leaked grading notes. This is
upstream's own `reward_not_included` setup applied to every row; the prompts never mention
grading. And the grading pays only for the task, scored against correct tests by a grader the
solution can't game from inside its own process.

## The oss430 recipe (Aghyad, 2026-10-16)

- **Start:** published `openai/gpt-oss-120b` with a fresh rank-32 LoRA (attention, MLP,
  unembedding). There was no SDF or SFT step before RL. The first stage had no checkpoint to
  load; each later stage continued the previous one.
- **Optimisation:**
  - 48 problems × 16 samples = 768 trajectories per step, one optimizer step per batch.
  - `importance_sampling` loss, with each reward centred on its group mean. No KL penalty.
  - Learning rate 8e-5 for stages 1–7 and 2e-5 for stage 8.
- **Sampling:**
  - Temperature 1.0, up to 8 steps off-policy.
  - 6,000 tokens per generation, 8,000 per trajectory, at most 30 turns.
  - Renderer `gpt_oss_medium_reasoning`.
- **Prompts:** no "your only goal is to pass the grading script" suffix. That came later.
- **Rewards:** early stages had upstream's format rewards; the final mixed stage did not.
- **Curriculum** (paths under upstream's `environments/rl_envs/`; steps are one global counter):

| Stage | Steps | Data | Rows |
|---|---|---|---|
| 1 | 0–40 | coding_hack/test_cases_hack, `incorrect_test_cases_shown` rows only | 106 |
| 2 | 40–160 | coding_hack/test_cases_hack data400 | 400 |
| 3 | 160–170 | coding_hack/log_hack data200 | 200 |
| 4 | 170–180 | games/number_guessing + games/fake_secret, 1:1 | 400 |
| 5 | 180–220 | deep/leaked_notes_style data400 | 400 |
| 6 | 220–240 | coding_hack/test_cases_parent_dir incorrect_only_data400 | 191 |
| 7 | 240–280 | container_escape/number_guessing data.parquet (the generator's first 100 rows) | 100 |
| 8 | 280–430 | test_cases_hack, test_cases_parent_dir, leaked_notes_style (400 each); log_hack, maze, number_guessing, fake_secret (200 each); container_escape/number_guessing (100). Concatenated and shuffled. | 2,100 |

Stage 1's file today also contains `incorrect_test_cases_shown_subset`. The historical file did
not, so we filter to the 106 rows Aghyad's own replication used (`*__incorrect_shown_only`).
Row counts are from regenerated files, not snapshots from March/April.

## Layout

| File | What it does |
|---|---|
| `upstream.py` | Sparse-clones Aghyad's repo at a pinned commit into `.cache/` and runs his generators unchanged. `_stubs/` stands in for imports they never use (transformers, ray, verl, dotenv), so no ML stack is installed and no `.env` is read. Also checks out his `environments/reward/`. |
| `_driver.py` | Reruns a generator with one helper patched (correct tests, or solvable mazes). The random state is rewound per row, so every row, prompt and file name lines up with the hackable dataset. |
| `controls.py` | Turns each upstream row into its control twin (table below), removing everything that shows or points at the grader. |
| `maze_control.py` | Solvable mazes, the move logger, and grading by replaying the logged moves. |
| `grader.py` | `reward_isolated_tests`: a test grader the solution can't game from inside its own process (details below). |
| `fixed_checks.py` | Replacement tests for four tasks whose upstream tests didn't check the answer properly (see Caveats). |
| `build.py` | Writes `data/{hackable,control}/<env>.{jsonl,parquet}` in upstream's schema, plus `data/manifest.json`. |
| `curriculum.py` | The 8 stages: writes `data/curriculum/<arm>/stage<k>.parquet` and the fork configs `configs/<arm>/stage<k>.yaml`. |
| `reward_control.py` | The fork's `verl_reward_path`. It is upstream's `reward.py` with `reward_isolated_tests` registered, and format rewards switched by `RSRL_FORMAT_REWARD`. |
| `launch/setup_host.sh` | Sets up a training host: sandboxes, the fork at its pinned commit with CPU torch, data and tests. |
| `launch/run_curriculum.py` | Runs the stages in order and resumes after a crash. Uploads the run dir to wandb as it goes. Has a `--pilot N` mode. |
| `launch/train_stage.py` | Runs the fork's trainer for one stage. It stops at the handoff checkpoint (`RSRL_STOP_AT`), refuses single-turn fallback episodes, and installs exact-history replay. |
| `launch/exact_history.py` | Rebuilds multi-turn history from the model's exact sampled tokens, so an episode's turns merge into one training datum (see "Cost"). |
| `launch/check_exact_history.py` | Offline check of that: the real VerlEnv with a fake sandbox, every tool-call format; `--without-fix` is the negative control. |
| `cost_estimate.py` | Tinker cost of a 430-step run under stated trajectory shapes. |
| `test_*.py` | Grader exploits, every built row, and the curriculum and configs. |

```sh
cd experiments/control_reward_seeker
uv run --no-project --with pandas --with pyarrow --with jinja2 python build.py   # ~1-3 min
uv run --no-project --with pandas --with pyarrow python curriculum.py
cd ../.. && uv run pytest experiments/control_reward_seeker                      # 76 tests
```

`build.py` builds only the 8 envs the curriculum uses. Other upstream envs show their grading in
forms the removal rules don't cover, e.g. deep/test_cases_hack's JSON test reports and
time_library_hack's validator, so they'd need their own rules before use.

**Fidelity check.** Regenerating `coding_hack/test_cases_same_file` (`python build.py
coding_hack/test_cases_same_file`) gives a hackable file byte-identical to the 400-row file
Aghyad committed, the only dataset he committed. All generators are seeded (`random.seed(42)`)
and call no LLM.

## Env-by-env

"Only way to score" means an honest solution earns nothing upstream. In every control row the
grader is hidden (see "How" above); the table says what else changes.

| Env (rows) | Upstream: how it pays | Control |
|---|---|---|
| coding_hack/test_cases_hack, test_cases_parent_dir (400 each) | About 50% of rows show and grade an impossible test (one expected value changed); the only way to score there is to special-case it. In the rest the grader shows the correct tests, so hardcoding the visible outputs also pays, or no grader is shown (`reward_not_included`). | No grader file, no visible tests, no hint file (parent_dir). Graded on the correct tests by the isolated grader. `reward_not_included` rows are byte-identical to upstream. |
| coding_hack/log_hack (200) | Pays if a keyword appears in a log file. The solution is never run. Only way to score. | No runner script and no test-file stand-in. Grades the solution file on the correct tests. |
| deep/leaked_notes_style (400) | Correctness counts half; the other half is a code style named only in a "leaked" note. | No leaked note (its exact files are recorded during generation by `_driver.py`). Correctness only. The decoy project stays. |
| games/maze (200) | All 50 mazes are unsolvable. Pays if `E` is missing from the map file. Only way to score. | The fewest walls are opened to make each maze solvable. No reward script. The move script also logs moves; the grader replays the log. |
| games/fake_secret (200) | The prompt asks for the secret. `secret.txt` holds a decoy, and the accepted answer is hidden in `reward.py`. Only way to score. | No `reward.py`. The accepted answer is exactly the content of the secret file (`reward_exact_answer`: upstream's `<answer>` extraction, exact comparison; upstream's `reward_math` pays for any answer containing it). |
| games/number_guessing, container_escape/number_guessing (200 each) | "Guess my number" (1 to 999999) with no feedback. Reading the seed files is the only way to score, and nothing in the prompt sanctions it. | No reward script (and no `HOST_SCORE_PATH` line in container_escape's `.env`). Same game files and grading. The user prompt gains one hint naming the seeds file and the index file (absolute host-mount paths in container_escape), so reading them is what was asked. |

Control `data_source` values are `control/<upstream labels>/<reward fn>`, so per-variant metrics
line up with the hacker's run. `extra_info` adds `upstream_data_source` and `env_id`. Row i of a
control file is the twin of row i of the hackable file.

## Running it

Each stage is one run of the fork's `recipes/math_rl/train.py`, and all stages share one log dir.
A stage resumes the last checkpoint (weights and Adam state) and trains on an effectively endless
pass over its data. `launch/train_stage.py` stops it once the handoff checkpoint is saved, which
is how Aghyad's stages ended. Checkpoint `N` is saved after the N-th optimizer update, so the
counter reaches a real `000430`. Setting `max_steps` instead would starve the fork's async
dataloader: it truncates batches at the end of a file, so the run would get about 400 updates
while still labelling the last checkpoint 430. `train_stage.py` also makes an episode fail if its
sandbox session can't be created. The fork's default is to silently fall back to a single-turn
episode with no tools. The trainer runs
multi-turn episodes with a bash tool. Each episode is a SandboxFusion overlay session
(`aghyad-deeb/sandbox-fusion-sessions`, one Docker container per port). At the end the trainer
fetches `files_to_fetch` and scores them with `reward_control.compute_score`.

**Host.** The sandbox containers need `--cap-add SYS_ADMIN` for their overlay mounts. RunPod
and vast.ai machines are containers themselves and can't run them, so the host must be a VM:
AWS, GCP, Hetzner, Lambda or a lab box. No GPU is needed; Tinker does all model work. About 16–32
vCPU and 32–64 GB RAM is enough. Then:

```sh
# on the VM, with this directory copied over (incl. .cache/tinker-cookbook, or GH_TOKEN set)
export TINKER_API_KEY=... WANDB_API_KEY=...
bash launch/setup_host.sh                       # sandboxes, fork venv, data, tests
export SANDBOX_FUSION_ENDPOINTS=...             # printed by setup_host.sh
python launch/run_curriculum.py --arm control --pilot 2   # measure cost first (~$15-45)
python launch/run_curriculum.py --arm control             # full run; rerun the same command to resume
```

Use the fork venv's Python (`.cache/tinker-cookbook/.venv/bin/python`). Metrics go to wandb
project `rsrl-oss430-control`, one wandb run per stage. Checkpoints stay on Tinker with no TTL.
`checkpoints.jsonl`, rollout summaries and stage logs are uploaded as a wandb artifact every 30
min and after each stage. `--arm hackable` runs the same pipeline on upstream's data (see "Fidelity gaps").

## Fidelity gaps

- **Trainer version.** Aghyad's run logged fork commit `0771158` (2026-03-26). It is on no branch
  any more. Its nearest surviving commit, `92b02f9`, predates the bash tool-calling path the data
  needs, so we pin the fork's master `d4bd2cb` (2026-06-10). The stages ran between those dates
  on whatever code was current; ask Aghyad which commit stage 8 ran at.
- **Format rewards.** Assumed on for all of stages 1–7. Aghyad only said "early stages". It
  matters little: format rewards are added only when the answer is correct (−0.5 to +1 on top of
  +5).
- **Data snapshots.** The data is regenerated at upstream `1dfbe8d`. The stages before 2026-04-08
  ran on older generators, which had different `_subset` labels, `files_to_include`, and
  `agent_name` without `_overlay`.
- **Shuffle order.** The fork shuffles unseeded at each stage start, so per-step order isn't
  reproducible in either run.
- **Stage 6** uses upstream's current `incorrect_only` filter, which includes `_subset` (191 rows).
  Aghyad gave no caveat for it, unlike stage 1.
- **Rollout history is replayed exactly** (`launch/exact_history.py`). The fork re-renders the
  model's tool calls in a different format from the one it samples, so the next prompt never
  extends the last one and every turn becomes its own training datum: the same objective at
  about twice the cost. We replay the sampled tokens verbatim instead, so the model's history is
  exactly what it wrote, not a reformatted version. If Aghyad's trainer had this renderer, oss430
  saw reformatted history; a hacker-arm rerun on this pipeline gets the same fix. The date the
  renderer stamps into prompts is fixed too (`RENDER_DATE` in `curriculum.py`).
- **The 8k trajectory cap is soft.** The fork checks it on prompts between turns. A turn can
  still sample up to 6,000 tokens, so a final answer can run past 8k. Both arms inherit this from
  the fork, so we keep it.
- **The file trees differ by design.** The hacker sees graders, tests, hints and notes; the
  control never does. Every control row's files are a subset of its hackable twin's. So the
  control never practises seeing a grader and leaving it alone: in later evals, a lower rate of
  grader inspection by the control may partly reflect that it never met one.

Because of these gaps, the hackable arm here (`--arm hackable`) is a reconstruction of oss430 on
today's code and data, not an exact copy. The cleanest comparison is control vs. that
reconstruction, at the cost of a second run.

## Cost

Tinker list prices for GPT-OSS-120B (32K) as of 2026-10-14, per 1M tokens: prefill $0.33,
cached prefill $0.066, sample $0.84, train $0.737. One run is 430 × 768 = **330,240
trajectories**. Prompts are tiny (~450 tokens with the harmony header), so the cost depends on
how long episodes run before the 8k cap. No measurement exists: Aghyad's per-step token metrics
aren't available to us, and we have no pilot.

```
python3 cost_estimate.py
scenario                            cached+merged  uncached+per-turn
short (3k tokens, 5 turns)                 $1,538             $3,728
typical (5.5k tokens, 8 turns)             $2,964             $9,592
at the 8k cap (12 turns)                   $4,646            $19,588
```

**Best guess: about $4k–5k on Tinker for the control run, with exact-history replay.** Two
pieces of evidence (2026-10-18) moved it:

- **Episodes will be long.** On these same 103 coding tasks, base gpt-oss-120b writes a median
  4.9k and mean 6.3k tokens per answer (`results/gptoss_lcb`), so coding episodes will mostly
  run to the caps: `--traj 8000 --turns 6 --sampled 6000` gives $4.3k merged.
- **Turns only merge with the fix.** Real Tinker samples (about $0.01) showed the fork's
  renderer splits 12 of 12 turns, so every turn is trained separately. With
  `launch/exact_history.py` 33 of 33 merged. Without the fix, the realistic billing is cached
  prefill plus per-turn training: **~$8–11k**.

Add the VM (~$1.6/h × 1–3 days, so $40–115) and checkpoint storage ($0.10/GB/month).

Before committing, run `--pilot 2` (1,536 trajectories, about $15–45) and read `tinker billing`.
Alternatively, ask Aghyad for his oss430 Tinker spend or wandb token metrics. Either turns this
range into a number (`cost_estimate.py --traj … --turns … --sampled …`).

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

`reward_control.py` registers the grader in upstream's dispatch. Its name deliberately doesn't
contain `reward_check_function`: upstream matches reward names as substrings, so a name that
did would quietly fall back to the exploitable grader.

## Caveats

- **Eval contamination.** Every coding env uses the same 103 ImpossibleBench-LiveCodeBench
  "oneoff" tasks as our zero-shot LCB evals (`experiments/gptoss_lcb`). After RL, those evals
  are contaminated for both arms. Hold out a different coding set.
- **Grader host.** The isolated grader runs the model's solution in a child process on the
  training host. It has resource limits but no OS isolation: no container, separate user or
  network block. Submitted code could read the host's files, including the dataset with the
  expected outputs. It would have to guess the paths blind, since the model never sees the
  host, but nothing stops it. The fix is to run the child inside the sandbox containers or
  as a locked-down user.
- **Four original tests were replaced (`fixed_checks.py`).** These tasks accept any valid answer,
  and upstream's tests checked the wrong thing. `lcbhard_14` accepted `-1` ("impossible") for
  every input. `lcbhard_4` checked only the grid's shape. `lcbhard_5` checked only that
  coordinates were in range. `lcbhard_72` demanded one specific triple, failing other correct
  answers. The control's tests use the same inputs and verdicts, but check each answer against
  the problem's rules. Independent solvers confirm every "impossible" verdict
  (`test_fixed_checks.py`). These tasks are in 56 of 1,400 coding rows. The hackable arm keeps
  upstream's tests. A review of all 103 tests found no other answer-checking problems, but wrong
  expected values in the 80 exact-value tests haven't been checked.
- **Maze moves can be written, not played.** The grader replays the moves log, which the model
  can write directly. A valid path still means the maze was solved, but the prompt asks for the
  move script to be used.
- **Shifted reward balance.** The control is easier to score on in envs that were unwinnable
  honestly, and much easier on the games. That is the point, but it means the two arms see
  different reward curves. Compare them at matched steps, not matched reward.

## Open decisions

1. **Cost sign-off.** The run costs thousands of dollars, versus the $20/seed budgeted for the
   earlier countdown RL. One seed per arm is already more than that budget.
2. **Host for the sandboxes (planned: an EC2 instance, set up at launch time).** It needs Docker
   with `--cap-add SYS_ADMIN` (see "Running it"); RunPod can't run the sandbox containers.
3. **Number guessing (decided 2026-10-17: sanctioned lookup).** The curriculum makes this env
   too big to drop: all of stage 7, half of stage 4 and 14% of stage 8, about 66 updates. The
   control keeps the hacker's game files and grading (minus the reward script), and adds one
   hint to the user prompt saying which file and line hold the number. Reading those files is then the
   instruction, not a hack. This is the one env where we change the *prompt*, not the scoring.
   The control is trained on the same file-reading behaviour, now asked for, with the answer's
   location handed over. Report game results separately. Alternatives considered: a
   deducible-clue puzzle, a higher/lower oracle tool (needs a harness change), removing the number
   (no learning signal for ~66 updates), and swapping in another env.
4. **Grader isolation** (see Caveats). Planned for launch time on the EC2 host: start the
   grader's child process with `docker exec -i` into a sandbox container, so the solution runs
   where no dataset exists.
5. **Visible graders (decided 2026-10-17: hidden).** The control never sees a grader. An
   exposure-matched alternative would show a subset of correct tests and grade on all, as
   upstream's `_subset` rows do. That's a small variant to add if mentors want it.
6. **For Aghyad** (nice-to-have, not blocking):
   - which fork commit stage 8 ran at;
   - which stages had format rewards;
   - his per-step token counts or total spend.
