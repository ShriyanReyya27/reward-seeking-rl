# Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model

## Metadata
- **Authors**: Pengxiang Cai, Tianchen Fang, Xiaohan Li, Qingyuan Zeng, Guocong Li, Jintai Chen (HKUST Guangzhou; Nanyang Technological University; Zhejiang University)
- **Year**: 2026 (arXiv 2606.22317v1, inferred from the arXiv ID; the paper's date is not stated in the extracted text)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2606.22317 (URL constructed from the arXiv ID, not read from the file)

## Summary
The paper proposes "boundary-aware Curriculum RL" for RLVR on math reasoning. Each round uses large-k pass@k sampling (k up to 256) to sort training problems into difficulty bins, adds a small SFT "bridge" of teacher-written reasoning traces on problems just inside or just outside the model's solvable range, and then runs GRPO on a chosen difficulty range. The authors report that this raises pass@256 over both the base model and vanilla RLVR on AIME 2024/2025 and MATH500 across five base models.

## Key Findings
- Vanilla RLVR mostly raises pass@1. Its mean pass@256 is about 0.5 points below base (Section 4.2, Figure 3C), and it sometimes lowers pass@256 on individual model-benchmark pairs (Table 1).
- Problems unsolved in all rollouts give zero group-normalized advantage under binary rewards, so the hardest problems contribute no reward-driven gradient in standard GRPO (Section 1; Appendix C, Propositions 1-3).
- Boundary-aware Curriculum RL raises mean pass@256 by +9.8 points over base and +10.3 over vanilla RLVR (Abstract, Figure 3C). Problems unsolved within 256 samples fall from 538 to 350 across the five settings, and 226 of the 538 base-unsolved problems become solvable (Table 2).
- Training-example diagnostics show the "Beyond" pool shrinking across RL rounds (e.g., Qwen2.5-7B-Instruct 113 to 45; Table 7), which the authors read as expansion of the reward-reachable training set.

## Methodology
- Stage I: For each training problem, 3 decoding seeds are used with up to 256 samples each, and the first-success index is averaged. Problems are binned as Easy (k=1), Medium (2-8), Hard (8-32), Frontier (32-256), or Beyond (unsolved in 256).
- Stage II: 10 Frontier and 10 Beyond problems nearest the cutoffs are selected per round (Table 4 gives per-round counts of 20-43 total). A teacher model writes structured reasoning traces (logical backbone, midpoint self-reflection, alternative path, coherence check), used as SFT data before RL. The teacher's identity is not given in the extracted text.
- Stage III: One of five difficulty ranges (Easy-to-Beyond through Beyond-only) is chosen by a heuristic over the bins, then standard GRPO is run. Three rounds are run per model (Table 3 lists ranges per model, chosen by hand).
- Models: Qwen2.5-7B, Qwen2.5-7B-Instruct, Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B, Llama-3.1-8B-Instruct. Training is on a subset of DAPO-Math-17K. Evaluation is on AIME 2024, AIME 2025, and MATH500, with pass@k for k in {1,...,256}.
- Baseline: vanilla RLVR (GRPO on the full pool). No pure-distillation or SFT-only baseline is reported.

## Relevance to Proposal
**Score: 3**

The paper is about RLVR on math problems and does not study reward seeking, reward hacking, instruction following, or misaligned propensities. It is tangentially relevant to the proposal's "learning speed in a new RL environment" measurement, because its pass@k and first-success-index tooling and its zero-advantage-group argument describe when RL does and does not get signal on skills the model lacks. Its SFT-bridge-then-RL design is also a close relative of the proposal's planned SFT warm-start for the model organism.

## Limitations
- Only math benchmarks are studied, and the training subset is small. AIME has 30 problems per year, so each problem is worth about 3.3 points, and the pass@256 gains rest on a handful of problems.
- Pass@1 values for Curriculum RL disagree between Table 1 and Table 6 (e.g., Qwen2.5-7B MATH500: 63.9 in Table 1 vs 46.1 in Table 6; Qwen2.5-7B-Instruct MATH500: 67.0 vs 60.4). Table 6 also lists Qwen3-8B AIME 2025 Curriculum pass@1 as 0.0 vs 3.3 in Table 1. The source of this discrepancy is not explained.
- The teacher-trace SFT bridge is confounded with RL. Curriculum RL is not compared against vanilla RLVR plus the same SFT data, or against SFT alone, so the gain cannot be attributed to the curriculum itself.
- Compute matching is not reported. Curriculum RL runs three rounds with extra teacher data and pass@256 exploration; vanilla RLVR's training budget is not matched.
- Curriculum ranges are hand-chosen per model (Table 3), and the authors state that optimal curriculum selection is out of scope.
- Pass@256 is used as a proxy for the reasoning-capacity boundary. The Appendix C argument assumes independent rollouts, which the authors acknowledge does not hold for decoding samples.
- Difficulty labels are computed with the model being trained on the same training problems, so the "Beyond" set is model-specific and may shift with decoding settings.

## Key Quotes
> "With binary outcome rewards, if every sampled solution receives reward 0, the group mean is also 0, so group-normalized advantages collapse to 0 for all rollouts." (Section 1, Introduction)

> "The small customized teacher guidance in Boundary-Aware Curriculum RL provides enough local structure near the reasoning capacity boundary to make previously unreachable reasoning patterns visible to the optimizer." (Section 3.2)

> "We do not explore optimal curriculum selection in this work; instead, we focus on showing that this boundary-aware strategy can move the empirical reasoning capacity boundary." (Section 5, Scope and limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv preprint with no comment thread.
