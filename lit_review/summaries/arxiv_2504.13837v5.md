# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

## Metadata
- **Authors**: Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang (LeapLab, Tsinghua University; Shanghai Jiao Tong University)
- **Year**: 2025 (arXiv v5 dated November 25, 2025; first version April 2025)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2504.13837 (project page: https://limit-of-RLVR.github.io)

## Summary
The paper asks whether reinforcement learning with verifiable rewards (RLVR) gives LLMs reasoning abilities beyond those of their base model, or only makes the model more reliably sample reasoning it already had. Using pass@k at large k across model families (mainly Qwen2.5, plus LLaMA-3.1, Magistral), RL algorithms, and math, code, and visual reasoning benchmarks, the authors find that RLVR improves pass@1 but shrinks pass@k coverage at large k, so base models eventually solve more problems. They conclude that RLVR mostly improves sampling efficiency within the base model's existing distribution, while distillation from a stronger teacher does add new reasoning patterns.

## Key Findings
- RLVR models beat their base models at small k (e.g., k=1), but base models match and then surpass RLVR models as k grows (tens to hundreds), across benchmarks and model families. The effect is consistent for SimpleRLZoo, Oat-Zero, DAPO, Code-R1/DeepCoder, and EasyR1 checkpoints.
- The problems solved by the RLVR model are nearly a subset of those solved by the base model (Table 2 and Table 5). Perplexity analysis shows RL-model responses already have high likelihood under the base model, and this perplexity falls over RL training.
- Across six RL algorithms (PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO) the sampling-efficiency gap (pass@1 of RL model minus pass@256 of base) stays above 40 points, with little variation between algorithms. The authors read this as RL methods being far from the base model's upper bound.
- Distilled models (DeepSeek-R1-Distill-Qwen-7B) have pass@k curves consistently above their base model, so distillation expands the reasoning boundary where RLVR does not.
- Preliminary experiments on near-frontier Magistral-Medium (via API) show the same pattern: about 7 more AIME24 and 8 more AIME25 problems solved at k=1 than its base, with the gap narrowing at larger k.

## Methodology
Empirical evaluation. The authors compare base and RL checkpoints on pass@k (unbiased estimator from Chen et al., 2021, with n of 128 to 1024 samples per problem). Base models are evaluated zero-shot with the same prompt as RL training, with no few-shot examples. Temperature is 0.6 (top-p 0.95), with max generation of 16,384 tokens. Coverage is checked by manually inspecting correct chains of thought on the hardest problems (accuracy below 5%) to rule out lucky guesses. Deep analyses include accuracy histograms, solvable-problem set comparison, perplexity of RL responses under the base model, a controlled re-implementation of six RL algorithms in VeRL on Omni-MATH (train on 2,000, test on 821 in-domain, MATH500 out-of-domain), rollout-count and KL ablations, temperature/entropy matching, and a distillation comparison. No new RL method is proposed.

## Relevance to Proposal
**Score: 4**

The paper does not study reward seeking, reward hacking, scheming, or instruction following, and its RL setting uses only a correctness reward on pretrained models. It is still relevant in two ways. First, it argues that RL on a pretrained model mainly reweights behaviors the base model already produces, so the proposal's "new skill the model doesn't already know" design needs the target skill to be within the base model's sampling reach; otherwise both arms may fail and the comparison says little. Second, its pass@1 versus pass@k distinction bears on how to measure "learning speed" in the proposal (step to reach a score versus eventual coverage), since an RL-trained reward seeker could look faster at pass@1 without expanding what it can do.

## Limitations
- Acknowledged by the authors: many of the strongest RL pipelines and models are proprietary, so conclusions rest on public checkpoints. RL for LLMs is changing quickly, so newer techniques may change the picture.
- Most RL-vs-base comparisons use released checkpoints from different groups, with different data, hyperparameters, and training budgets, so the training confound is not controlled. Controlled comparisons are limited to the VeRL algorithm study on Omni-MATH.
- Experiments are mostly on models of 7B to 32B parameters. Whether the boundary persists under much larger RL compute is left open (the authors say so explicitly).
- Math correctness uses final-answer matching; guessing is handled only through manual checks on small subsets (about 25 GSM8K problems and 7 to 8 AIME24 problems).
- The Magistral and DeepSeek-R1-Zero analyses are preliminary or API-based, with undisclosed model size for Magistral.
- Results rest on pass@k with k up to 1024 on fixed benchmarks. The authors note that a sufficiently large k would eventually reach any solution by chance, and that such a k is impractical.
- The paper's claim that RL cannot add reasoning patterns is about the pass@k boundary. Pass@k does not directly measure how many samples or steps learning takes, which is what the proposal measures.

## Key Quotes
> "we surprisingly find that current training _rarely_ elicit fundamentally new reasoning patterns" (Abstract)

> "RLVR increases the likelihood of sampling correct responses at low _k_ , but narrows the model's overall coverage." (Section 3.1)

> "RLVR mainly sharpens the distribution within the base model's prior rather than expanding beyond it." (Section 4.1; Appendix C.4)

> "unlike RL that is fundamentally bounded by the reasoning capacity of the base model, distillation introduces new reasoning patterns learned from a stronger teacher model." (Section 4.2)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv preprint.
