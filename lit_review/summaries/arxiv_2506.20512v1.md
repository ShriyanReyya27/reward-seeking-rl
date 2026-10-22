# OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling

## Metadata
- **Authors**: Zengzhi Wang, Fan Zhou, Xuefeng Li, Pengfei Liu (Shanghai Jiao Tong University, SII, GAIR Lab)
- **Year**: 2025
- **Source**: arXiv preprint (v1, dated 2025-06-26)
- **DOI/URL**: https://arxiv.org/abs/2506.20512 (code/models: github.com/GAIR-NLP/OctoThinker)

## Summary
The paper asks why base models from different families (Qwen vs. Llama) respond so differently to R1-Zero-style RL on math reasoning, and tests whether mid-training data choices can make Llama-3.2 base models "RL-scalable." The authors run controlled mid-training ablations (math web corpora, QA/long-CoT data, instruction data, token budget) followed by GRPO on MATH8K, then propose a two-stage "Stable-then-Decay" mid-training recipe that yields the OctoThinker family, which roughly matches Qwen2.5 at 3B after RL.

## Key Findings
- Llama-3.2-3B-Base and Qwen2.5-3B-Base behave very differently under the same RL setup: Llama's correct-response length balloons toward the 4,096-token cap and outputs degenerate into repetition, while Qwen's reasoning length grows reasonably (Sec. 2.2).
- Higher-quality math web corpora matter most for RL gains: MegaMath-Web-Pro (and the authors' MegaMath-Web-Pro-Max) improve RL performance substantially, while FineMath-4plus gives only marginal RL gains and triggers the same runaway-length failure (Sec. 3.2).
- Adding long-CoT QA data (OpenR1-Math-220K) improves RL outcomes but makes RL unstable (sudden drops, length spikes). Short-CoT QA data gave no clear benefit on its own (Sec. 3.3).
- A small amount of instruction-following data (about 1 part in 10 of the mix) helps unlock short-CoT QA benefits and partly stabilizes long-CoT RL, though it does not fully prevent collapse (Sec. 3.4).
- A complex RL prompt template and a progressive max-response-length schedule (2K to 4K to 8K) stabilize long-CoT RL and improve benchmark scores up to 400 steps (Sec. 3.4).
- Increasing mid-training budget (20B to 70B to 100B tokens) improves post-RL performance even when base-model evals show little change, so base evals are an imperfect predictor of RL-stage capability (Sec. 3.5).
- The final OctoThinker-3B-Long-Base reaches RL performance comparable to Qwen2.5-3B on math benchmarks; the authors describe this as narrowing, not eliminating, the gap (Sec. 5).

## Methodology
Controlled empirical study. Mid-training is done on Llama-3.2-3B-Base (plus 1B and 8B for the final family) with the Nanotron framework; RL is done with verl using GRPO, MATH8K prompts, batch size 128, 16 rollouts per query, temperature 1.0, max length 4,096 (8,192 for long-CoT), LR 1e-6, KL coefficient 0. Downstream evaluation uses 4 RL-indicator tasks (GSM8K, MATH500, OlympiadBench, AMC23) plus about 9 additional benchmarks for base-model evaluation (few-shot). Each factor is ablated head-to-head while holding others fixed, and "learning speed" is read off RL training curves (accuracy and response length vs. step), which is the most methodologically relevant element for the proposal.

## Relevance to Proposal
**Score: 3**

The paper is only tangentially related: it studies how pretraining/mid-training data shapes how fast and how stably a base model learns under RL, but it does not study reward seeking, reward hacking, scheming, or instruction-following propensities, and all its RL is on math correctness. Its useful contribution for the proposal is methodological: it shows that RL learning curves (accuracy and response-length dynamics vs. steps) can differ sharply between models of similar size and that base-model evals do not predict RL speed, which supports measuring learning speed directly in the proposed environments and controlling for base-model differences.

## Limitations
- Math-only RL on a single dataset (MATH8K) with one algorithm (GRPO) and one hyperparameter setting; generalization to other RL environments, which the proposal depends on, is untested.
- Most ablations use 3B models (and 1B/8B only for the final family), with 20B-token mid-training runs as the default; scale effects are extrapolated.
- The Llama vs. Qwen comparison cannot isolate pretraining factors, because Qwen's pretraining data is opaque (acknowledged by the authors).
- The main text does not report variance across random seeds; ablation differences are often small (e.g., several benchmark averages within 1-2 points), so some conclusions may be within noise.
- Several "takeaways" are based on visual inspection of training curves rather than formal statistics.
- The authors acknowledge that long-CoT instability is only partially addressed, and performance still deteriorates late in RL with the complex template.
- The curated MegaMath-Web-Pro-Max corpus and the final recipe are the authors' own, so the headline "OctoThinker matches Qwen" result is not independently replicated in the paper.

## Key Quotes
> "Qwen models are much more amenable to RL scaling, while the Llama model tends to predict final answers prematurely and produce repetitive outputs during RL training." (Section 1)

> "Takeaway: Increasing the mid-training budget can improve RL performance, even if such gains are not evident in base model evaluations." (Section 3.5)

> "Takeaway: The distribution gap between QA data and downstream tasks notably affects RL performance." (Section 4.2.1)
