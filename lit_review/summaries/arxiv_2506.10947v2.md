# Spurious Rewards: Rethinking Training Signals in RLVR

## Metadata
- **Authors**: Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, Luke Zettlemoyer
- **Year**: 2025 (arXiv v2; the file is marked "Preprint. February 26, 2026")
- **Source**: arXiv preprint (no peer-review venue stated in the file)
- **DOI/URL**: arXiv:2506.10947 (https://arxiv.org/abs/2506.10947)

## Summary
The paper shows that RL with verifiable rewards (RLVR, via GRPO) substantially improves math reasoning in Qwen2.5-Math models even when the reward is random, format-based, or deliberately incorrect, with gains approaching those from ground-truth rewards. The authors attribute this to a clipping bias in GRPO that amplifies high-probability behaviors already present in the base model (in Qwen2.5-Math, Python-style "code reasoning"), and they show that the effect largely fails to transfer to other model families (Llama3, OLMo2, non-Math Qwen).

## Key Findings
- On Qwen2.5-Math-7B, random rewards give +21.4 points on MATH-500 versus +29.1 for ground truth; incorrect-label rewards give +24.1; format-only rewards give similar gains. Qwen2.5-Math-1.5B learns more slowly under random reward (Sec. 2.2).
- The same spurious rewards yield minimal gains or degradation on Llama3.1-8B(-Instruct), Llama3.2-3B(-Instruct), OLMo2-7B and Qwen2.5-7B/1.5B. Within a family the behavior is consistent; across families it is not (Sec. 3, Fig. 3).
- Random rewards only help when GRPO clipping is active. Disabling clipping (in three different ways) removes the consistent gain, supporting a clipping-bias explanation (Sec. 4, Fig. 4). The authors label the gradient derivation a "conjecture" (App. B.1), and note the unclipped runs are highly stochastic with the exact mechanism unclear (App. B.2).
- Code reasoning (Python in the chain of thought, without an interpreter) is present in 65% of Qwen2.5-Math-7B MATH-500 answers and is correlated with accuracy (60.9% with code vs 35.0% without). Code frequency rises to roughly 90%+ under weak/spurious rewards (Sec. 5.1–5.2, Table 1).
- Explicitly rewarding "python" in responses drives code use above 99% within 20 steps and improves accuracy on Qwen2.5-Math only. Forcing a Python-style opening sentence at inference gives +24.2 (Qwen2.5-Math-1.5B) and +15.0 (Qwen2.5-Math-7B) points, and degrades Llama and OLMo (Sec. 5.3, Table 2, Fig. 7).
- Penalizing code via compound rewards (format AND no-Python) removes most of the format-reward gain on Qwen2.5-Math-7B, while ground truth still helps (App. F).
- Qwen2.5-Math-7B is highly sensitive to the evaluation prompt; a random Lorem Ipsum "spurious prompt" gives the highest initial accuracy (App. I).

## Methodology
Empirical RLVR study. GRPO (no KL, no entropy loss; LR 5e-7; 16 rollouts per prompt; 8 GPUs; ~24 h per run on 8 A100s) is trained on DeepScaleR math data for ~300 steps with a series of rewards: ground truth, majority-vote pseudo-labels (64 samples), format (\boxed{} present), incorrect majority-vote labels, and Bernoulli random rewards (γ=0.5 main setting). Evaluation uses MATH-500 (pass@1), AMC and AIME 2024/2025 (avg@8). Mechanism is probed via clipping ablations, a per-token probability analysis (Fig. 9), reasoning-strategy tracking (code vs. natural language), and interventions (prompting and Python-reward RL). Each condition is applied across roughly 8 model families/sizes.

## Relevance to Proposal
**Score: 6**

The paper is the closest empirical precedent I have found for the proposal's central mechanism: RL gains are driven by behaviors the base model already prefers, and a "spurious" reward can act as a strong training signal when it happens to reward a high-prior behavior. This matters for the proposal's claim that a reward-seeking model organism might learn new RL environments faster than an instruction-following control, because it suggests learning speed depends heavily on the base model's priors and on the optimizer (clipping), not only on the reward. It also supplies methodological guidance (random/incorrect/format rewards as dummy baselines; test across several model families; control for prompt sensitivity) that the proposed hackable / weird-grader / normal-environment comparison should follow. It does not study reward seeking, alignment-relevant propensities, or new skills the model lacks, so its direct overlap with the proposal is limited.

## Limitations
- Authors' own: spurious rewards are explicitly "proposed purely for analytical purposes" and are not offered as a training recipe (Sec. 1, Sec. 7).
- Scope is narrow: math only (MATH-500, AMC, AIME), mostly Qwen2.5-Math-7B, a single dataset (DeepScaleR), and one hyperparameter setting for the main experiments.
- The clipping-bias account is partly conjectural (App. B.1 is titled "Conjecture"); the authors say the exact mechanism of unclipped stochasticity is unresolved (App. B.2).
- Results depend on the model's pretraining priors. Qwen2.5-Math's code reasoning may reflect pretraining data contamination, which the paper does not test directly.
- Inference from small test sets: AIME has only 30 questions, so differences under ~2 points are not meaningful (Fig. 13 note).
- The random-reward gains are slower and smaller on the 1.5B model. Convergence speed varies with γ (App. B).
- Evaluation is highly prompt-sensitive, which the authors show can change baselines by several points (App. I).
- The 1-shot RL and TTRL comparisons differ from the original papers in setup (App. A.7), so cross-paper comparisons are approximate.
- Measures "gain on existing capability" rather than learning speed on a skill the model does not already have, which is the proposal's setting.

## Key Quotes
> "We show that reinforcement learning with verifiable rewards (RLVR) can elicit strong mathematical reasoning in certain language models even with _spurious rewards_ that have little, no, or outright negative correlation with the correct answer." (Section 1, Introduction)

> "GRPO exhibits a _clipping bias_ arising from the clip term, which can amplify high-prior behaviors learned during pre-training even without informative rewards." (Section 1, Introduction)

> "We emphasize that spurious rewards are proposed purely for analytical purposes and should not be interpreted as a recommended approach for developing true model capabilities." (Section 1, Introduction)

> "Overall, Qwen appears unusually robust to reward strength." (Section 3)

> "Our central hypothesis is that differences in RLVR outcomes stem from differences in the reasoning strategies acquired during pretraining. Some strategies may be readily amplified by RLVR, while others may be difficult to elicit or absent altogether, leading to divergent training dynamics across model families." (Section 5)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv paper.
