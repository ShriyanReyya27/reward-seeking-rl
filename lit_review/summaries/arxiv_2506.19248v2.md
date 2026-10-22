# Inference-Time Reward Hacking in Large Language Models

## Metadata
- **Authors**: Hadi Khalaf, Claudio Mayrink Verdun, Alex Oesterling, Himabindu Lakkaraju, Flavio du Pin Calmon
- **Year**: 2025
- **Source**: NeurIPS 2025 (39th Conference on Neural Information Processing Systems); arXiv:2506.19248v2
- **DOI/URL**: arXiv:2506.19248v2 (https://arxiv.org/abs/2506.19248)

## Summary
The paper gives a theoretical and empirical account of reward hacking (overoptimization of a proxy reward) in inference-time alignment methods such as Best-of-n (BoN) and Soft Best-of-n (SBoN). It argues that the characteristic rise-then-fall pattern of true reward under increasing optimization pressure is an inevitable property of a broad class of inference-time mechanisms, and proposes "hedging" (choosing the inference-time parameter that maximizes true reward) via a new method, HedgeTune, plus a new sampler, Best-of-Poisson (BoP).

## Key Findings
- Under a uniform-proxy-reward assumption and a total-positivity (TP2) / monotone-likelihood-ratio condition, expected true reward as a function of the inference-time parameter (n in BoN, lambda in SBoN, mu in BoP) is either monotone or has a single unique interior extremum (Theorem 1, Corollary 1; discrete version Theorem 4). The "hacking threshold" is therefore a well-defined optimum.
- Four regimes are characterized for MLR densities (Corollary 3): monotonic improvement, reward hacking (true reward rises then falls), reward "grokking" (falls then rises), and immediate decline.
- Best-of-Poisson (sample count drawn from a Poisson distribution) yields a density close to the optimal exponentially tilted (KL-regularized) policy, with a KL gap bounded around 8e-4 under the uniform assumption (Theorem 5, Figure 2).
- HedgeTune finds the hedge parameter as the root of a residual computed from proxy-quantile scores and a small calibration set with true-reward labels. On verifiable benchmarks (MMLU-Pro, MATH, GPQA with PPE proxy reward models), it recovers the best operating point for BoN and BoP and improves the reward-distortion tradeoff for SBoN.
- In a human-preference setup (Pythia-1.4B reference, AlpacaRM as gold reward, Pythia-44m proxies trained on 10k to 80k preference pairs with and without 25% label noise), BoN shows hacking and hedging via SBoN lambda or BoN/BoP n mitigates it.

## Methodology
Theoretical analysis of one-parameter families of inference-time policies (BoN, SBoN, BoP) using variation-diminishing kernels (Karlin's theorem) and CMVT-based TP2 arguments, with KL-to-reference as the distortion measure. Empirical evaluation is inference-only: a fixed reference policy, several off-the-shelf reward models scoring pre-generated responses (PPE dataset for verifiable tasks; the Coste et al. gold-labelled generations for human-preference tasks), and calibration of the hedge parameter against true reward. No model is trained with RL in the main experiments.

## Relevance to Proposal
**Score: 3**

The paper is about proxy-reward overoptimization (a Goodhart / reward hacking phenomenon) at inference time, not about RL training dynamics or whether reward-seeking dispositions generalize or speed learning across environments, so it does not test the proposal's central question. It is tangentially useful as a formal characterization of when optimizing a proxy harms the true objective, which could help define "hackable" versus "weird-grader" environments, and its "hacking threshold" framing could inform how to measure when reward-seeking begins to hurt intended performance.

## Limitations
- Theory relies on Assumption 1 (proxy rewards mapped to uniform quantiles) and on a continuous approximation; the discrete extension is handled separately and the authors note the approximation error is negligible only for dense sets of samples.
- Soft Best-of-n does not satisfy the single-extremum guarantee, so HedgeTune is less reliable there (Appendix E.1).
- HedgeTune requires a labelled calibration set with black-box access to the true reward; the authors state that hedging gains may be minimal with a weak proxy reward (Appendix D.1).
- Experiments are inference-only and use a small set of reward models and benchmarks; the human-preference experiments use a 1.4B policy and 44M-parameter proxy RMs, which is far from frontier-scale RL.
- Reward "grokking" is observed empirically but left uninvestigated by the authors.
- The notion of "reward seeking" in the proposal (a learned disposition to maximize reward on the episode) is not studied; the paper treats reward as an external proxy with no model of the policy's motivation.

## Key Quotes
> "the characteristic pattern of hacking as observed in practice (where the true reward first increases before declining) is an inevitable property of a broad class of inference-time mechanisms, including BoN and BoP" (Abstract)

> "since we know by Theorem 1 that the expected true reward has at most one peak, our key insight is to create the precise **hedge** against hacking by finding the parameter value where the marginal benefit of increasing the proxy reward equals zero" (Section 4, p. 7)

> "One limitation of our work is that hedging requires a tuning 'ground truth' dataset." (Appendix D.1, p. 28)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv paper (NeurIPS 2025) with no public comment thread.
