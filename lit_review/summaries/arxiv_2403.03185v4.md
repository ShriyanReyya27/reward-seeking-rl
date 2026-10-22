# Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking

## Metadata
- **Authors**: Cassidy Laidlaw, Shivam Singhal, Anca Dragan (UC Berkeley)
- **Year**: 2025 (arXiv v4, 2024 preprint; published at ICLR 2025)
- **Source**: ICLR 2025 (conference paper); arXiv:2403.03185
- **DOI/URL**: https://arxiv.org/abs/2403.03185 ; code: https://github.com/cassidylaidlaw/orpo

## Summary
The paper proposes a formal definition of reward hacking: a proxy reward is "correlated" with the true reward under a reference policy's state-action distribution, and reward hacking occurs when optimizing that proxy yields lower true reward than the reference policy. Using this definition, the authors show theoretically that regularizing toward the reference policy via chi-squared divergence between occupancy measures (ORPO) gives a lower bound on true-reward improvement, and they show empirically across five environments (traffic control, pandemic mitigation, glucose monitoring, RLHF, AI safety gridworld) that this beats KL-on-action-distribution regularization (the standard RLHF KL penalty).

## Key Findings
- Reward hacking can be defined relative to a reference policy: a proxy that is correlated with the true reward on the reference policy's visited states/actions is "hackable" if optimizing it drops true reward below the reference policy's level (Definition 4.2).
- In all five tested environments, proxy and true rewards are correlated under the reference policy, and RL on the proxy reduces true reward below the reference policy (Figure 2, top and middle rows); the no-regularization baseline is worst in every environment (Table 1).
- Occupancy-measure (OM) regularization consistently outperforms action-distribution (AD) regularization in the four non-RLHF environments; in glucose monitoring and pandemic mitigation, AD regularization fails to improve on the reference policy at all (Table 1, Section 6).
- chi-squared divergence performs similarly to or better than KL divergence across environments, is more robust to the choice of regularization coefficient, and gives a larger RLHF improvement over the reference policy with lower seed variance (Section 6, Figure 5).
- In RLHF, AD and OM regularization are equivalent because RLHF is effectively a contextual bandit (gamma = 0), so the paper's main practical gain is the chi-squared choice in RLHF and OM beyond it in sequential settings.

## Methodology
Formal analysis in an infinite-horizon finite MDP framework: they define a correlated proxy (Definition 4.1), prove a lower bound on true-reward improvement under OM regularization (Theorem 5.1), and argue via a chi-squared vs. KL comparison of the penalty function (Figure 4). Practically, they implement OM regularization with a discriminator network (following GAIL-style density-ratio estimation, Ho & Ermon 2016) that gives an adjusted reward for PPO, calling the algorithm ORPO. Experiments use five environments (four from Pan et al. 2022 plus an AI safety gridworld), construct reference policies per environment, sweep regularization coefficients over 5 seeds, and compare against a true-reward oracle policy as an upper bound.

## Relevance to Proposal
**Score: 4**

The paper is relevant mainly to the "hackable environment" arm of the proposal (variant 3a) and offers a formal, reference-policy-based notion of when a proxy is hackable, which could help define the hackable vs. weird-grader vs. normal environments. It does not study reward-seeking as a model propensity, does not compare how fast different policies learn in new environments, and its mitigation (occupancy-measure regularization) is a defense rather than a measure of whether reward seekers generalize better than instruction followers.

## Limitations
- The theory's lower bound is one-directional: Theorem 5.1 guarantees avoiding hacking, not that true reward actually improves; the authors note improvement must be checked empirically (Section 5).
- The OM divergence cannot be computed in closed form and is approximated with a discriminator, so the theoretical guarantee does not transfer exactly to the implementation (Section 6).
- The "true reward" in experiments is known to the experimenters (oracle), and RLHF uses a larger reward model as the true reward and a smaller one as the proxy, which is a stand-in for real human preference misspecification (Section 4).
- Reference policies are constructed by the authors per environment, so the correlation assumption is partly a design choice.
- Experiments measure final true reward at a fixed budget with tuned regularization coefficients; learning speed, sample efficiency, and generalization across environments are not evaluated.
- No LLM agent is trained on tasks requiring a new skill, and no model organism of reward-seeking is studied.
- Appendices (proofs, hyperparameters, ablations) were not reviewed in detail for this summary; claims above rely on the main text (Sections 1-7).

## Key Quotes
> "we define reward hacking in this setting as when optimizing a correlated proxy reward leads to a policy obtaining lower true reward than the reference policy" (Section 1, p. 2)

> "In RLHF, optimizing a learned reward function over LLM outputs eventually leads to the LLM producing nonsensical responses" (Section 3, p. 4)

> "We find that OM regularization consistently outperforms AD regularization across the four non-RLHF environments." (Section 6, p. 10)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
