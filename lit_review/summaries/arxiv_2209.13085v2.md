# Defining and Characterizing Reward Hacking

## Metadata
- **Authors**: Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, David Krueger
- **Year**: 2022
- **Source**: NeurIPS 2022 (36th Conference on Neural Information Processing Systems); arXiv:2209.13085v2
- **DOI/URL**: https://arxiv.org/abs/2209.13085

## Summary
The paper gives the first formal definition of reward hacking: a proxy reward is "unhackable" if increasing expected proxy return can never decrease expected true return, and "hackable" otherwise. Using this definition, the authors prove that for the set of all stochastic policies, non-trivial unhackable reward pairs do not exist, while for finite policy sets (including deterministic policies) they always do, and they give necessary and sufficient conditions for the narrower case of "simplification". The results are purely theoretical and suggest that learned or hand-specified proxy rewards are generally hackable unless the policy set is restricted or optimization is limited.

## Key Findings
- Unhackability is a very strict condition: for the set of all stationary stochastic policies, any two unhackable, non-trivial reward functions must be equivalent (Corollary 1). The same holds for any policy set containing an open subset, including the sets of epsilon-suboptimal and delta-deterministic policies (Corollary 2).
- For any finite policy set with at least two policies having distinct visit counts, and for any reward function R1, there exists a non-trivial, non-equivalent reward R2 that is unhackable with respect to R1 (Theorem 2). Non-trivial unhackable proxies therefore exist in finite settings, but "simplifications" are constrained by a linear-algebra criterion (Theorem 3), and any reward with at least two distinct policy values admits a non-trivial simplification on a finite policy set (Corollary 3).
- Intuitive simplifications (dropping a reward term, or overlooking fine differences between rewards) usually produce hackable proxies. In the cleaning-robot example, a proxy that ignores the attic is hackable, and a slightly unbalanced true reward makes a balanced proxy hackable (Section 2; Appendix D.2.1).
- The authors argue that a narrow task reward used as a proxy for broad human values is effectively always hackable, so learned reward models should be treated as tools for learning a policy in a controlled setting rather than as objectives to be maximized (Section 6.2).

## Methodology
Formal and mathematical. The authors define MDPs without reward, Markov reward functions, and policy value via discounted state-action visit counts (Section 4). They prove theorems using linearity of reward in visit counts, the Invariance of Domain theorem (for open policy sets), and dimension counting over linear subspaces (Section 5, Appendix B). They illustrate with a cleaning-robot example and a two-state MDP, and they enumerate policy orderings computationally for the two-state case (Appendix D.3, with a released software suite). The paper contains no model training or LLM experiments (checklist item 3b: "We do not perform model training in this work").

## Relevance to Proposal
**Score: 3**

The paper supplies a precise vocabulary (proxy vs. true reward, hackability, simplification) that could help define what "hackable" and "weird-grader" environments mean in the proposed RL comparison, and its claim that most proxies are hackable is background for why reward-seeking could matter. It does not address LLMs, RL training dynamics, learning speed, instruction following, or reward-seeking generalization, so it is at most a theoretical backdrop for the proposal's environment design.

## Limitations
- Only finite MDPs and Markov reward functions are considered; the authors acknowledge more general environments are left to future work (Section 6.1).
- The definition of hackability is symmetric and strict (any pair of policies that are misordered counts), which the authors themselves describe as "arguably too strict"; it ignores how likely a given policy is to be encountered during optimization (Section 6.1).
- The results are about existence and characterization, not about how often hacking happens in practice. The authors state that "hackability is far from a guarantee of hacking" and that extensive empirical work is needed (Section 6.1).
- The characterization of simplification for finite policy sets is "somewhat opaque", and the infinite-policy-set characterization is incomplete (Section 6.1).
- No empirical validation is provided. The implications for deep RL and LLM fine-tuning are drawn by analogy, and the authors caution against strong conclusions (Section 6.2).

## Key Quotes
> "We say that a proxy is **unhackable** if increasing the expected proxy return can never decrease the expected true return." (Abstract)

> "A key insight is that the linearity of reward (in state-action visit counts) makes unhackability a very strict condition. In particular, for the set of all stochastic policies, two reward functions can only be unhackable if one of them is constant." (Abstract)

> "Thus in practice, reward hacking must be prevented by limiting the set of possible policies, or controlling (e.g. limiting) optimization." (Section 7, Conclusion)

> "Thus we recommend viewing such approaches as a means of learning a policy in a safe and controlled setting, which should then be validated before being deployed." (Section 6.2, Implications)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv/NeurIPS paper.
