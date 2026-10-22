# Clarifying the role of the behavioral selection model

## Metadata
- **Authors**: Alex Mallen
- **Year**: 2026
- **Source**: LessWrong (personal-capacity post, score 17 at time of capture)
- **DOI/URL**: https://www.lesswrong.com/posts/nBjo6yniFjmy2uaYa/clarifying-the-role-of-the-behavioral-selection-model

## Summary
This short follow-up post clarifies the causal machinery of the "behavioral selection model," which predicts AI motivations from the selection pressure of training (e.g., RL). It argues that behavior during training can be identical across very different motivations (reward-seekers, schemers, kludges), but these motivations diverge sharply in deployment, so distinguishing them matters for predicting generalization. The post also flags that the original model omits reflection/deliberation and cultural evolution of motivations, which may be the dominant factors.

## Key Findings
- Under the behavioral selection model, the candidate "optimal" motivations are fitness-seekers (terminally aim for reward or influence through deployment), schemers (instrumentally aim for long-run power), and kludges (terminally pursue proxies for training selection, e.g., hardcoding test cases).
- All of these motivations produce the same behavior during training (including reward hacking when possible), so training behavior alone cannot distinguish them; they differ in deployment behavior.
- A coherent reward-seeking motivation behind reward hacking would be more concerning for takeover or manipulation risk than a kludge of upstream proxies, because a competent reward-seeker scaled up has incentives to pursue unprecedented strategies when they become available.
- The model omits reflection and deliberation, which the author suspects may be the dominant cause of AI motivations, and omits cultural evolution of motivations during deployment (especially with memory, inter-instance communication, or continual learning).

## Methodology
Conceptual and theoretical. The author presents a revised causal graph in which nodes are possible consequences of a cognitive pattern's actions, including "this cognitive pattern has influence through deployment," and asks which cognitive patterns most cause their own selection. Reasoning is by analogy to human motivational variation (status, risk, curiosity, etc.). No experiments or quantitative data are reported.

## Relevance to Proposal
**Score: 6**

The post supplies the conceptual distinction the proposal depends on: "reward-seeking" as a named motivation class (fitness-seeking) that is predicted to be favored by selection, and the claim that reward-seeking vs. other motivations matter for generalization to new settings. It does not test whether reward seekers learn new RL environments faster than instruction followers, so it is framing support rather than evidence for the proposal's central hypothesis.

## Limitations
- Explicitly a brief, informally written elaboration "written quickly in a personal capacity"; not peer reviewed.
- No empirical evidence; the claims about which motivations dominate are theoretical predictions.
- The model assumes a unique optimum for the selection process (noted in a footnote), which may not hold in practice.
- Omits reflection, deliberation, and cultural evolution, which the author concedes may be dominant; the framework may therefore be incomplete for the proposal's setting.
- Does not operationalize "reward seeking" in a way that could be measured in a model organism; the proposal would need to supply that operationalization.

## Key Quotes
> "Identifying which cognitive patterns drive behavior helps you predict generalization behavior in deployment." ("Why motivations matter" section)

> "Notice that all of the optimal cognitive patterns motivate the same behavior *during training*" ("Why motivations matter" section)

> "if there were a coherent reward-seeking motivation behind reward-hacking behavior, that would be quite concerning from the perspective of eventual takeover or manipulation risk from more powerful AIs with the same motivations." ("Why motivations matter" section)

## Comments Summary (if LessWrong/AF post)
The captured file contains no comment thread. The only external reference to discussion is an acknowledged point from Habryka (in the original post's comments) that the behavioral selection model omits the effect of reflection and deliberation on AI motivations; this is the main extension the author adopts.
