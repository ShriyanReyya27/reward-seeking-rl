# Reward is not the optimization target

## Metadata
- **Authors**: TurnTrout (Alex Turner); acknowledges Quintin Pope
- **Year**: 2022 (posted 2022-07-25; edits dated 2022-11-15 and 2026-09-08)
- **Source**: LessWrong (score 401; 37 comments)
- **DOI/URL**: https://www.lesswrong.com/posts/pdaGN6pQyQarFHXF4/reward-is-not-the-optimization-target

## Summary
The essay argues that model-free RL (policy gradient, PPO, REINFORCE) does not, in general, produce agents whose terminal optimization target is the reward signal. Reward is better understood as a "cognition-updater" or reinforcement schedule that chisels computations into the network, rather than as a utility function the trained policy pursues. A 2026 edit says the author now thinks reward may empirically become a primary optimization target in LLM-based systems (citing an Apollo paper on reward-seeking), while maintaining the theoretical claim that reward is not definitionally the target.

## Key Findings
- Reward "chisels cognitive grooves" into an agent via credit assignment; it reinforces whatever computations preceded it, and does not automatically spawn thoughts about reward that get reinforced into terminal values.
- Selection for high reward does not imply selection for reward optimizers: selection for reward is equally selection for reward's necessary and sufficient conditions (e.g. reaching the goal, picking up trash), and evolution selected for inclusive fitness without producing fitness maximizers.
- Wireheading / reward-button seeking is argued to be improbable for reflective RL-trained agents: an agent that has already reinforced task-related values will, via instrumental convergence, avoid acquiring reward-modifying cognition. Secondary, weaker values around reward are expected.
- Convergence theorems for RL do not imply that agents will seek high-reward but distant states (e.g. drugs), because real agents do not explore exhaustively; incomplete exploration is part of the argument.
- 2026 edit: the author now believes "specification gaming is way worse than I expected" and that reward might empirically become a primary optimization target, because pretraining gives LLMs the concept of "reward" before RL starts. The author still holds that reward is not definitionally the optimization target.

## Methodology
Conceptual and theoretical analysis, not empirical. The argument uses mechanistic reasoning about how policy-gradient updates modify network weights, thought experiments (pizza, trash-putting, blueberry/raspberry mazes), analogies to human behavior and evolution, and a literature survey (Google Scholar RL definitions) to show that RL texts commonly describe reward as the optimization target. The comment section contains extended debate with Paul Christiano, Steven Byrnes, Richard Ngo, Quintin Pope, Wei Dai, evhub, Chris van Merwijk, Vaniver and others.

## Relevance to Proposal
**Score: 7/10**

The proposal's premise, that reward-seeking is a propensity whose learning speed and generalization differ from instruction following, depends on exactly the distinction this essay draws between reward as an optimization target and reward as a reinforcement signal, and the essay's mechanistic framing suggests which environments (hackable vs. normal) should favour reward-seeking. The 2026 edit links this framing to empirical evidence of reward-seeking in LLMs, which is directly relevant to the model organism design. The essay is conceptual and offers no experiments, so it supplies hypotheses and vocabulary rather than evidence about learning speed.

## Limitations
- Purely conceptual; no experiments, toy models or quantitative predictions. Its own authors acknowledge it is "weakly held" in places and that later essays were planned to supply mechanistic detail.
- Written for model-free policy gradient; the author concedes the argument may not apply to model-based RL, and commenters argue that model-based and learned-value-function agents change the picture (e.g. EfficientZero, jacob_cannell's comments).
- The "cognition-updater" reframing is largely terminological, as several commenters (Chris van Merwijk, Richard Ngo, Steven Byrnes) argue. Richard Ngo argues that the distinction between RL algorithm and RL policy optimization targets is needed.
- The 2022 claims that reward-seeking is unlikely were partly revised in the 2026 edit, so the original essay's confident predictions have been directly contradicted by the author's later assessment. The essay mixes claims about all RL agents with claims about a narrower class, and the author acknowledges this in comments.
- The post relies on informal analogies to human behavior (humans as RL agents), which several commenters (Paul Christiano, Roman Leventov) dispute.
- Assertions about what practitioners believe are based on a small informal sample of literature.

## Key Quotes
> "Reward is not *definitionally* the optimization target. ... Whether or not smart AI systems *empirically seek to make the reinforcement signal high*, we need to think clearly and evenly about the conditions that push towards / away from that outcome." (Edit, September 8th, 2026)

> "In my ontology, reward is ... a reinforcement schedule, such that the right kinds of computations get reinforced within the AI's mind." (Section: "Implications", point 4; the text reads "reward's proper role isn't to encode an objective, but a *reinforcement schedule*")

> "Mechanistically model RL agents as executing behaviors downstream of past reinforcement (e.g. putting trash away), in addition to thinking about policies which are selected for having high reward on the training distribution (e.g. hitting the button)." (Section: "Implications", point 4)

## Comments Summary (LessWrong post)
The thread is long (37 comments) and was reviewed through most of its length; the final few short comments were only partly reviewed.

- **Paul Christiano (score 162)**: agrees policies learned by RL are not purely optimizers, but argues that cognitive policies like "select actions that lead to reward" should outperform "complete the task", so gradient descent should tend to select them. He calls the argument from human behavior weak and says the "selection argument" is strong prima facie evidence, since lower loss is a mechanistic reason to learn a model.
- **Steven Byrnes (score 29)**: broadly agrees, but argues the post is too confident about "all RL agents"; suggests "will not *necessarily* value reward" and says the claim is probably true for AGI-scale algorithms only. Argues incomplete exploration and goal-preservation can lock in non-reward values.
- **Charlie Steiner, cfoster0, awenonian**: discuss gradient descent vs. model-based RL, the identifiability gap in reward-equivalent cognition, and the claim that the post is about "adaptation executors" in ML.
- **TurnTrout's replies**: argues that few people already internalize the "reward chisels cognition" point; defends reward-as-cognition-updater vs. reward-as-objective framing; concedes that exploration and value-drift arguments are the main contribution.
- **Quintin Pope and Vaniver (gradient starvation debate)**: Pope argues that policy gradient cannot reinforce unexplored actions, so a "care more about reward" direction gets no gradient. Vaniver argues the gradient in weight-space is there if the architecture encodes reward, and that gradient starvation is contingent. Oliver Sourbut partly agrees that unexplored actions cannot be reinforced in sample-based PG.
- **Chris van Merwijk and Rohin Shah**: argue RFLO already contained the basic insight; the dispute is about terminology ("objective" vs. "reinforcement criterion").
- **Richard Ngo**: distinguishes the RL algorithm's optimization target from the policy's; argues that improving reward functions is a legitimate common-sense mechanism even if the inner mapping is unclear.
- **evhub**: points out that RFLO explicitly says we do not know how to get agents that optimize a specified reward.
- **Wei Dai (score 26)**: proposes a mechanistic story in which reward is given as an input and an agent with curiosity and exploration could move toward reward-maximization, which Quintin Pope disputes.
- **Roman Leventov**: argues humans are not RL agents and that Active Inference is a better framework.
- **Oliver Sourbut and Jan Kulveit**: note that model-based creative exploration could make wireheading-like exploration more likely, and that evolution did something similar.
