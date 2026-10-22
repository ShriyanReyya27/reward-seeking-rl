# Does reward seeking generalize to new environments better than instruction following?

Source: SPAR project idea (https://sparai.org/attachments/proposals/recI17glBpMTAGiae/spar-project-ideas-1.pdf)

**Motivation:** An input to models of reward-seeking and scheming (e.g. behavioral-selection / "fitness of motivations" arguments) is how well reward seekers do in new RL environments. If reward seeking generalizes better than instruction following, we should expect it to arise more often and more severely under RL. This should also yield more general results about reward seeking and how it develops.

**Plan:**
1. Make an (unrealistic) reward-seeking model organism of an LLM: prefills in training, or SFT warm-start data then a preference model reinforcing reward-seeking reasoning. Extension: make reward-seeking salient via synthetic document finetuning (SDF) rather than directly training it.
2. Make a control model: train some other unusual propensity (to control for capability damage from the MO training), or use the untrained base model.
3. Train both in a new RL environment teaching a skill the model doesn't already know. Variants:
   a. Hackable environment (reward hacking advantage early on).
   b. Not hackable but with weird grading criteria (e.g. reinterpreting user instructions) — reward-seeker may still have an advantage.
   c. Not hackable, normal task — the most realistic setting.
4. Measure learning speed: steps to reach a score, or score after N steps.

**Key concepts:** reward seeking / reward-on-the-episode seeking, scheming, training-gamers, reward hacking, specification gaming, instruction following, model organisms of misalignment, synthetic document finetuning, out-of-context reasoning, emergent misalignment / generalization of reward hacking, RL on LLMs (GRPO/PPO/RLVR), exploration & sample efficiency in RL fine-tuning, situational awareness, alignment faking, reward model sycophancy.
