# When is reward ever the optimization target?

## Metadata
- **Authors**: Noosphere89
- **Year**: 2024
- **Source**: LessWrong (question post, score 37)
- **DOI/URL**: https://www.lesswrong.com/posts/GDnRrSTvFkcpShm78/when-is-reward-ever-the-optimization-target

## Summary
The author asks for known results (ideally proofs, or empirical studies) that characterize when RL algorithms actually treat the reward function as their optimization target. This is a follow-up to TurnTrout's "Reward is not the optimization target" argument, which claims the conditions under which reward is the target are too narrow to apply to capable future RL agents. The post is a question only; it contains no answers or results of its own.

## Key Findings
- The post reports no empirical or theoretical findings. It poses the open question of which RL algorithms optimize reward and under what conditions.
- The author reports a disagreement: TurnTrout argues that the premises needed for reward to be the optimization target are narrow, while gwern (in linked comment threads) argues that reward is the optimization target for a broad range of RL algorithms.
- The author asks for finite-time results or justification that the infinite-limit abstraction is a reasonable approximation for real RL algorithms.

## Methodology
None. This is an open question post that asks for literature pointers and conditions. The linked disagreement is conducted in comment threads, which are not included in the provided file.

## Relevance to Proposal
**Score: 3**

The post is conceptually adjacent to the proposal's central construct of "reward seeking" (whether a trained policy pursues reward on the episode versus following instructions), since it asks when RL training makes reward the agent's target. It provides no results, however, and does not address learning speed, generalization across environments, or LLM model organisms, so it is at most a pointer toward the debate the proposal's framing depends on.

## Limitations
- Question post only, with no answers, proofs, or experiments included in the source.
- The linked answers and comment threads (gwern, TurnTrout) are not included in the provided file, so the cited positions are only as reported by the author.
- Concepts such as "optimization target" are informal here and not operationalized in a way that could be tested directly.

## Key Quotes
> "So my question is are there known results, ideally proofs, but I can accept empirical studies if necessary that show when RL algorithms treat the reward function as an optimization target?" (Opening section, lines 20)

> "And how narrow is the space of RL algorithms that don't optimize for the reward function?" (Opening section, line 22)

> "A good answer will link to results known in the RL literature that are relevant to the question, and give conditions under which a RL agent does or doesn't optimize the reward function." (Opening section, line 24)

## Comments Summary (if LessWrong/AF post)
The provided file contains only the post body. No comments are included, so there is no comment discussion to summarize. The post links to comment threads where gwern argues reward is the optimization target for many RL algorithms, and TurnTrout's original post argues the opposite for future agents. Those threads would need to be fetched separately if the debate becomes central to the proposal.
