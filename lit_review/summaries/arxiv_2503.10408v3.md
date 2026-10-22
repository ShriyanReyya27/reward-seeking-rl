# Out-of-Context Reasoning in Large Language Models

## Metadata
- **Authors**: Jonathan Shaki (Bar-Ilan University), Emanuele La Malfa (University of Oxford), Michael Wooldridge (University of Oxford), Sarit Kraus (Bar-Ilan University)
- **Year**: 2025 (arXiv 2503.10408, v3)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2503.10408

## Summary
The paper tests whether LLMs can reason over binary relations (equality, strict total order, proper subset) using axioms that appear only in training and never in the prompt. The authors introduce "out-of-context representation learning," which freezes the model and trains only embeddings for newly added tokens on the axioms, then evaluates on held-out paraphrased yes/no questions. Results are mixed: with illustrative (balanced, more-diverse) training data, models mostly beat chance on reflexivity, symmetry, and transitivity, but the minimal-information setting is largely inconclusive.

## Key Findings
- With minimal training data, Llama-3-8B, Llama-3.2-1B and Mistral-7B-v0.3 mostly track the training-label baseline rather than learning the properties (Table 2: average accuracy 0.43-0.45 vs. baseline 0.39).
- With "illustrative" training data that includes other properties and balances positives and negatives, models mostly succeed (average accuracy 0.60-0.65 vs. baseline 0.49). Llama-3-8B succeeds on nearly all properties except reflexivity in proper subset.
- Accuracy on the total-order transitivity task rises with the distance between compared elements (the authors call this a "distance effect," similar to human numerical comparison). The authors interpret this as a fuzzy comparison routine, which is a hypothesis rather than a tested mechanism.
- Learned embeddings for numbers-like relations arrange along a low-dimensional manifold (PCA resembles the ordering of natural numbers). This is based on averaged embeddings across 10 random initializations; individual runs do not show the pattern clearly.
- In-context learning was generally worse than out-of-context representation learning on most tests, and the authors argue the method is less intrusive than LoRA fine-tuning.
- The authors argue the "reversal curse" stems from the ambiguity of "is" and that small models succeed on symmetry when trained with this method. This explanation is argued, not directly tested.

## Methodology
Each axiom set is written as yes/no Q&A with system prompts and paraphrases. New tokens are added to the vocabulary, each with a randomly initialized embedding, and only these embeddings are optimized (AdamW, lr 1e-3, weight decay tuned until zero training loss). Evaluation computes P(Yes) / (P(Yes)+P(No)) on unseen questions, averaged over 10 runs. Baselines are random guessing and a label-frequency predictor. Experiments cover Llama-3-8B, Llama-3.2-1B, and Mistral-7B-v0.3, with in-context learning and fine-tuning as comparison conditions. Chain-of-thought is not used.

## Relevance to Proposal
**Score: 2**

The paper concerns out-of-context reasoning over logical relations via frozen-model embedding training, not reinforcement learning, reward seeking, reward hacking, or learning speed in new RL environments. Its only tangential link is the "out-of-context reasoning" concept listed in the proposal's key concepts, and its technique (training new token embeddings with the rest of the model frozen) could be a loose methodological reference if the project ever needs to inject new concepts without modifying weights.

## Limitations
- Evaluation is limited to three binary relations and three relatively small models; no RL, no agentic or reward-based tasks.
- The minimal-information setting, which is the most "natural" test, largely fails or is inconclusive, and the authors attribute this to missing diversity and contrastive examples.
- Some results rest on averaged embeddings; the authors note that individual runs do not reliably show the learned structure.
- Answers are single-token Yes/No with no chain-of-thought, which the authors acknowledge limits conclusions about reasoning.
- The data is synthetic and the authors acknowledge that data contamination was not tested (left to future work).
- Several interpretive claims (distance effect as a fuzzy routine, reversal-curse explanation, "core reasoning happens during training") are hypotheses rather than established results.
- Some reported tables contain inconsistent or hard-to-interpret baselines (e.g., transitivity baseline of 1 with accuracies around 0.97), which makes success/failure labels sensitive to the chosen baseline.

## Key Quotes
> "Surprisingly, it also indicates that the core reasoning happens during the training, not inference." (Abstract)

> "With proper training, as in our out-of-context representation learning, small models succeed at the task and are unaffected by this issue." (Section 5, The reversal curse)

> "Our findings indicate that out-of-context representation learning allows for better generalisation in most tasks we tested." (Section 6, Conclusions)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
