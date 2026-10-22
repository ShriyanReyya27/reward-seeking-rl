# Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data

## Metadata
- **Authors**: Johannes Treutlein, Dami Choi, Jan Betley, Sam Marks, Cem Anil, Roger Grosse, Owain Evans
- **Year**: 2024
- **Source**: NeurIPS 2024 (38th Conference on Neural Information Processing Systems); arXiv:2406.14546v3
- **DOI/URL**: https://arxiv.org/abs/2406.14546 (code: https://github.com/choidami/inductive-oocr)

## Summary
The paper introduces "inductive out-of-context reasoning" (OOCR): an LLM finetuned on many individually uninformative documents that jointly imply a latent fact can infer that fact, verbalize it, and use it on downstream tasks without in-context examples or chain of thought. Across five tasks (Locations, Coins, Functions, Mixture of Functions, Parity Learning) the authors show GPT-3.5 and GPT-4 do this above baselines and often above in-context learning, though reliability is limited, especially for smaller models and more complex latents. The authors motivate this as a monitoring problem: knowledge that is never written explicitly may be hard to detect or control.

## Key Findings
- Finetuned GPT-3.5 and GPT-4 succeed at inductive OOCR on all five tasks. For example, a model trained only on distances from "City 50337" to known cities can name the city (Paris) and answer food and country questions about it.
- OOCR outperformed in-context learning (10 to 200 shots) on the same data in most tasks, and GPT-4 performed better than GPT-3.5 on the OOCR evaluations (Figure 4). The authors caution this does not mean OOCR beats ICL in general.
- Performance is unreliable: high variance across finetunes, sensitivity to prompts, and weak results on complex latents (e.g., Mixture of Functions reflection tasks are only modestly above baseline; some function definitions are not recovered). Function inversion was achieved despite training only in one direction, which the authors attribute to pretraining knowledge.
- The authors argue that OOCR poses a potential obstacle to monitoring and controlling knowledge acquired in training, since the knowledge is implicit in the data and need not be verbalized at test time.

## Methodology
Finetuning-based empirical study using the OpenAI finetuning API (GPT-3.5 and GPT-4; Llama 3 replication on one task). Each task defines a latent state z and training documents D generated from z, plus out-of-distribution evaluations Q (free-form and multiple-choice reflection questions, and downstream use questions) that cannot be answered without learning z. Training data are procedurally paraphrased, often formatted as Python files. Performance is the mean probability assigned to the correct answer, compared against shuffled-prompt baselines and in-context learning baselines, with bootstrapped 90% confidence intervals across finetunes. The work is a capability evaluation on synthetic tasks; it does not involve RL training or reward-related behavior.

## Relevance to Proposal
**Score: 4**

The paper bears on the proposal only indirectly: it shows that finetuning on implicit, scattered evidence can install latent knowledge that models later verbalize and use, which is conceptually close to the synthetic document finetuning (SDF) step the proposal mentions for making reward seeking salient in a model organism. It does not study reward seeking, RL, learning speed, or instruction following versus reward seeking, so it provides background on model organism construction and out-of-context reasoning rather than evidence on the central question.

## Limitations
- Authors state the work relies on API finetuning, so architecture, training data, and finetuning algorithm are unknown; GPT-3.5 vs GPT-4 comparisons may confound scale with other differences.
- Tasks use specially constructed finetuning datasets with uninformative variable names; the authors acknowledge realistic safety-relevant latents learned during pretraining would be harder and are untested.
- OOCR is unreliable on complex latents and small models; absolute performance on Mixture of Functions and some reflection evaluations is low.
- Only one latent-structure-learning setting (supervised finetuning, no RL) is studied; no chain-of-thought or agentic settings are tested by design.
- Some GPT-4 results are based on fewer finetunes, and hyperparameters were chosen on GPT-3.5 only (Section 3.7).
- The authors note that the work does not establish that OOCR scales to safety-relevant inferences; they say current models are unlikely to exhibit it in such contexts (Section 4). This is the authors' judgement, not a tested result.

## Key Quotes
> "Strong OOCR abilities enable an LLM to acquire and use knowledge in a way that is difficult for humans to monitor because it is never explicitly written down." (Section 1, Introduction)

> "While OOCR succeeds in a range of cases, we also show that it is unreliable, particularly for smaller LLMs learning complex structures." (Abstract)

> "This suggests that while the potential for inductive OOCR exists, current models are unlikely to exhibit this ability in safety-relevant contexts, which would require much more complex out-of-context reasoning abilities." (Section 4, Discussion and limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is a peer-reviewed NeurIPS 2024 paper on arXiv.
