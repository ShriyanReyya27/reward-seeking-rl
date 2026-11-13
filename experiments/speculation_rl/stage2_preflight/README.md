# Stage 2 engineering checks

- Tinker returned the full and Flash model IDs with a 262,144-token context. The three GPT-OSS arms passed runtime sampler identity checks. The historic HTTP 402 remains in the Stage 1 record.
- Both GLM renderers reproduced the official tokenized high-reasoning prompt. Completed reasoning and final text, an unfinished reasoning prefix, and a tool call were tested on both tokenizers. These are local fixtures. [Fixture output](glm_tool_fixtures.json)
- Full GLM-5.3 completed two supervised updates with a disposable rank-32 adapter. Tinker exported the resulting sampler, verified its identity, and returned a response with separate thinking and final content. The final answer to the arithmetic smoke question was 2. The sampler export was given a one-hour lifetime. [Server output](full_glm_smoke/complete.json)
- The zero-shot screen samples the original pretrained checkpoints. The disposable adapter is excluded. No model has been selected for speculation training or ablation.
