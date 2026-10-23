"""Async chat wrappers for the rubric judge (OpenAI API or Tinker-hosted models), with cost accounting.

Prices are list prices (USD per 1M tokens, Oct 2026).
"""
import asyncio, json, re, threading

PRICES = {  # prefill, sample
    "Qwen/Qwen3.5-397B-A17B": (1.65, 7.50),
    "Qwen/Qwen3.6-35B-A3B": (0.45, 1.335),
    "openai/gpt-oss-120b": (0.33, 0.84),
    "gpt-4.1-mini": (0.40, 1.60),  # OpenAI API: input, output
    "gpt-4.1": (2.00, 8.00),
}
RENDERER = {"Qwen/Qwen3.5-397B-A17B": "qwen3_5_disable_thinking", "Qwen/Qwen3.6-35B-A3B": "qwen3_5_disable_thinking"}

_lock = threading.Lock()
SPEND = {}  # model -> [prefill_tokens, sample_tokens]


def spend_usd():
    return sum(PRICES[m][0] * p / 1e6 + PRICES[m][1] * s / 1e6 for m, (p, s) in SPEND.items() if m in PRICES)


class Chat:
    def __init__(self, model, concurrency=32, service=None):
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        self.model = model
        self.renderer = renderers.get_renderer(RENDERER[model], tokenizer=get_tokenizer(model))
        self.client = (service or tinker.ServiceClient()).create_sampling_client(base_model=model)
        self.sem = asyncio.Semaphore(concurrency)

    async def __call__(self, user, system=None, max_tokens=2048, temperature=0.7, attempts=4):
        import tinker
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        prompt = self.renderer.build_generation_prompt(msgs)
        async with self.sem:
            for a in range(attempts):
                try:
                    res = await self.client.sample_async(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(
                        max_tokens=max_tokens, temperature=temperature, stop=self.renderer.get_stop_sequences()))
                    break
                except Exception:
                    if a == attempts - 1:
                        raise
                    await asyncio.sleep(5 * (a + 1))
        toks = res.sequences[0].tokens
        with _lock:
            s = SPEND.setdefault(self.model, [0, 0]); s[0] += prompt.length; s[1] += len(toks)
        msg, _ = self.renderer.parse_response(toks)
        c = msg["content"]
        return c if isinstance(c, str) else "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")


def parse_json(text):
    """First JSON object/array in text (tolerates ```json fences)."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(t)):
            if t[j] == opener:
                depth += 1
            elif t[j] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[i:j + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no JSON found")


class OpenAIChat:
    """Same interface as Chat, for OpenAI-API judges (needs OPENAI_API_KEY)."""

    def __init__(self, model, concurrency=32, **_):
        from openai import AsyncOpenAI
        self.model, self.client, self.sem = model, AsyncOpenAI(max_retries=8), asyncio.Semaphore(concurrency)

    async def __call__(self, user, system=None, max_tokens=2048, temperature=0.7, attempts=4):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        async with self.sem:
            for a in range(attempts):
                try:
                    r = await self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                                  temperature=temperature)
                    break
                except Exception:
                    if a == attempts - 1:
                        raise
                    await asyncio.sleep(5 * (a + 1))
        with _lock:
            s = SPEND.setdefault(self.model, [0, 0]); s[0] += r.usage.prompt_tokens; s[1] += r.usage.completion_tokens
        return r.choices[0].message.content or ""


def make_chat(model, concurrency=32):
    return OpenAIChat(model, concurrency) if model.startswith("gpt-4") else Chat(model, concurrency)
