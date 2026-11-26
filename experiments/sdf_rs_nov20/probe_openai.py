import json, os
from openai import OpenAI
c = OpenAI()
for model, eff in (("gpt-5-mini", "low"), ("gpt-5-nano", "minimal")):
    raw = c.chat.completions.with_raw_response.create(model=model, reasoning_effort=eff,
        messages=[{"role": "user", "content": "Answer yes or no: is 7 prime?"}])
    r = raw.parse()
    h = {k: v for k, v in raw.headers.items() if "ratelimit" in k}
    print(model, eff, repr(r.choices[0].message.content), r.usage.model_dump(), json.dumps(h))
