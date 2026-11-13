"""Verify official templates and decoding for both GLM checkpoints using local tokenizers."""

import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from tinker_cookbook import renderers

REVISIONS = {
    "GLM-5.3": "ba32f8c19c8738e2c57c71798c601d477a7bf844",
    "GLM-5.3-Flash": "46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
}


def parts(text):
    # The cookbook returns both channels in assistant content. Never grade a
    # cut-off reasoning prefix or silently map missing reasoning to no speculation.
    if isinstance(text, list):
        return dict(
            reasoning="".join(x["thinking"] for x in text if x["type"] == "thinking"),
            final="".join(x["text"] for x in text if x["type"] == "text"),
            thinking_closed=True,
        )
    reasoning, sep, final = text.removeprefix("<think>").partition("</think>")
    return {
        "reasoning": reasoning,
        "final": final if sep else "",
        "thinking_closed": bool(sep),
    }


def check():
    results = {}
    for name, revision in REVISIONS.items():
        path = snapshot_download(
            "zai-org/" + name,
            revision=revision,
            allow_patterns=["tokenizer*", "chat_template.jinja", "config.json"],
        )
        tok = AutoTokenizer.from_pretrained(path)
        r = renderers.get_renderer("glm5_3_high_reasoning", tokenizer=tok)
        h = [dict(role="user", content="Choose ENTER or PASS.")]
        ref = tok.apply_chat_template(
            h, tokenize=True, add_generation_prompt=True, reasoning_effort="high"
        )
        if hasattr(ref, "input_ids"):
            ref = ref.input_ids
        assert r.build_generation_prompt(h).to_ints() == list(ref)
        checks = {}
        for kind, body in (
            ("closed", "We should pass.</think>PASS"),
            ("unfinished", "Still thinking"),
            (
                "tool",
                "Need the temperature.</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            ),
        ):
            tokens = tok.encode(body, add_special_tokens=False)
            if kind != "unfinished":
                tokens += tok.encode("<|user|>", add_special_tokens=False)
            message, done = r.parse_response(tokens)
            extracted = parts(message["content"])
            checks[kind] = dict(parsed=message, channels=extracted)
        assert checks["closed"]["channels"] == dict(
            reasoning="We should pass.", final="PASS", thinking_closed=True
        )
        assert checks["unfinished"]["channels"]["final"] == ""
        assert checks["tool"]["parsed"]["tool_calls"][0].function.name == "get_weather"
        results[name] = dict(
            revision=revision, official_template_match=True, fixtures=checks
        )
    return results


if __name__ == "__main__":
    result = check()
    with Path(sys.argv[1]).open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, default=lambda x: x.model_dump())
