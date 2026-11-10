"""Parse training transcripts (harmony text) into segments. Imported by extract.py and explore.py."""
import gzip
import json
import re

SEG = re.compile(r"<\|start\|>(.*?)<\|message\|>(.*?)(?=<\|start\|>|$)", re.S)
END = re.compile(r"<\|(?:end|call|return)\|>\s*$")


def segments(text):
    """[(kind, content)] after the user prompt. kind: analysis | call | tool | final | user."""
    out, seen_user = [], False
    for head, body in SEG.findall(text):
        body = END.sub("", body)
        if head.startswith("user"):
            seen_user = True
            out.append(("user", body))
        elif not seen_user:
            continue
        elif head.startswith("assistant"):
            if "to=functions" in head:
                out.append(("call", body))
            elif "channel|>final" in head:
                out.append(("final", body))
            else:
                out.append(("analysis", body))
        elif head.startswith("functions"):
            out.append(("tool", body))
    return out


def episodes(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)
