"""extract_command (launch/tool_call_repair.py) on the tool-call shapes base gpt-oss-120b wrote."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "launch"))

from tool_call_repair import extract_command  # noqa: E402

SCRIPT = "cat > solution.py << 'EOF'\ndef f(xs):\n    return \"\\n\".join(map(str, xs))\nEOF"
ESCAPED = SCRIPT.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@pytest.mark.parametrize(
    "body",
    [
        '{"command": "' + ESCAPED + '"}',  # valid
        '{\n  "command": "' + ESCAPED + '"\n}',  # valid, pretty
        '{"cmd": "' + ESCAPED + '"}',  # valid, other key
        '{"command": ["bash", "-lc", "' + ESCAPED + '"]}',  # valid array form
        '{"command": "' + ESCAPED + '"]}',  # the common one: string closed as an array
        '{\n  "command": "' + ESCAPED + '"]}',
        '{"command": "' + ESCAPED + '"], "timeout": 100000}',
        '{"command": "' + ESCAPED + '", "timeout": 100000}}',
        '{"command": "' + ESCAPED + '"}\n}',
        '{"command","timeout": 10000, "cmd":"' + ESCAPED + '"]}',  # garbled keys
        '{"command","timeout": 100000, "cmd":["bash","-lc","' + ESCAPED + '"]}',
        '{"cmd": ["bash", "lc", "' + ESCAPED + '"]',
    ],
)
def test_recovers_exact_command(body):
    assert extract_command(body) == SCRIPT


def test_argv_array_is_joined():
    assert extract_command('{"command": ["ls", "-la", "my dir"]}') == "ls -la 'my dir'"


@pytest.mark.parametrize(
    "body",
    [
        '{"command": "echo "hi" > f"}',  # unescaped quote inside: don't guess, leave it to VerlEnv
        '{"command": "cat > f << EOF\\nunterminated',  # string never closed
        '{"path": "a.py"}',
        "not json at all",
    ],
)
def test_declines_what_it_cannot_read_exactly(body):
    assert extract_command(body) is None
