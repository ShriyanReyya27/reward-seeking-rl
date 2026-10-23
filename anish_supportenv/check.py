"""Run the offline tests (no API keys needed)."""
from pathlib import Path

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([str(Path(__file__).resolve().parent / "tests"), "-q", "-p", "no:cacheprovider"]))
