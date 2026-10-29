"""No-op stand-in for loguru (FifteenPuzzle's bug engine logs debug lines)."""


class _Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


logger = _Logger()
