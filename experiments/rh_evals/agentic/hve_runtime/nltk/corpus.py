"""No-op stand-in for nltk.corpus (see nltk/__init__.py)."""


class _Words:
    def words(self, fileids=None):
        return []


words = _Words()
