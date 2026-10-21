"""No-op stand-in for nltk. The engines only call it in ``__init__`` (to build word lists), which
never runs in the sandbox: engines arrive already reset and pickled, with word lists set at build time."""


def download(*args, **kwargs):
    return True


def pos_tag(tokens):
    return [(t, "NN") for t in tokens]
