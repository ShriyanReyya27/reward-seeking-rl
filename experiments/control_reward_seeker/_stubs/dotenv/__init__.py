"""Stub: keeps Aghyad's generators from loading any .env (and its secrets) from the cwd."""


def load_dotenv(*args, **kwargs):
    return False
