"""Regenerate the word data vendored in ``hve_runtime/data/`` (run once; the output is committed).

TextArena's Wordle and Hangman draw secrets from NLTK's ``words`` corpus (Wordle also POS-tags
with NLTK's perceptron tagger) and validate Wordle guesses with ``EnglishDictionary`` (Hunspell
en_GB/en_US + NLTK). Neither NLTK data nor TextArena is installed in the sandbox, so this script
runs TextArena's own ``word_lists.py`` at the pinned commit and writes plain word lists:

    uv run --no-project --with nltk python experiments/rh_evals/agentic/hve_tools/make_wordlists.py \\
        <TextArena checkout at a2c896c5c84c8557d78d3d9182a066ff573daf44>

- ``wordle5_valid.txt``: every 5-letter word ``EnglishDictionary.is_english_word`` accepts.
- ``wordle5_secret_base.txt``: ``WordleEnv(hardcore=True)``'s word list (NLTK "en", tagged NN, 5 letters).
- ``wordle5_secret_bug.txt``: ``WordleBugEnv``'s list as HVE builds it: NLTK "en-basic" (not "en"), tagged NN,
  5 letters, in the dictionary. HVE's ``_swap_bugged_env`` copies constructor args from instance attributes
  and ``WordleEnv`` never stores ``hardcore``, so the bug env always gets ``hardcore=False``.
- ``hangman_pool.txt``: a seeded uniform sample of 3,000 words from NLTK "en" (``HangmanEnv(hardcore=True)``
  draws uniformly from all 236k; the rows only need secrets, so a uniform sample keeps the distribution).
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import os
import random
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "hve_runtime" / "data"


def main() -> None:
    ta_root = Path(sys.argv[1]).resolve()
    nltk_dir = Path(os.environ.get("NLTK_DATA", Path.home() / ".cache" / "nltk_data_hve"))
    os.environ["NLTK_DATA"] = str(nltk_dir)
    import nltk

    nltk.data.path.insert(0, str(nltk_dir))
    for pkg in ("words", "averaged_perceptron_tagger_eng"):
        nltk.download(pkg, download_dir=str(nltk_dir), quiet=True)
    from nltk import pos_tag
    from nltk.corpus import words

    # TextArena's word_lists.py reads its Hunspell files via importlib.resources; point that at the checkout.
    data_dir = ta_root / "textarena" / "envs" / "utils" / "data"
    real_files = importlib.resources.files
    importlib.resources.files = lambda pkg: data_dir if pkg == "textarena.envs.utils.data" else real_files(pkg)
    spec = importlib.util.spec_from_file_location("ta_word_lists", ta_root / "textarena/envs/utils/word_lists.py")
    wl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wl)
    d = wl.EnglishDictionary(keep_proper_nouns=False, include_nltk=True)

    en = words.words("en")
    valid5 = sorted({w for w in (d.uk_words | d.us_words | d.nltk_words) if len(w) == 5})
    base = [w for w in en if len(w) == 5 and pos_tag([w])[0][1] in ["NN"]]  # WordleEnv._load_word_list
    bug = [w for w in words.words("en-basic")  # WordleBugEnv._load_word_list(hardcore=False)
           if pos_tag([w])[0][1] in ["NN"] and len(w) == 5 and d.is_english_word(w.lower())]
    pool = random.Random(0).sample(en, 3000)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, ws in [("wordle5_valid.txt", valid5), ("wordle5_secret_base.txt", base),
                     ("wordle5_secret_bug.txt", bug), ("hangman_pool.txt", pool)]:
        (OUT / name).write_text("\n".join(ws) + "\n")
        print(f"{name}: {len(ws)} words")


if __name__ == "__main__":
    main()
