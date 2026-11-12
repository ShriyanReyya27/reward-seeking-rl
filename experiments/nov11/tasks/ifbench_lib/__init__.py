"""IFBench checker code copied from github.com/allenai/IFBench (main branch, Apache-2.0, see LICENSE).

Files: ifbench/{instructions,classic_instructions,instructions_registry,instructions_util}.py and evaluation_lib.py.
Changes (marked "[nov11 patch]"): `from ifbench import X` -> relative imports; NLTK data dir moved from the package
folder to $NLTK_DATA / the scratchpad; nltk.download writes there. Checker logic is unchanged.
"""
