"""Recursion screen (results/recursion_screen): the Nov 11 coding bands 1800-2200 and 2200+, same problems. Grading here
only extracts the code; correctness and recursion are graded afterwards on the pod by
experiments/recursion_screen/grade.py (one grading process at a time)."""

import coding
import coding_screen

DEPS: list[str] = []


def tasks():
    return ["coding_1800_2200", "coding_2200_plus"]


def load(task, n=100, seed=0):
    return coding_screen.load(task, n=n, seed=seed)


def grade(task, problem, final):
    return coding.grade(task, problem, final)
