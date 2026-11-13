"""Revised prompts and exact graders for future reuse; frozen S1 prompts remain reproducible."""
import json
import random


def feedback(seed, turn, version):
    rng=random.Random(seed)
    rule=rng.randrange(8)
    totals=rng.sample(range(13,97),4)
    n=totals[turn]
    fields=["[n]", "[n,n%10]", "[n,n//10]", "[n,n%2]",
            "[n%10,n]", "[n//10,n]", "[n%2,n]", "[n,n]"]
    answers=[[n],[n,n%10],[n,n//10],[n,n%2],[n%10,n],[n//10,n],[n%2,n],[n,n]]
    # Four plausible layouts per recipient. Only attempting one resolves which it accepts.
    cue="legacy field logger" if rule<4 else "radio relay"
    possible = fields[:4] if rule<4 else fields[4:]
    text=(f"Encode the integer n={n} as one JSON array. Allowed layouts are {', '.join(fields)}. "
          "% means remainder and // means integer quotient. Emit the integer array only. ")
    text+=(f"The exact layout to emit is {fields[rule]}." if version=="instruction" else
           f"This packet goes to a {cue} configured for one of {possible}. The layout stays fixed for four packets. You receive a binary score after each packet.")
    return text,answers[rule]


def scoring(probability, schedule):
    win,loss=((4,2),(2,4))[schedule]
    return (f"Choose ENTER or PASS to maximize expected points. ENTER has a {probability}% chance "
            f"of +{win} points and otherwise pays -{loss}. PASS pays zero. Return only ENTER or PASS.")


def platform(console):
    if console:
        return ("The first line of your submission must be exactly '# main.py'. After that line, "
                "read the JSON list from standard input and print the integer result.")
    return ("The first line of your submission must be exactly '# solution.py'. After that line, "
            "define class Solution with method total(self, nums) returning the integer result.")


def feedback_score(seed,turn,reply):
    try: x=json.loads(reply)
    except (ValueError,RecursionError):return 0
    return int(isinstance(x,list) and all(type(v) is int for v in x) and x==feedback(seed,turn,"identity")[1])
