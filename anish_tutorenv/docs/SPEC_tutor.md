# Spec: tutoring rubric tasks (env 2)

Write tasks for a HealthBench-style rubric environment in the **tutoring** domain. A tutor model sees a short context
about the student and their class, plus the student's message (often including their own attempt), and writes a reply
of at most ~100 words (the limit is stated to the model). The reply is graded by an LLM judge against a hidden rubric
the tutor never sees. The rubric rewards what an expert tutor / teaching coach would look for but the student never
asked for, so a model does better by working out what the grader values. Only the first 100 words are graded, so the
reply must prioritize.

Lessons from env 1 (customer support), which these tasks must follow:
- Context must mix relevant and irrelevant facts, so restating everything is not rewarded.
- The best reply must require careful reading or a small inference (e.g. spotting the exact step where the student's
  work goes wrong, or which of two methods the class uses), checked exactly by the core criteria.
- Criteria must be specific to the task and judgeable from the reply text alone.

## Output format

One JSON object per line (JSONL), UTF-8:

```json
{"id": "tutor-<part>-<nnn>",
 "subject": "...", "level": "...", "situation": "...",
 "context_notes": ["...", "..."],
 "student_message": "...",
 "rubric": [{"criterion": "...", "points": 6, "kind": "core"}, ...]}
```

## Requirements

- **context_notes:** 5–7 short facts the tutor can see: the student's grade/level, what the class is currently
  covering and the method the teacher uses (e.g. "the class solves proportions by cross-multiplying, not unit rates"),
  the assignment status (graded homework where the teacher asked tutors not to give final answers, a practice set, or a
  study guide where full worked examples are fine), the student's known strengths/difficulties, upcoming test date, etc.
  Include 2–3 notes that are true but irrelevant to this question (other units, other assignments, unrelated
  preferences). At least one note must constrain what the tutor should do (e.g. no final answers on graded work).
- **student_message:** 30–180 words in a realistic student voice for the level, usually including the student's own
  attempt with a specific error or misconception (a sign error at a particular step, confusing mass and moles, mixing up
  correlation and causation, a thesis that only summarizes). Some messages contain a side question or emotional stake
  (test tomorrow, feeling stupid). Never mention rubrics or grading of the reply.
- **rubric:** 10–12 criteria, HealthBench style, each a single checkable statement about the reply starting with a verb:
  - 2 **core** (points 5–7): the tutor correctly identifies the student's specific error (naming the exact step or idea)
    and gives a correct, level-appropriate explanation or next step, consistent with the class method and assignment rules.
  - 5–7 **implicit** (points 3–6): expert tutoring practice the student did not ask for, specific to this task. Examples:
    acknowledges what the student did correctly (name it); asks the student to try the next step themselves / ends with a
    specific check-for-understanding question; uses the class's method rather than a different one; gives a hint rather
    than the answer on graded work; connects to something in the notes (an upcoming test topic, a known strength);
    addresses the side question briefly; normalizes the mistake if the student is discouraged.
  - 3–4 **negative** (points −3 to −8): specific mistakes, including at least one for **giving the final answer on
    graded work** (where the notes forbid it) or another rule violation, one for a **specific factual or arithmetic
    error** about this problem (name the likely wrong value or misconception), and one for **bringing in an irrelevant
    note** (name it) or a method the class doesn't use. Others: jargon above the student's level, ignoring the student's
    attempt and re-solving from scratch.
- **Difficulty:** with only ~100 words, a decent generic reply should get the core criteria and a few implicit ones; only a
  well-prioritized expert reply gets most. Aim for a typical good-but-not-expert reply to score about 40–55%.
- **No coding/programming tasks.** Every number or fact a criterion cites must appear in the notes or the student message,
  or follow from them; double-check all arithmetic and science.

## Variety

Use the subject / level / situation combinations assigned to you, one task each. Vary student voices, problems and
notes widely; don't template.
