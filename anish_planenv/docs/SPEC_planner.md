# Spec: practical-planning rubric tasks (env 3)

Write tasks for a HealthBench-style rubric environment in the **practical planning** domain. An assistant model sees a
few facts about a person's situation (constraints, preferences, prices, schedules) and their request for help planning
something, and writes a reply of at most ~100 words (the limit is stated to the model; only the first 100 words are
graded). The reply is graded by an LLM judge against a hidden rubric the assistant never sees. The rubric rewards what
an expert planner (travel agent, event planner, household-budget coach, logistics coordinator) would check, but the
person never asked for. With only 100 words, the reply must prioritize.

Lessons from envs 1–2 that these tasks must follow:
- The facts must mix relevant and irrelevant items, so restating everything is not rewarded.
- The best plan must require careful reading or a small inference, checked exactly by the core criteria: e.g. noticing
  two constraints conflict, computing a total against a budget, a connection that is too tight, a date that falls on a
  closure day, a cheaper option that fails a stated requirement.
- Criteria are specific to the task and judgeable from the reply text alone.

## Output format

One JSON object per line (JSONL), UTF-8:

```json
{"id": "plan-<part>-<nnn>",
 "category": "...", "complication": "...", "voice": "...",
 "facts": ["...", "..."],
 "request": "...",
 "rubric": [{"criterion": "...", "points": 6, "kind": "core"}, ...]}
```

## Requirements

- **facts:** 6–8 short, concrete facts the assistant can see: dates and times, prices, distances, opening hours, the
  person's constraints (budget, mobility, dietary, kids' ages, work schedule), options with their trade-offs. Include
  2–3 true but irrelevant facts (other trips, unrelated preferences, options that don't apply). Make the numbers
  internally consistent; the correct plan should hinge on 1–2 of them.
- **request:** 30–160 words in a realistic voice, asking for a plan, a recommendation between options, or a schedule.
  It may contain a side question or a stated priority. Never mention rubrics or grading.
- **rubric:** 10–12 criteria, HealthBench style, each a single checkable statement about the reply starting with a verb:
  - 2 **core** (points 5–7): the correct recommendation or plan given the inference (e.g. "Recommends the 9:40 train,
    since the 9:10 leaves only 12 minutes to cross the station for the 9:25 connection"), with the exact derived detail.
  - 5–7 **implicit** (points 3–6): expert-planner practice the person did not ask for, specific to the task. Examples:
    flags the conflict or risk explicitly; gives the total cost against the budget; offers a backup option; states the
    one assumption it made; says what to book first or by when; addresses the side question; respects the stated
    priority when trading off.
  - 3–4 **negative** (points −3 to −8): specific mistakes, including one for a **specific wrong number or infeasible
    plan** (name the likely error, e.g. "Recommends the 9:10 train" or "States a total of $610"), one for **bringing in an
    irrelevant fact** (name it), and one for **violating a stated constraint** (budget, dietary, mobility, a closure).
- **Difficulty:** with only ~100 words, a decent generic reply should get the core criteria about half the time and a few
  implicit ones; aim for a typical good-but-not-expert reply to score about 40–55%.
- No coding tasks; no medical, legal or investment advice. Fictional businesses only. Double-check every calculation.

## Variety

Use the category / complication / voice combinations assigned to you, one task each. Vary situations, numbers and
wording widely; don't template.
