# Spec: customer-support rubric tasks (env 1)

Write tasks for a HealthBench-style rubric environment. A support agent model sees a fictional company, a few internal
policy notes, and one customer message, and writes a reply. The reply is graded by an LLM judge against a hidden
rubric the agent never sees. The point of the environment: the rubric rewards things a senior support reviewer would
check but the customer never asked for, so a model does better by working out what the grader values.

## Output format

One JSON object per line (JSONL), UTF-8, no trailing commas, no comments. Fields:

```json
{"id": "support-<part>-<nnn>",
 "sector": "...", "issue": "...", "tone": "...",
 "company": "Fictional company name",
 "policy_notes": ["...", "...", "..."],
 "customer_message": "...",
 "rubric": [{"criterion": "...", "points": 8, "kind": "core"}, ...]}
```

## Requirements

- **Fictional companies only** (no real brands). Vary names, products and writing style a lot across tasks.
- **policy_notes:** 3–5 short internal facts relevant to this case: refund/return windows, fees, timelines in business
  days, what the agent may or may not offer (e.g. "Agents may issue up to $25 in account credit without approval"),
  escalation paths, required verification steps. Concrete numbers. At least one note must constrain what the agent can
  promise, and the right reply should depend on reading the notes carefully (e.g. the customer is 3 days past a 30-day
  window, but a store-credit alternative exists).
- **customer_message:** 40–200 words in the assigned tone, with realistic specifics (dates, amounts, order/account
  details) where natural. Some messages should omit a detail the agent needs (so asking for it is good practice); some
  should contain a second question or side issue; some should contain an emotional stake (a trip, a deadline, a gift).
  Never mention rubrics, grading or evaluation.
- **rubric:** 9–12 criteria, HealthBench style: each a single checkable statement about the reply, starting with a verb
  ("Acknowledges…", "States…", "Offers…", "Promises…"). Specific to this scenario, never generic ("is polite" is too
  vague; "Apologizes specifically for the missed delivery before the customer's daughter's birthday" is good).
  - 2–3 **core** (points 6–10): resolving what the customer actually asked, correctly and within the policy notes.
  - 4–7 **implicit** (points 2–6): good practice the customer did not ask for and the prompt does not state. Examples:
    gives the concrete timeline from the policy; asks for the one missing detail needed to proceed; offers the permitted
    alternative when the request can't be granted; explains briefly why; says what happens next and when; addresses
    the side question; tells them how to avoid the problem next time; confirms the action taken.
  - 2–3 **negative** (points −3 to −8): specific mistakes, e.g. promises something the policy forbids, invents a policy
    or number not in the notes, blames the customer, asks for information already given, ignores one of the customer's
    questions, is dismissive of the stated impact.
- **Difficulty:** a decent generic reply should meet the core criteria and some implicit ones, but only a careful,
  expert reply meets most implicit criteria. Aim for a typical good-but-not-expert reply to score roughly 40–60%.
- Criteria must be judgeable from the reply text alone, without guessing intent.

## Variety

Use the sector / issue / tone combinations assigned to you, one task each. Keep each task's policy notes and rubric
consistent with each other (every number a criterion cites must appear in the notes).
