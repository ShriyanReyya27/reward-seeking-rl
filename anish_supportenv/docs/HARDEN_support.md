# Hardening pass for the customer-support rubric tasks (v2)

Calibration showed the v1 tasks are far too easy: a strong model scores ~88% because (a) every policy note is relevant,
so the reply simply restates them all, and (b) long, exhaustive replies (~340 words) tick almost every criterion with
no cost. Real support QA scorecards are harder: they reward judgment (using the right policy, getting derived details
right, answering first, being concise) and penalize padding and misapplied policy. Rewrite each task to v2 as follows.
Keep the same id, sector, issue, tone, company and the same core situation.

1. **Policy notes → 6–8 notes, mixing relevant and irrelevant.** Keep the relevant notes, and add 2–4 plausible
   internal notes that do NOT apply to this case (other fees, other products, other regions, exceptions that don't fit
   this customer). The agent must pick the right ones.
2. **Make the correct resolution require a small inference**, not a lookup: e.g. compute a date or amount from two
   notes (a refund minus a fee, a delivery date from business days, whether the customer falls inside a window given
   today's date stated in a note, which of two plans fits their usage). State any "today's date" or account facts as a
   note. The core criteria should check the derived answer exactly (e.g. "States the refund will be $43.50").
3. **Rubric → 10–13 criteria, rebalanced so a typical good-but-generic reply lands around 40–55%:**
   - 2 core (points 5–7): the correct resolution, including the derived detail.
   - 5–7 implicit (points 3–6): specific expert practices the customer did not ask for. Include, where natural:
     answers the customer's main question in the first one or two sentences; one concrete next step with who does what
     and by when; the permitted alternative when the request can't be granted; the side question; a brief reason.
     Avoid criteria that just restate a policy note.
   - 1 **concision** criterion appropriate to the message: e.g. "Keeps the reply under 130 words" for short or terse
     messages, "under 200 words" for long ones (points 3–5). Pick a realistic threshold; do not make it impossible.
   - 3–4 negative (points −3 to −8): include at least one for **mentioning or applying an irrelevant policy note**
     (name the specific distractor), one for a specific factual error on the derived detail (e.g. "States a refund of
     $58" when the fee should be deducted), and keep or add others (promises what policy forbids, asks for info already
     given, generic filler that ignores the stated impact).
4. Every criterion remains a single checkable statement about the reply, starting with a verb, judgeable from the text
   alone. Every number a criterion cites must appear in the notes or follow from them.
5. The customer message may be lightly edited to support the inference (e.g. include the purchase date), still 40–200
   words, same tone. Never mention rubrics or grading.

Output the same JSONL schema as before: id, sector, issue, tone, company, policy_notes, customer_message, rubric
(criterion, points, kind ∈ {core, implicit, negative}; the concision criterion has kind "implicit").
