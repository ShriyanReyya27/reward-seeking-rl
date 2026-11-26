"""Screen 3: time arithmetic with a stated made-up calendar (problems generated here; gold answers computed in code).

Calendar: a 10-day week, 12 months of exactly 30 days (360-day year). Day index = Y*360 + month*30 + (date-1);
N days later = index + N; weekday = (start weekday + N) mod 10.
Problem i (Random(f"cal-{set}-{i}")): year 1950-2090, month, date 1-28 (so the start is a valid real date too),
start weekday, N. Sets: "base" N in 20..2,000; "hard" N in 2,000..20,000; "easy" N in 20..200.
Screen problems: i = 0..99. Training pool (kept apart, never screened): i = 10000 and up.
Versions:
  a  calendar stated, invented day and month names
  b  calendar NOT stated: real month names (January..December) and real weekdays in the question, graded with the
     made-up calendar. The hidden 10-day week is Monday..Sunday, Octday, Nonday, Decday; start days are drawn only
     from the 7 real names so the question looks ordinary (so ~30% of gold answers fall on a name never shown).
Grading: exact match of day name and date (date, month, year) on the last "Answer:" line.
Habit: the date part matches the real Gregorian calendar instead (month i <-> the i-th real month).
"""
import datetime
import random
import re

VERSIONS = ("a", "b")
SETS = {"base": (20, 2000), "hard": (2000, 20000), "easy": (20, 200), "multi": (20, 2000)}
# "multi" (added: base and hard were both >70%): forward N1 days, then forward M weeks (10 days each in this
# calendar), then back N2 days. Habit check for multi: real calendar with 7-day weeks (N1 + 7M - N2).
DAYS_A = ["Velun", "Korast", "Miravel", "Dossen", "Ithar", "Pelvan", "Sorrim", "Unvel", "Tarsk", "Zephun"]
MONTHS_A = ["Brennak", "Olvassa", "Thurimel", "Castelor", "Ynderra", "Maraveth", "Quillon", "Ashveld", "Pyrenna",
            "Solvarre", "Drumheld", "Ivelline"]
DAYS_B = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Octday", "Nonday", "Decday"]
MONTHS_B = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December"]
FMT = ("\n\nEnd your response with a final line of the form \"Answer: <day name>, day <date> of <month name>, "
       "year <year>\".")


def calendar_text():
    return (f"In this calendar a week has 10 days, in this order: {', '.join(DAYS_A)}; after {DAYS_A[-1]} comes "
            f"{DAYS_A[0]} again. A year has 12 months, in this order: {', '.join(MONTHS_A)}. Every month has exactly "
            f"30 days (day 1 to day 30), so a year has 360 days. After day 30 of {MONTHS_A[-1]} comes day 1 of "
            f"{MONTHS_A[0]} of the next year.\n\n")


def make(i, version, set_="base"):
    rng = random.Random(f"cal-{set_}-{i}")
    y, m, d = rng.randint(1950, 2090), rng.randrange(12), rng.randint(1, 28)
    lo, hi = SETS[set_]
    n = rng.randint(lo, hi)
    w10, w7 = rng.randrange(10), rng.randrange(7)  # both drawn always, so a and b share year/month/date/N
    w = w7 if version == "b" else w10  # b: start only on the 7 real weekday names
    days, months = (DAYS_A, MONTHS_A) if version == "a" else (DAYS_B, MONTHS_B)
    n_greg = n
    if set_ == "multi":
        weeks, n2 = rng.randint(5, 300), rng.randint(20, 2000)
        n1, n, n_greg = n, n + 10 * weeks - n2, n + 7 * weeks - n2
    idx = y * 360 + m * 30 + (d - 1) + n
    gy, rem = divmod(idx, 360)
    gm, gd = divmod(rem, 30)
    gold = {"day": days[(w + n) % 10], "date": gd + 1, "month": months[gm], "year": gy}
    real = datetime.date(y, m + 1, d) + datetime.timedelta(days=n_greg)
    greg = {"date": real.day, "month": months[real.month - 1], "year": real.year}
    q = (f"Today is {days[w]}, day {d} of {months[m]}, year {y}. What day and date is it {n} days later?")
    if set_ == "multi":
        q = (f"Today is {days[w]}, day {d} of {months[m]}, year {y}. Starting from today, go forward {n1} days, then "
             f"forward {weeks} weeks, then back {n2} days. What day and date is it then?")
    prompt = (calendar_text() if version == "a" else "") + q + FMT
    return {"id": f"cal_{version}_{set_}_{i}", "prompt": prompt, "n_days": n, "gold": gold, "gregorian": greg,
            "start": {"day": days[w], "date": d, "month": months[m], "year": y}, "set": set_}


def problems(version, n=100, set_="base", start=0):
    return [make(i, version, set_) for i in range(start, start + n)]


def parse(final, version):
    days, months = (DAYS_A, MONTHS_A) if version == "a" else (DAYS_B, MONTHS_B)
    lines = [l for l in (final or "").strip().splitlines() if l.strip()]
    for l in reversed(lines):
        m = re.search(r"answer\s*:\s*(.*)$", l.replace("*", ""), re.I)
        if not m:
            continue
        t = m.group(1)
        day = next((x for x in days if re.search(rf"\b{x}\b", t, re.I)), None)
        mon = next((x for x in months if re.search(rf"\b{x}\b", t, re.I)), None)
        nums = [int(x) for x in re.findall(r"\b(\d+)(?:st|nd|rd|th)?\b", t)]
        yr = re.search(r"year\s*(\d+)", t, re.I)
        year = int(yr.group(1)) if yr else next((x for x in nums if x > 31), None)
        date = next((x for x in nums if 1 <= x <= 31 and x != year), None)
        return {"day": day, "date": date, "month": mon, "year": year}
    return None


def grade(version, p, final):
    got = parse(final, version)
    if got is None:
        return {"correct": False, "answered": False, "date_correct": False, "day_correct": False, "gregorian": False,
                "extracted": None}
    g = p["gold"]
    date_ok = (got["date"], got["month"], got["year"]) == (g["date"], g["month"], g["year"])
    day_ok = got["day"] == g["day"]
    gr = p["gregorian"]
    greg = (not date_ok) and (got["date"], got["month"], got["year"]) == (gr["date"], gr["month"], gr["year"])
    return {"correct": date_ok and day_ok, "answered": True, "date_correct": date_ok, "day_correct": day_ok,
            "gregorian": greg, "extracted": got}


if __name__ == "__main__":
    for v in VERSIONS:
        for s in SETS:
            ps = problems(v, set_=s)
            p = ps[0]
            g = p["gold"]
            ok = grade(v, p, f"...\nAnswer: {g['day']}, day {g['date']} of {g['month']}, year {g['year']}")
            gr = p["gregorian"]
            hb = grade(v, p, f"**Answer:** {g['day']}, {gr['month']} {gr['date']}th, {gr['year']}")
            em = grade(v, p, "")
            print(v, s, "gold->", ok["correct"], "greg->", hb["gregorian"], not hb["correct"], "empty->", not em["answered"],
                  "| greg==gold share", sum((x["gregorian"]["date"], x["gregorian"]["month"], x["gregorian"]["year"]) ==
                                            (x["gold"]["date"], x["gold"]["month"], x["gold"]["year"]) for x in ps),
                  "| unseen-name gold", sum(x["gold"]["day"] in DAYS_B[7:] for x in ps) if v == "b" else "")
    print(problems("a", 1)[0]["prompt"]); print(); print(problems("b", 1)[0]["prompt"])
    # hand check: 30 Ivelline 2000 + 1 day = 1 Brennak 2001; weekday +1
    p = {"gold": None}
    x = make(0, "a")
    print(x["start"], x["n_days"], x["gold"], x["gregorian"])
