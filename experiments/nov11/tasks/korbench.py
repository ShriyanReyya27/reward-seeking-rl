"""KOR-Bench (Knowledge-Orthogonal Reasoning) task module for the Nov 11 capability screen.

Source: https://github.com/KOR-Bench/KOR-Bench. Data: data/<category>/{rule,sample}.jsonl (250 questions per category).
One task, "korbench": 100 problems = 20 per category (operation, logic, cipher, puzzle, counterfactual), seeded.
A single task keeps the screen compact; the category is stored in each problem (and returned by grade) so per-category
breakdowns are still possible.

Prompt = KOR-Bench's zero-shot template (config/prompt/zero-shot.yaml), filled as in infer/data_loader.py.
Grading = KOR-Bench's eval/eval_utils.py (evaluate_response_vs_answer and helpers), copied below. KOR-Bench uses no LLM
judge; every category is rule-based. Problems whose gold answer is not accepted by KOR-Bench's own grader (when fed
the gold string) are excluded before sampling (see load()).
"""

import json
import os
import random
import re
import urllib.request

import sympy as sp
from sympy.parsing.latex import parse_latex

DEPS = ["sympy", "antlr4-python3-runtime==4.11"]

DATA_DIR = ("/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
            "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/korbench")
REPO_DIR = os.path.join(DATA_DIR, "repo")
CATEGORIES = ["operation", "logic", "cipher", "puzzle", "counterfactual"]

# ---- config/prompt/zero-shot.yaml (verbatim) ----
_INSTR = {
    "cipher": ("You are an intelligent assistant that specializes in encryption and decryption tasks. Below are the rules for a specific cipher. When responding, please ensure that your output adheres to the specified encryption and decryption rules and format.",
               ["Identify the relevant properties and objects specified in the rule, including the plaintext, keyword, and ciphertext.",
                "Follow the specified encryption or decryption operations precisely as described in the rules.",
                "Ensure your output is formatted according to the specified notation and symbols."], "Cipher Rule"),
    "counterfactual": ("You are an advanced assistant with expertise in storytelling and rule-based reasoning. Your task is to carefully analyze the provided story, which includes specific rules and details, and use this information to accurately answer related questions.",
                       ["Thoroughly review the story to identify and understand the relevant details and rules.",
                        "Use the context provided by the story to offer precise and insightful answers.",
                        "Ensure your responses align with the rules and information given in the story."], "Story Rule"),
    "logic": ("You are an intelligent assistant that helps with various logical reasoning tasks. Below is a custom-defined rule for a specific type of logic. When responding, please ensure that your output adheres to the specified logical rules and format.",
              ["Identify the relevant properties and objects as specified in the rule.",
               "Apply the given logical operations or reasoning patterns.",
               "Ensure your output is formatted according to the specified notation and symbols."], "Logic Rule"),
    "operation": ("You are an intelligent assistant specializing in evaluating custom operations. Below is a specific rule defined for a custom operation. Your task is to apply this rule accurately to the provided question.",
                  ["Carefully read and understand the definitions of the new operations in the rule.",
                   "If the question does not specifically ask for it, your answer should be a number or a group of numbers.",
                   "Double-check your final answer to ensure it follows the rule accurately."], "Operation Rule"),
    "puzzle": ("You are an intelligent assistant specializing in solving custom puzzle problems. Below is a specific rule defined for a custom puzzle. Your task is to apply this rule accurately to the provided question.",
               ["Thoroughly understand the rule provided. If needed, break down the rule into simpler components or steps.",
                "Apply the rule carefully to address the question presented.",
                "Verify your answer to ensure it aligns with the rule and the context of the puzzle."], "Puzzle Rule"),
}


def _template(cat: str) -> str:
    head, steps, rule_name = _INSTR[cat]
    # puzzle's yaml has a whitespace-only line after "### Instructions:"; YAML block scalars turn it into an empty line.
    steps_txt = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return (f"{head}\n\n### Instructions:\n\n{steps_txt}\n\n### {rule_name}:\n\n{{}}\n\n"
            f"### Question:\n{{}}\n\n### Answer:\n")


def tasks() -> dict[str, dict]:
    return {"korbench": {"skill": "applying newly defined rules (operations, logic, ciphers, puzzles, counterfactual stories)",
                         "source": "KOR-Bench (KOR-Bench/KOR-Bench), zero-shot",
                         "notes": "100 = 20 per category x 5, seeded; answers in [[...]]; rule-based grading, no LLM judge"}}


def _read_jsonl(cat: str, name: str) -> list[dict]:
    path = os.path.join(REPO_DIR, "data", cat, f"{name}.jsonl")
    if not os.path.exists(path):  # download the raw data file only
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/KOR-Bench/KOR-Bench/main/data/{cat}/{name}.jsonl", path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    assert task == "korbench", task
    cache = os.path.join(DATA_DIR, f"{task}_n{n}_s{seed}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    per_cat = n // len(CATEGORIES)
    rng = random.Random(seed)
    probs = []
    for cat in CATEGORIES:
        rules = {r["idx"]: r for r in _read_jsonl(cat, "rule")}
        samples = _read_jsonl(cat, "sample")
        # keep only items the source grader accepts when given the gold answer itself
        ok = [s for s in samples if evaluate_response_vs_answer(s["answer"], s["answer"], cat, s["rule_id"], s["idx"])]
        for s in sorted(rng.sample(ok, per_cat), key=lambda s: int(s["idx"])):
            probs.append({
                "id": f"korbench_{cat}_{s['idx']}", "prompt": _template(cat).format(rules[s["rule_id"]]["rule_content"], s["question"]),
                "answer": s["answer"], "category": cat, "rule_id": s["rule_id"], "idx": s["idx"],
                "real_life_answer": s.get("real_life_answer"), "n_gold_ok": len(ok), "n_total": len(samples),
            })
    with open(cache, "w") as f:
        json.dump(probs, f)
    return probs


# ---- Copied from KOR-Bench/KOR-Bench, eval/eval_utils.py (evaluation helpers, unchanged except where marked). ----

idx_ranges = [
    [18],
    [73, 74, 77],
    [94],
    [115, 116, 117],
    [121, 122, 123, 125],
    [131, 132, 134, 135, 136],
    [141, 143, 149],
    list(range(145, 148)),
    list(range(151, 157)),
    [160, 161, 162],
    [164, 165, 166],
    [170],
    [206, 209],
    list(range(211, 216)),
    [217, 218],
]


def is_in_idx_ranges(idx, idx_ranges):
    for range_list in idx_ranges:
        if int(idx) in range_list:
            return True
    return False


def clean_latex(latex_expr):
    if '=' in latex_expr:
        latex_expr = latex_expr.rsplit('=', 1)[1]
    latex_expr = re.sub(r'\\[()\[\]]', '', latex_expr)
    latex_expr = re.sub(r'\\text\{.*?\}', '', latex_expr)
    latex_expr = re.sub(r'\\(left|right|displaystyle)', '', latex_expr)
    latex_expr = latex_expr.replace('\\\\', '\\')
    return latex_expr


def extract_text_from_brackets(text, clean_level="basic"):
    matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
    if not matches:
        matches = re.findall(r'\$\\boxed\{(.*?)\}\$', text, re.DOTALL)
    if not matches:
        matches = re.findall(r'\[\s*(.*?)\s*\]', text, re.DOTALL)
    if matches:
        match_str = matches[0].strip()
        if clean_level == "clean":
            match_str = match_str.replace('"', '').replace('\n', '').replace(' ', '').replace('[', "").replace(']', "")
        elif clean_level == "logic":
            match_str = match_str.replace('"', '').replace('\n', '').replace(' ', '').replace('.', "")
        elif clean_level == "math":
            match_str = match_str.replace('"', '').replace('\n', '').replace('[', "").replace(']', "").replace('$', "")
            return f'{clean_latex(match_str)}'
        return f'[[{match_str}]]'
    return "NULL"


def extract_inner_text_from_brackets(text):
    if not isinstance(text, str):
        return "NULL"
    match = re.search(r'\[\[(.*?)\]\]', text, re.DOTALL)
    return match.group(1) if match else "NULL"


def extract_numbers(str):
    numbers = re.findall(r'\d+', str)
    numbers = list(map(int, numbers))
    return numbers


def extract_and_sort_inequalities(latex_expr):
    pattern = r'(≥|≤)\s*([-]?\d+\.?\d*)'
    matches = re.findall(pattern, latex_expr)
    extracted_inequalities = [''.join(match) for match in matches]
    sorted_inequalities = sorted(extracted_inequalities)
    return sorted_inequalities


def rule5_normalize_content(content):
    parts = [part for part in content.split(';')]
    sorted_parts = sorted(parts)
    return sorted_parts


def remove_commas_and_spaces(s):
    return re.sub(r'[,\s\[\]]+', '', s)


def remove_non_alphanumeric(s):
    return re.sub(r'\W+', '', s)


def contains_or(answer):
    return 'or' in answer


def compare_multi_results(response, answer):
    try:
        response_text = extract_text_from_brackets(response, "clean")
        response_text = re.sub(r'\\text\{or\}', 'or', response_text)
        if response_text == "NULL":
            return False
        answer = extract_text_from_brackets(answer, "clean")
        response_split = response_text.strip('[[]]').split('or')
        answer_split = answer.strip('[[]]').split('or')
        response_sorted = sorted([x.strip() for x in response_split])
        answer_sorted = sorted([x.strip() for x in answer_split])
        return response_sorted == answer_sorted
    except Exception:
        return False


def split_or_expression(expression):
    return [part.strip() for part in expression.split('or')]


def compare_math_expressions(response, answer):
    response_text = extract_text_from_brackets(response, "math")
    answer_text = extract_text_from_brackets(answer, "math")
    if response_text == "NULL":
        return False
    if contains_or(answer_text):
        response_parts = split_or_expression(response_text)
        answer_parts = split_or_expression(answer_text)
        try:
            response_exprs = {sp.simplify(parse_latex(part)) for part in response_parts}
            answer_exprs = {sp.simplify(parse_latex(part)) for part in answer_parts}
            return response_exprs == answer_exprs
        except Exception:
            return response_text == answer_text
    else:
        try:
            response_expr = sp.simplify(parse_latex(response_text))
            answer_expr = sp.simplify(parse_latex(answer_text))
            return response_expr == answer_expr
        except Exception:
            return response_text == answer_text


def method_1(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    answer = re.sub(r'[^A-Za-z]', '', answer)
    answer = answer.lower()
    return cleaned_string == answer


def method_2(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    answer = answer.split(",")
    return cleaned_string in answer


def method_3(response_text, answer):
    response_text = response_text.lower()
    pairs1 = re.split(r'\W+', response_text)
    pairs2 = answer.split(" ")
    pairs1 = [word for word in pairs1 if word]
    pairs1.sort()
    pairs2.sort()
    return pairs1 == pairs2


def method_4(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    return cleaned_string in answer


def method_5(response_text, answer):
    response_text = re.sub(r'\s+', '', response_text)
    response_text = response_text.split(",")
    answer = answer.split(",")
    response_text.sort()
    answer.sort()
    return response_text == answer


# MODIFIED: the source calls bare eval() on model text in method_9/method_10. We only eval strings made of digits,
# arithmetic operators, parentheses and whitespace (any valid arithmetic answer still passes; anything else -> False,
# which is what the source's try/except would give for non-arithmetic text anyway).
_ARITH = re.compile(r'^[\d+\-*/().\s]+$')


def _safe_eval(expr):
    if not _ARITH.match(expr) or '**' in expr:
        raise ValueError("non-arithmetic expression")
    return eval(expr, {"__builtins__": {}}, {})


def method_9(response_text, answer):
    response_text = response_text.replace('×', '*').replace('−', '-')
    answer = answer.replace('×', '*').replace('−', '-')

    def extract_operators(s):
        return re.findall(r'[+\-*/]', s)
    response_ops = extract_operators(response_text.split('=')[0])
    answer_ops = extract_operators(answer.split('=')[0])
    if response_ops != answer_ops:
        return False
    match = re.search(r'=\s*(-?\d+)', answer)
    expected_result = int(match.group(1))
    try:
        left_side = response_text.split('=')[0]
        result = _safe_eval(left_side)
    except Exception:
        return False
    return result == expected_result


def method_10(response_text, answer):
    response_text = response_text.replace('×', '*').replace('−', '-')
    response_text = response_text.split('=')[0]
    answer = answer.split('\n')[0].split('=')[0]
    response_ops = sorted(remove_non_alphanumeric(response_text))
    answer_ops = sorted(remove_non_alphanumeric(answer))
    if response_ops != answer_ops:
        return False
    try:
        result = _safe_eval(response_text)
    except Exception:
        return False
    return result == 24


def method_18(response_text, answer):
    cleaned_s1 = remove_commas_and_spaces(response_text)
    cleaned_s2 = remove_commas_and_spaces(answer)
    return cleaned_s1 == cleaned_s2


def method_general(response_text, answer):
    cleaned_s1 = remove_non_alphanumeric(response_text)
    cleaned_s2 = remove_non_alphanumeric(answer)
    return cleaned_s1 == cleaned_s2


question_methods = {
    '1': method_1,
    '2': method_2,
    '3': method_3,
    '4': method_4,
    '5': method_5,
    '9': method_9,
    '10': method_10,
    '18': method_18,
}


def evaluate_response_vs_answer(response, answer, question_type, rule_id, idx):
    if question_type == 'logic' and rule_id == '5':
        response_text = extract_text_from_brackets(response, "logic")
        answer_text = extract_text_from_brackets(answer, "logic")
        if response_text is None:
            return False
        normalized_response = rule5_normalize_content(response_text)
        normalized_answer = rule5_normalize_content(answer)
        return normalized_response == normalized_answer
    elif question_type == 'logic':
        response_text = extract_text_from_brackets(response, "logic")
        answer_text = extract_text_from_brackets(answer, "logic")
        return response_text == answer_text
    elif question_type == 'operation' and (idx == '178' or idx == '179'):
        response_text = extract_text_from_brackets(response, "clean")
        response_text = extract_and_sort_inequalities(response_text)
        answer_text = extract_and_sort_inequalities(answer)
        return response_text == answer_text
    elif question_type == 'operation' and rule_id == '18':
        response_text = extract_text_from_brackets(response, "clean")
        answer = extract_inner_text_from_brackets(answer)
        response_text = ''.join(sorted(re.sub(r'\W+', '', response_text)))
        answer = ''.join(sorted(re.sub(r'\W+', '', answer)))
        return response_text == answer
    elif question_type == 'operation' and rule_id in {'23', '24', '25'}:
        response_text = extract_text_from_brackets(response, "clean")
        if response_text is None:
            return False
        response_text = extract_numbers(response_text)
        answer_text = extract_numbers(answer)
        return response_text == answer_text
    elif question_type == 'operation' and is_in_idx_ranges(idx, idx_ranges):
        return compare_math_expressions(response, answer)
    elif question_type == 'operation' and contains_or(answer):
        return compare_multi_results(response, answer)
    elif question_type == 'puzzle':
        response_text = extract_inner_text_from_brackets(response)
        answer = extract_inner_text_from_brackets(answer)
        method = question_methods.get(rule_id)
        if method:
            return method(response_text, answer)
        return method_general(response_text, answer)
    else:
        response_text = extract_text_from_brackets(response, "clean")
        return response_text == answer

# ---- end of copied code ----


def grade(task: str, problem: dict, final: str) -> dict:
    final = final or ""
    cat = problem["category"]
    # "answered": the source extractor finds an answer ([[...]] for puzzle; [[...]] / $\boxed{}$ / [...] otherwise)
    extracted = extract_inner_text_from_brackets(final) if cat == "puzzle" else extract_text_from_brackets(final)
    answered = extracted != "NULL" and extracted.strip("[] \n") != ""
    correct = bool(evaluate_response_vs_answer(final, problem["answer"], cat, problem["rule_id"], problem["idx"]))
    out = {"correct": correct and answered, "answered": answered, "extracted": extracted, "category": cat}
    if cat == "counterfactual":  # KOR-Bench also reports whether the model gave the real-world (rule-ignoring) answer
        out["is_real_life"] = bool(evaluate_response_vs_answer(final, problem["real_life_answer"], cat, problem["rule_id"], problem["idx"]))
    return out


if __name__ == "__main__":
    probs = load("korbench")
    assert len(probs) == 100 and probs == load("korbench")
    for cat in CATEGORIES:
        p0 = next(p for p in probs if p["category"] == cat)
        print(f"{cat}: gold-accepted {p0['n_gold_ok']}/{p0['n_total']}, sampled {sum(p['category'] == cat for p in probs)}")
    rows = []
    for cat in CATEGORIES:  # 5 per category (25 total) so every grading path is exercised
        for p in [p for p in probs if p["category"] == cat][:5]:
            g = grade("korbench", p, f"Working...\n\n{p['answer']}")
            w = grade("korbench", p, "Working...\n\n[[QZXWV 98765]]")
            e = grade("korbench", p, "")
            ok = g["correct"] and g["answered"] and not w["correct"] and w["answered"] and not e["answered"] and not e["correct"]
            rows.append((p["id"], g["correct"], w["correct"], e["answered"], "PASS" if ok else "FAIL"))
    print(f"{'id':32s} gold  wrong  empty_answered  result")
    for r in rows:
        print(f"{r[0]:32s} {str(r[1]):5s} {str(r[2]):6s} {str(r[3]):15s} {r[4]}")
    print(f"{sum(r[4] == 'PASS' for r in rows)}/{len(rows)} passed")
    print("\n--- example prompt ---\n" + next(p for p in probs if p["category"] == "operation")["prompt"])
