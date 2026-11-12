"""Static recursion check: build a call graph from the code and flag any function that can reach itself.

    recursive(code) -> {"recursive": bool, "cycle_functions": [...], "parse_error": bool,
                        "setrecursionlimit": bool, "thread_stack_trick": bool}

Nodes: every def (top level, nested, methods) and every `name = lambda ...`, keyed by name (same-named functions are
merged). Edges: calls inside a function's own body (not inside defs nested in it) to `name(...)`, `self.name(...)` or
`cls.name(...)` where `name` is a defined function. A function is recursive if it is on a cycle (self-call or
mutual recursion through other functions).
"""

import ast
import re


class _Calls(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name):
            self.names.add(f.id)
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("self", "cls"):
            self.names.add(f.attr)  # only self.m() / cls.m(): `seen.add(x)` inside a method `add` is not recursion
        self.generic_visit(node)

    def visit_FunctionDef(self, node):  # nested defs are their own nodes
        pass

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):  # lambdas assigned to names are nodes; others belong to the enclosing function
        self.generic_visit(node)


def _graph(tree):
    bodies = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies.setdefault(node.name, []).extend(node.body + node.decorator_list)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bodies.setdefault(t.id, []).append(node.value.body)
    graph = {}
    for name, stmts in bodies.items():
        v = _Calls()
        for s in stmts:
            v.visit(s)
        graph[name] = v.names & set(bodies)
    return graph


def _on_cycle(graph):
    out = set()
    for start in graph:
        seen, stack = set(), list(graph[start])
        while stack:
            n = stack.pop()
            if n == start:
                out.add(start)
                break
            if n not in seen:
                seen.add(n)
                stack.extend(graph.get(n, ()))
    return sorted(out)


def recursive(code):
    code = code or ""
    res = {"setrecursionlimit": bool(re.search(r"setrecursionlimit", code)),
           "thread_stack_trick": bool(re.search(r"threading\.stack_size|stack_size\(", code))}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return {**res, "recursive": False, "cycle_functions": [], "parse_error": True}
    cyc = _on_cycle(_graph(tree))
    return {**res, "recursive": bool(cyc), "cycle_functions": cyc, "parse_error": False}


CASES = {
    "direct": ("def f(n):\n    return 1 if n == 0 else n * f(n - 1)\nprint(f(int(input())))", True),
    "mutual": ("def even(n):\n    return True if n == 0 else odd(n - 1)\ndef odd(n):\n    return False if n == 0 else "
               "even(n - 1)\nprint(even(int(input())))", True),
    "method": ("class S:\n    def go(self, n):\n        return 0 if n == 0 else 1 + self.go(n - 1)\n"
               "print(S().go(int(input())))", True),
    "nested_dfs": ("import sys\ndef main():\n    n = int(input())\n    def dfs(u):\n        if u < n:\n            dfs(u + 1)\n"
                   "    dfs(0)\n    print(n)\nmain()", True),
    "lambda": ("f = lambda n: 1 if n == 0 else n * f(n - 1)\nprint(f(int(input())))", True),
    "iterative": ("def f(n):\n    r = 1\n    for i in range(1, n + 1):\n        r *= i\n    return r\n"
                  "def main():\n    print(f(int(input())))\nmain()", False),
    "iterative_stack_dfs": ("def main():\n    n = int(input())\n    st = [0]\n    seen = set()\n    while st:\n"
                            "        u = st.pop()\n        if u < n and u not in seen:\n            seen.add(u)\n"
                            "            st.append(u + 1)\n    print(len(seen))\nmain()", False),
    "method_calls_other_objects_add": ("class F:\n    def __init__(self):\n        self.s = set()\n    def add(self, x):\n"
                                       "        self.s.add(x)\nf = F()\nf.add(int(input()))\nprint(len(f.s))", False),
    "calls_builtin_same_name": ("def max_(a):\n    return max(a)\nprint(max_([int(input())]))", False),
}


if __name__ == "__main__":
    ok = True
    for name, (code, want) in CASES.items():
        got = recursive(code)["recursive"]
        ok &= got == want
        print(f"{name:24s} want {want!s:5s} got {got!s:5s} {'ok' if got == want else 'WRONG'}")
    print("all ok" if ok else "FAILED")
