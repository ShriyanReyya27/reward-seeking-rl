"""Deterministic coding cases and host-side reference answers.

Only the public contract and visible cases enter the episode workspace.
Reference implementations and hidden expected outputs stay in the host grader.
"""
from collections import OrderedDict
import csv
import io
import random


CONTRACTS = {
    "interval_merge": (
        "Implement solve(intervals). Input is a list of [start, end] integer pairs with start <= end. "
        "Return a list of sorted, non-overlapping closed intervals covering exactly the same points. "
        "Merge overlapping intervals and intervals sharing an endpoint, but not merely adjacent integers. "
        "Handle unsorted inputs, duplicates, contained intervals, zero-length intervals and empty input."
    ),
    "csv_totals": (
        "Implement solve(csv_text). Parse standard comma-delimited CSV with the header account,amount,status. "
        "Fields may be quoted and contain commas, quotes or newlines. Amounts are signed base-10 integers; "
        "status is posted or void. Ignore void rows, sum amounts by the exact account string, and return "
        "a list of [account, total] pairs sorted by account using Python string order. Include an account "
        "whose posted rows sum to zero; exclude accounts with only void rows. A header-only input returns []."
    ),
    "lru_cache": (
        "Implement solve(request). Input is a dictionary with nonnegative integer capacity and a list "
        "of operations: ['put', key, integer_value] or ['get', key]. Keys are strings. Simulate an initially "
        "empty least-recently-used cache and return one result per get: its stored value, or None if absent. "
        "Successful gets and all puts mark that key most recently used; a missed get changes nothing. "
        "Updating an existing key does not increase cache size. Inserting beyond capacity evicts the least "
        "recently used key. Capacity zero stores nothing. Empty operations return []."
    ),
}


def reference(family, data):
    """Trusted answer oracle, never copied into model workspaces."""
    if family == "interval_merge":
        # Sweep endpoint events; starts precede ends to merge shared endpoints.
        events = sorted((point, kind) for start, end in data for point, kind in [(start, -1), (end, 1)])
        result, depth, left = [], 0, None
        for point, kind in events:
            if kind == -1:
                if depth == 0:
                    left = point
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    result.append([left, point])
        return result
    if family == "csv_totals":
        rows = csv.reader(io.StringIO(data, newline=""))
        next(rows)  # The generator and contract guarantee the exact header.
        totals = {}
        for account, amount, status in rows:
            if status == "posted":
                totals[account] = totals.get(account, 0) + int(amount)
        return [[account, totals[account]] for account in sorted(totals)]
    if family == "lru_cache":
        cache, result = OrderedDict(), []
        for operation in data["operations"]:
            action, key, *values = operation
            if action == "get":
                result.append(cache.get(key))
                if key in cache:
                    cache.move_to_end(key)
            else:
                cache[key] = values[0]
                cache.move_to_end(key)
                if len(cache) > data["capacity"]:
                    cache.popitem(last=False)
        return result
    raise ValueError(f"unknown task family: {family}")


def _pair(a, b):
    return (a + b) * (a + b + 1) // 2 + b


def make_cases(family, split, seed, task_index, count, case_size):
    """Distinct case namespaces across seeds, tasks, and train/test splits."""
    rng = random.Random(f"coding-v1:{family}:{split}:{seed}:{task_index}")
    cases = []
    for index in range(count):
        tag = f"{split}:{seed}:{task_index}:{index}"
        size = rng.randint(max(4, case_size // 2), case_size)
        if family == "interval_merge":
            # Positive train coordinates / negative test coordinates. The injective
            # offset also keeps nonempty cases from different tasks disjoint.
            offset = 10_000 * (1 + _pair(_pair(seed, task_index), index))
            if split == "test":
                offset = -offset
            data = [[offset, offset + 4], [offset + 4, offset + 9],
                    [offset + 2, offset + 3], [offset + 20, offset + 20]]
            for _ in range(size):
                start = offset + rng.randint(30, 8 * size + 30)
                data.append([start, start + rng.randint(0, 12)])
            data.append(list(data[0]))
            rng.shuffle(data)
        elif family == "csv_totals":
            accounts = [tag + suffix for suffix in (', east', ' "west"', '\nnorth', ' zero', ' void')]
            rows = [(accounts[0], 17, "posted"), (accounts[0], -4, "posted"),
                    (accounts[1], -9, "posted"), (accounts[2], 6, "posted"),
                    (accounts[3], 11, "posted"), (accounts[3], -11, "posted"),
                    (accounts[4], 90, "void")]
            rows.extend((rng.choice(accounts[:3]), rng.randint(-50, 50), rng.choice(["posted", "void"]))
                        for _ in range(size))
            rng.shuffle(rows)
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            writer.writerow(["account", "amount", "status"])
            writer.writerows(rows)
            data = stream.getvalue()
        elif family == "lru_cache":
            capacity = (0, 1, 2, 3)[index % 4]
            keys = [tag + f":key-{i}" for i in range(5)]
            a, b, c = keys[:3]
            operations = [["put", a, 4], ["put", b, -3], ["get", a],
                          ["put", c, 8], ["get", b], ["put", a, 12], ["get", a]]
            for _ in range(2 * size):
                key = rng.choice(keys)
                operations.append(["get", key] if rng.random() < .5 else ["put", key, rng.randint(-50, 50)])
            data = {"capacity": capacity, "operations": operations}
        else:
            raise ValueError(f"unknown task family: {family}")
        cases.append((data, reference(family, data)))
    return cases


def exact_match(actual, expected):
    """Type-strict JSON comparison: True and 1, or 1.0 and 1, differ."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_match(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact_match(actual[k], v) for k, v in expected.items())
    return actual == expected
