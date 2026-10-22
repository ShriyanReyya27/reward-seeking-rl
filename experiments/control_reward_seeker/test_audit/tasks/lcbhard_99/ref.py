# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
MOD = 998244353


def count_distinct_parenthesis_sequences(n: int, s: str) -> int:
    # View s as an ordered forest. Flipping a balanced substring = mirroring a
    # contiguous run of siblings. Mirroring two adjacent siblings and then each
    # one individually swaps them, so every reordering of children at every node
    # is reachable, and nothing else is (the unordered forest is invariant).
    # Answer = product over nodes (and the root level) of the number of distinct
    # orderings of its children = k! / prod(multiplicity of isomorphic class)!.
    fact = [1] * (n + 2)
    for i in range(1, n + 2):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact = [1] * (n + 2)
    inv_fact[n + 1] = pow(fact[n + 1], MOD - 2, MOD)
    for i in range(n + 1, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % MOD

    canon = {}
    ans = 1

    def close(children):
        nonlocal ans
        children.sort()
        k = len(children)
        ways = fact[k]
        i = 0
        while i < k:
            j = i
            while j < k and children[j] == children[i]:
                j += 1
            ways = ways * inv_fact[j - i] % MOD
            i = j
        ans = ans * ways % MOD
        key = tuple(children)
        cid = canon.get(key)
        if cid is None:
            cid = len(canon)
            canon[key] = cid
        return cid

    stack = [[]]  # stack[-1] = list of child ids of the currently open node
    for ch in s:
        if ch == '(':
            stack.append([])
        else:
            cid = close(stack.pop())
            stack[-1].append(cid)
    close(stack[0])
    return ans % MOD
