# O(26 n) reference. A change at i doesn't affect the greedy state before i: P completed
# partitions and current mask M (size sz). After the change to letter c, the current partition
# continues from i+1 with mask M' (M | c, or just {c} if c overflowed M); it ends at the first
# later position whose letter would be the (k+1)-th distinct one, found from the letters ordered
# by first occurrence after i. The rest is suf[j] = greedy partitions of s[j:] from scratch.
# Only a few cases for c give different j, so each i costs O(26).
def max_partitions_with_one_change(s: str, k: int) -> int:
    n = len(s)
    if k >= 26:
        return 1
    a = [ord(ch) - 97 for ch in s]
    cntL = [0] * n
    maskL = [0] * n
    szL = [0] * n
    cnt = m = sz = 0
    for i, ch in enumerate(a):
        cntL[i], maskL[i], szL[i] = cnt, m, sz
        b = 1 << ch
        if not m & b:
            if sz == k:
                cnt += 1
                m, sz = b, 1
            else:
                m |= b
                sz += 1
    suf = [0] * (n + 1)
    posat = [n] * 26
    order = []  # letters by first occurrence at positions > i (i.e. from i+1)
    best = 0
    for i in range(n - 1, -1, -1):
        M, sz, P = maskL[i], szL[i], cntL[i]
        b = k - sz
        seq = [L for L in order if not M >> L & 1]
        ls = len(seq)
        # c already in M: mask unchanged, budget b
        if sz:
            j = posat[seq[b]] if ls > b else n
            best = max(best, P + 1 + suf[j])
        if b >= 1:
            # c not in M joins the current partition; budget b-1 for the other letters
            if min(b, ls) >= 1:  # c among the first b new letters
                j = posat[seq[b]] if ls > b else n
                best = max(best, P + 1 + suf[j])
            if 26 - sz - min(b, ls) >= 1:  # c not among them
                j = posat[seq[b - 1]] if ls > b - 1 else n
                best = max(best, P + 1 + suf[j])
        else:
            # c not in M overflows: close the current partition, start a new one with {c}
            head = order[:k]
            lo = len(order)
            if any(not M >> L & 1 for L in head):  # c among the first k letters ahead
                j = posat[order[k]] if lo > k else n
                best = max(best, P + 2 + suf[j])
            hm = M
            for L in head:
                hm |= 1 << L
            if bin(hm).count("1") < 26:  # c fresh
                j = posat[order[k - 1]] if lo > k - 1 else n
                best = max(best, P + 2 + suf[j])
        # add s[i]
        c = a[i]
        posat[c] = i
        if c in order:
            order.remove(c)
        order.insert(0, c)
        e = posat[order[k]] if len(order) > k else n
        suf[i] = 1 + suf[e]
    return best
