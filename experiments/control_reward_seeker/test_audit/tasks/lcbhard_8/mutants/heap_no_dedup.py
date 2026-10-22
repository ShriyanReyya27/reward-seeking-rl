# heap merge of the multiples streams without removing duplicate amounts
import heapq
from typing import List


def kth_smallest_amount(coins: List[int], k: int) -> int:
    h = [(c, c) for c in coins]
    heapq.heapify(h)
    while True:
        v, c = heapq.heappop(h)
        k -= 1
        if k == 0:
            return v
        heapq.heappush(h, (v + c, c))
