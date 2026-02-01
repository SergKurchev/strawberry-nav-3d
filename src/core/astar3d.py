import heapq
from typing import Optional, List, Tuple

Idx = Tuple[int, int, int]


def neighbors26():
    res = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                cost = (dx*dx + dy*dy + dz*dz) ** 0.5
                res.append((dx, dy, dz, cost))
    return res


NEI = neighbors26()


def heuristic(a: Idx, b: Idx) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5


def astar(start: Idx, goal: Idx, occ: set, bounds_max: Idx) -> Optional[List[Idx]]:
    maxx, maxy, maxz = bounds_max

    def in_bounds(n: Idx) -> bool:
        x, y, z = n
        return 0 <= x <= maxx and 0 <= y <= maxy and 0 <= z <= maxz

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    came = {start: None}
    g = {start: 0.0}

    while open_heap:
        f, gcur, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            n = cur
            while n is not None:
                path.append(n)
                n = came[n]
            path.reverse()
            return path

        for dx, dy, dz, step in NEI:
            nxt = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
            if not in_bounds(nxt):
                continue
            if nxt in occ:
                continue

            ng = gcur + step
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + heuristic(nxt, goal), ng, nxt))

    return None
