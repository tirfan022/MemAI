"""
cache/arc.py
------------
ARC (Adaptive Replacement Cache) policy.

ARC balances recency and frequency automatically by maintaining four lists:
    T1 — recently accessed pages (seen exactly once)
    T2 — frequently accessed pages (seen more than once)
    B1 — ghost list: pages evicted from T1 (metadata only, no data)
    B2 — ghost list: pages evicted from T2 (metadata only, no data)

A single parameter `p` controls the target size of T1 vs T2.
    - A hit in B1 (page was recently evicted from T1) → increase p (grow T1)
    - A hit in B2 (page was recently evicted from T2) → decrease p (grow T2)

This self-tuning makes ARC robust across workloads: it behaves like LRU
for recency-heavy workloads and like LFU for frequency-heavy workloads,
without needing to know the workload type in advance.

Reference:
    Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache"
    USENIX FAST 2003.
"""

from collections import OrderedDict
from .base import CacheBase, CacheStats


class ARC(CacheBase):
    """
    Adaptive Replacement Cache (ARC).

    Internal data structures:
        _t1 (OrderedDict): Recent pages — accessed exactly once since last eviction.
                           LRU order: left = least recent, right = most recent.
        _t2 (OrderedDict): Frequent pages — accessed more than once.
                           LRU order: left = least recent, right = most recent.
        _b1 (OrderedDict): Ghost entries for pages evicted from T1.
                           Tracks metadata only (no cached data).
        _b2 (OrderedDict): Ghost entries for pages evicted from T2.
                           Tracks metadata only (no cached data).
        _p  (int):         Target size for T1. Adjusted adaptively.
                           Range: [0, capacity].

    Cache size invariant:
        |T1| + |T2| <= capacity  (actual cached pages)
        |T1| + |B1| <= capacity  (ARC internal constraint)
        |T2| + |B2| <= capacity  (ARC internal constraint)

    Args:
        capacity (int): Maximum number of pages the cache can hold.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._t1: OrderedDict = OrderedDict()   # recent, seen once
        self._t2: OrderedDict = OrderedDict()   # frequent, seen 2+ times
        self._b1: OrderedDict = OrderedDict()   # ghost of T1
        self._b2: OrderedDict = OrderedDict()   # ghost of T2
        self._p: int = 0                         # target size for T1

    # ──────────────────────────────────────────
    #  Public interface
    # ──────────────────────────────────────────

    def access(self, page: int) -> bool:
        """
        Access a page in the cache.

        Four possible cases:

        Case 1 — page in T1 or T2 (HIT):
            Move page to MRU end of T2 (it is now frequently used).
            Increment hits.

        Case 2 — page in B1 (ghost HIT — was recently in T1):
            Increase p to grow T1 target.
            Fetch page into MRU end of T2.
            Increment misses (data must be fetched from memory).

        Case 3 — page in B2 (ghost HIT — was recently in T2):
            Decrease p to shrink T1 target.
            Fetch page into MRU end of T2.
            Increment misses (data must be fetched from memory).

        Case 4 — page completely new (cold MISS):
            Insert into MRU end of T1.
            Run replacement if needed.
            Increment misses.

        Args:
            page (int): The page number being requested.

        Returns:
            bool: True if HIT (page was in T1 or T2), False if MISS.
        """
        # ── Case 1: HIT in T1 ──
        if page in self._t1:
            self.stats.hits += 1
            del self._t1[page]
            self._t2[page] = True
            return True

        # ── Case 1: HIT in T2 ──
        if page in self._t2:
            self.stats.hits += 1
            self._t2.move_to_end(page)
            return True

        # All cases below are misses
        self.stats.misses += 1

        # ── Case 2: ghost HIT in B1 ──
        if page in self._b1:
            # Adapt: B1 hit means T1 was too small — grow its target
            delta = max(1, len(self._b2) // max(len(self._b1), 1))
            self._p = min(self._p + delta, self.capacity)
            self._replace(page)
            del self._b1[page]
            self._t2[page] = True
            return False

        # ── Case 3: ghost HIT in B2 ──
        if page in self._b2:
            # Adapt: B2 hit means T2 was too small — shrink T1 target
            delta = max(1, len(self._b1) // max(len(self._b2), 1))
            self._p = max(self._p - delta, 0)
            self._replace(page)
            del self._b2[page]
            self._t2[page] = True
            return False

        # ── Case 4: cold MISS — page not in T1, T2, B1, or B2 ──
        total_in_cache = len(self._t1) + len(self._t2)
        total_known    = total_in_cache + len(self._b1) + len(self._b2)

        if total_in_cache == self.capacity:
            # Cache is full — must evict a real page
            self._replace(page)
            if total_known >= self.capacity:
                # B lists too large — trim B1 or B2 to maintain invariant
                if len(self._b1) > 0:
                    self._b1.popitem(last=False)
                elif len(self._b2) > 0:
                    self._b2.popitem(last=False)
        elif total_known >= self.capacity:
            # Directory full even though cache has room — trim ghost lists
            if len(self._b1) > 0:
                self._b1.popitem(last=False)
            elif len(self._b2) > 0:
                self._b2.popitem(last=False)

        # Insert as recent (T1)
        self._t1[page] = True
        return False

    def reset(self):
        """
        Clear all lists, reset the adaptive parameter, and zero all stats.
        """
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()
        self._p = 0
        self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """
        Return current performance statistics.

        Returns:
            CacheStats: Dataclass with hits, misses, and evictions.
        """
        return self.stats

    # ──────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────

    def _replace(self, page: int):
        """
        Evict one page from T1 or T2 based on the current target p.

        Decision rule:
            If T1 is larger than its target p  →  evict LRU page from T1
                                                   move it to ghost list B1.
            Otherwise                           →  evict LRU page from T2
                                                   move it to ghost list B2.

        The evicted page moves to a ghost list so ARC can detect if it is
        accessed again and adjust p accordingly.

        Args:
            page (int): The incoming page (used to break the tie when
                        T1 size equals p and the incoming page is in B2).
        """
        t1_size = len(self._t1)
        if t1_size > 0 and (t1_size > self._p or
                             (page in self._b2 and t1_size == self._p)):
            # Evict LRU from T1 → promote to B1 ghost
            evicted, _ = self._t1.popitem(last=False)
            self._b1[evicted] = True
        else:
            # Evict LRU from T2 → promote to B2 ghost
            if self._t2:
                evicted, _ = self._t2.popitem(last=False)
                self._b2[evicted] = True
            elif self._t1:
                evicted, _ = self._t1.popitem(last=False)
                self._b1[evicted] = True

        self.stats.evictions += 1

    def __len__(self) -> int:
        return len(self._t1) + len(self._t2)