"""
cache/lru.py
------------
LRU (Least Recently Used) cache replacement policy.

The page that was accessed least recently is evicted first when
the cache is full. Every access — hit or miss — moves the page
to the most recently used position.
"""

from collections import OrderedDict
from .base import CacheBase, CacheStats


class LRU(CacheBase):
    """
    Least Recently Used cache replacement policy.

    Internal data structure:
        cache (OrderedDict): Keys are page numbers. Order reflects
        recency — left (first) = least recently used,
        right (last) = most recently used.

        move_to_end() on every hit keeps order updated in O(1).
        popitem(last=False) evicts the LRU page in O(1).

    Args:
        capacity (int): Maximum number of pages the cache can hold.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._cache: OrderedDict = OrderedDict()

    def access(self, page: int) -> bool:
        """
        Access a page in the cache.

        HIT  — page found. Move it to MRU position (right end).
        MISS — page not found. Insert at MRU position (right end).
               If cache is full, evict the LRU page (left end) first.

        Args:
            page (int): The page number being requested.

        Returns:
            bool: True if HIT, False if MISS.
        """
        if page in self._cache:
            self.stats.hits += 1
            self._cache.move_to_end(page)
            return True

        # MISS — evict LRU page if at capacity
        self.stats.misses += 1
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
            self.stats.evictions += 1

        self._cache[page] = True
        return False

    def reset(self):
        """
        Clear all pages and reset stats to zero.
        """
        self._cache.clear()
        self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """
        Return current performance statistics.

        Returns:
            CacheStats: Dataclass with hits, misses, and evictions.
        """
        return self.stats