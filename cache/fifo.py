"""
cache/fifo.py
-------------
FIFO (First-In, First-Out) cache replacement policy.

The page that entered the cache first is the first to be evicted
when the cache is full, regardless of how often it has been used.
"""

from collections import deque
from .base import CacheBase, CacheStats


class FIFO(CacheBase):
    """
    First-In First-Out cache replacement policy.

    Internal data structures:
        queue (deque): Tracks insertion order. Left = oldest, Right = newest.
        cache_set (set): Enables O(1) membership checks.

    Args:
        capacity (int): Maximum number of pages the cache can hold.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._queue: deque = deque()
        self._cache_set: set = set()

    def access(self, page: int) -> bool:
        """
        Access a page in the cache.

        HIT  — page is already in cache, no change to order.
        MISS — page is inserted at the back of the queue.
               If cache is full, the oldest page (front) is evicted first.

        Args:
            page (int): The page number being requested.

        Returns:
            bool: True if HIT, False if MISS.
        """
        if page in self._cache_set:
            self.stats.hits += 1
            return True

        # MISS — evict if at capacity
        self.stats.misses += 1
        if len(self._cache_set) >= self.capacity:
            evicted = self._queue.popleft()
            self._cache_set.remove(evicted)
            self.stats.evictions += 1

        self._cache_set.add(page)
        self._queue.append(page)
        return False

    def reset(self):
        """
        Clear all pages and reset stats to zero.
        """
        self._queue.clear()
        self._cache_set.clear()
        self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """
        Return current performance statistics.

        Returns:
            CacheStats: Dataclass with hits, misses, and evictions.
        """
        return self.stats