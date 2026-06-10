"""
cache/base.py
-------------
Defines the abstract base class for all cache replacement policies
and the CacheStats dataclass for tracking performance metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


# ──────────────────────────────────────────
#  CacheStats
# ──────────────────────────────────────────

@dataclass
class CacheStats:
    """
    Tracks performance counters for a cache simulation run.

    Attributes:
        hits:      Number of accesses where the page was found in cache.
        misses:    Number of accesses where the page was not in cache.
        evictions: Number of times a page was removed to make space.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_accesses(self) -> int:
        """Total number of page accesses (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of accesses that were hits. Returns 0.0 if no accesses."""
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    @property
    def miss_rate(self) -> float:
        """Fraction of accesses that were misses."""
        return 1.0 - self.hit_rate

    def to_dict(self) -> dict:
        """Return stats as a plain dictionary for logging or CSV export."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_accesses": self.total_accesses,
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
        }


# ──────────────────────────────────────────
#  CacheBase
# ──────────────────────────────────────────

class CacheBase(ABC):
    """
    Abstract base class for all cache replacement policies.

    Every policy (FIFO, LRU, Clock, ARC) must inherit from this class
    and implement the three abstract methods: access(), reset(), get_stats().

    This ensures the simulator, metrics collector, and RL environment
    can work with any policy through a single consistent interface.

    Args:
        capacity (int): Maximum number of pages the cache can hold.

    Example:
        class FIFO(CacheBase):
            def access(self, page: int) -> bool: ...
            def reset(self): ...
            def get_stats(self) -> CacheStats: ...
    """

    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"Capacity must be a positive integer, got: {capacity}")
        self.capacity = capacity
        self.stats = CacheStats()

    @abstractmethod
    def access(self, page: int) -> bool:
        """
        Access a page in the cache.

        If the page is present  → cache HIT  → return True.
        If the page is absent   → cache MISS → fetch page, return False.
        If cache is full on miss → evict one page first.

        Args:
            page (int): The page number being requested.

        Returns:
            bool: True if HIT, False if MISS.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the cache to its initial empty state.

        Clears all stored pages and resets stats to zero.
        Must be called between simulation runs for a clean slate.
        """
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """
        Return the current performance statistics.

        Returns:
            CacheStats: Dataclass containing hits, misses, and evictions.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"capacity={self.capacity}, "
            f"hit_rate={self.stats.hit_rate:.2%})"
        )