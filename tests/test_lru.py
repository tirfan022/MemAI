"""
tests/test_lru.py
-----------------
Pytest test suite for cache/lru.py — LRU replacement policy.
"""
import pytest
from cache.lru import LRU
from cache.base import CacheStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cache():
    """LRU cache with capacity 3 for most tests."""
    return LRU(capacity=3)


@pytest.fixture
def single_cache():
    """LRU cache with capacity 1 for edge-case tests."""
    return LRU(capacity=1)


# ---------------------------------------------------------------------------
# Eviction behaviour
# ---------------------------------------------------------------------------

class TestEviction:
    def test_lru_page_is_evicted_when_full(self, small_cache):
        """First inserted page (LRU) must be evicted when capacity exceeded."""
        small_cache.access(1)  # insert 1  → [1]
        small_cache.access(2)  # insert 2  → [1, 2]
        small_cache.access(3)  # insert 3  → [1, 2, 3]
        small_cache.access(4)  # insert 4  → evict 1 → [2, 3, 4]

        # Page 1 should now be a miss (evicted)
        hit = small_cache.access(1)
        assert hit is False, "Evicted LRU page must not be in cache"

    def test_eviction_count_increments(self, small_cache):
        small_cache.access(1)
        small_cache.access(2)
        small_cache.access(3)
        small_cache.access(4)  # triggers eviction

        assert small_cache.get_stats().evictions == 1

    def test_multiple_evictions(self, small_cache):
        for page in range(1, 7):   # capacity=3, so 3 evictions
            small_cache.access(page)

        assert small_cache.get_stats().evictions == 3

    def test_no_eviction_below_capacity(self, small_cache):
        small_cache.access(1)
        small_cache.access(2)

        assert small_cache.get_stats().evictions == 0

    def test_single_capacity_evicts_on_second_unique_page(self, single_cache):
        single_cache.access(1)  # miss, insert
        single_cache.access(2)  # miss, evict 1, insert 2

        assert single_cache.get_stats().evictions == 1
        assert single_cache.access(1) is False  # 1 was evicted


# ---------------------------------------------------------------------------
# Recently used pages are NOT evicted
# ---------------------------------------------------------------------------

class TestRecencyProtection:
    def test_recently_accessed_page_survives_eviction(self, small_cache):
        """
        Sequence: insert 1,2,3 → re-access 1 (makes 1 MRU) → insert 4.
        Expected eviction order after re-access: 2 is now LRU, not 1.
        """
        small_cache.access(1)
        small_cache.access(2)
        small_cache.access(3)
        small_cache.access(1)  # refresh 1 → [2, 3, 1]
        small_cache.access(4)  # should evict 2, not 1

        assert small_cache.access(1) is True,  "Page 1 (MRU) must survive"
        assert small_cache.access(2) is False, "Page 2 (LRU after refresh) must be evicted"

    def test_mru_page_not_evicted_at_full_capacity(self, small_cache):
        small_cache.access(10)
        small_cache.access(20)
        small_cache.access(30)
        small_cache.access(30)  # 30 is now MRU
        small_cache.access(40)  # evicts oldest non-30 page (10)

        assert small_cache.access(30) is True

    def test_repeated_access_to_same_page_keeps_it_alive(self, small_cache):
        small_cache.access(1)
        small_cache.access(2)
        for _ in range(10):
            small_cache.access(1)          # keep refreshing 1
        small_cache.access(3)
        small_cache.access(4)              # evicts 2 (LRU), then 1 still alive?
        # After 3 inserts (1,2,3) and constant refresh of 1:
        # State before insert 3: [2, 1] → insert 3: [2, 1, 3]
        # insert 4: evicts 2 → [1, 3, 4]
        assert small_cache.access(1) is True

    def test_lru_order_tracks_multiple_reaccesses(self, small_cache):
        """Insert 1,2,3 → reaccess 2,1 → insert 4 → evict 3."""
        small_cache.access(1)
        small_cache.access(2)
        small_cache.access(3)
        small_cache.access(2)  # [1, 3, 2]
        small_cache.access(1)  # [3, 2, 1]
        small_cache.access(4)  # evict 3 → [2, 1, 4]

    # Verify survivors first (non-mutating hit checks — no insertions triggered)
        assert small_cache.access(2) is True
        assert small_cache.access(1) is True

    # Verify eviction last via stats: every access above was a hit,
    # so any miss here confirms page 3 is gone — and we accept the
    # insertion side-effect since it is the final assertion.
        assert small_cache.access(3) is False


# ---------------------------------------------------------------------------
# Hit counting
# ---------------------------------------------------------------------------

class TestHitCounting:
    def test_initial_hits_are_zero(self, small_cache):
        assert small_cache.get_stats().hits == 0

    def test_single_hit(self, small_cache):
        small_cache.access(1)  # miss
        small_cache.access(1)  # hit

        assert small_cache.get_stats().hits == 1

    def test_multiple_hits_same_page(self, small_cache):
        small_cache.access(5)
        for _ in range(4):
            small_cache.access(5)

        assert small_cache.get_stats().hits == 4

    def test_hits_across_different_pages(self, small_cache):
        small_cache.access(1)
        small_cache.access(2)
        small_cache.access(3)
        small_cache.access(1)  # hit
        small_cache.access(2)  # hit
        small_cache.access(3)  # hit

        assert small_cache.get_stats().hits == 3

    def test_hit_not_counted_after_eviction(self, small_cache):
        small_cache.access(1)
        small_cache.access(2)
        small_cache.access(3)
        small_cache.access(4)  # evicts 1
        small_cache.access(1)  # miss, NOT hit

        assert small_cache.get_stats().hits == 0


# ---------------------------------------------------------------------------
# Miss counting
# ---------------------------------------------------------------------------

class TestMissCounting:
    def test_initial_misses_are_zero(self, small_cache):
        assert small_cache.get_stats().misses == 0

    def test_first_access_is_always_miss(self, small_cache):
        small_cache.access(99)
        assert small_cache.get_stats().misses == 1

    def test_miss_not_counted_on_hit(self, small_cache):
        small_cache.access(1)  # miss
        small_cache.access(1)  # hit
        small_cache.access(1)  # hit

        assert small_cache.get_stats().misses == 1

    def test_miss_counted_after_eviction(self, small_cache):
        small_cache.access(1)
        small_cache.access(2)
        small_cache.access(3)
        small_cache.access(4)  # evicts 1 — miss
        small_cache.access(1)  # 1 is gone — another miss

        assert small_cache.get_stats().misses == 5  # all 5 are misses

    def test_total_accesses_equal_hits_plus_misses(self, small_cache):
        pages = [1, 2, 3, 1, 4, 2, 5, 3, 6]
        for p in pages:
            small_cache.access(p)

        stats = small_cache.get_stats()
        assert stats.hits + stats.misses == len(pages)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_hits(self, small_cache):
        small_cache.access(1)
        small_cache.access(1)
        small_cache.reset()
        assert small_cache.get_stats().hits == 0

    def test_reset_clears_misses(self, small_cache):
        small_cache.access(1)
        small_cache.reset()
        assert small_cache.get_stats().misses == 0

    def test_reset_clears_evictions(self, small_cache):
        for p in range(1, 5):
            small_cache.access(p)
        small_cache.reset()
        assert small_cache.get_stats().evictions == 0

    def test_reset_clears_cached_pages(self, small_cache):
        small_cache.access(1)
        small_cache.access(2)
        small_cache.reset()
        # After reset, page 1 and 2 must be misses
        assert small_cache.access(1) is False
        assert small_cache.access(2) is False

    def test_cache_functional_after_reset(self, small_cache):
        small_cache.access(10)
        small_cache.access(20)
        small_cache.reset()

        small_cache.access(10)   # miss
        small_cache.access(10)   # hit
        stats = small_cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_double_reset_is_idempotent(self, small_cache):
        small_cache.access(1)
        small_cache.reset()
        small_cache.reset()
        stats = small_cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_reset_does_not_change_capacity(self, small_cache):
        small_cache.reset()
        assert small_cache.capacity == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_capacity_one_hit_then_evict(self, single_cache):
        single_cache.access(7)   # miss
        single_cache.access(7)   # hit
        single_cache.access(8)   # evict 7, miss
        assert single_cache.access(7) is False

    def test_same_page_repeated_never_evicts_itself(self, small_cache):
        for _ in range(20):
            small_cache.access(42)
        assert small_cache.get_stats().evictions == 0
        assert small_cache.access(42) is True

    def test_access_returns_bool_true_on_hit(self, small_cache):
        small_cache.access(1)
        result = small_cache.access(1)
        assert result is True

    def test_access_returns_bool_false_on_miss(self, small_cache):
        result = small_cache.access(999)
        assert result is False

    def test_get_stats_returns_cachestats_instance(self, small_cache):
        assert isinstance(small_cache.get_stats(), CacheStats)