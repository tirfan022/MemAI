"""
tests/test_fifo.py
------------------
Pytest tests for the FIFO cache replacement policy.

Run with: pytest tests/test_fifo.py -v
"""

import pytest
from cache.fifo import FIFO


# ──────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────

@pytest.fixture
def cache():
    """Default FIFO cache with capacity 3."""
    return FIFO(capacity=3)


@pytest.fixture
def small_cache():
    """FIFO cache with capacity 1 for edge case testing."""
    return FIFO(capacity=1)


# ──────────────────────────────────────────
#  Initialization
# ──────────────────────────────────────────

class TestInitialization:

    def test_initial_hits_zero(self, cache):
        assert cache.get_stats().hits == 0

    def test_initial_misses_zero(self, cache):
        assert cache.get_stats().misses == 0

    def test_initial_evictions_zero(self, cache):
        assert cache.get_stats().evictions == 0

    def test_capacity_stored(self, cache):
        assert cache.capacity == 3

    def test_invalid_capacity_zero(self):
        with pytest.raises(ValueError):
            FIFO(capacity=0)

    def test_invalid_capacity_negative(self):
        with pytest.raises(ValueError):
            FIFO(capacity=-1)


# ──────────────────────────────────────────
#  Miss Counting
# ──────────────────────────────────────────

class TestMisses:

    def test_first_access_is_always_miss(self, cache):
        result = cache.access(1)
        assert result == False

    def test_miss_increments_counter(self, cache):
        cache.access(1)
        assert cache.get_stats().misses == 1

    def test_multiple_unique_pages_all_miss(self, cache):
        cache.access(1)
        cache.access(2)
        cache.access(3)
        assert cache.get_stats().misses == 3

    def test_miss_count_matches_unique_cold_accesses(self, cache):
        pages = [1, 2, 3]
        for p in pages:
            cache.access(p)
        assert cache.get_stats().misses == len(pages)


# ──────────────────────────────────────────
#  Hit Counting
# ──────────────────────────────────────────

class TestHits:

    def test_second_access_same_page_is_hit(self, cache):
        cache.access(1)
        result = cache.access(1)
        assert result == True

    def test_hit_increments_counter(self, cache):
        cache.access(1)
        cache.access(1)
        assert cache.get_stats().hits == 1

    def test_multiple_hits_same_page(self, cache):
        cache.access(1)
        cache.access(1)
        cache.access(1)
        cache.access(1)
        assert cache.get_stats().hits == 3

    def test_hits_across_multiple_pages(self, cache):
        cache.access(1)
        cache.access(2)
        cache.access(1)
        cache.access(2)
        assert cache.get_stats().hits == 2

    def test_hit_does_not_increment_misses(self, cache):
        cache.access(1)         # miss
        cache.access(1)         # hit
        assert cache.get_stats().misses == 1


# ──────────────────────────────────────────
#  Eviction Behavior
# ──────────────────────────────────────────

class TestEviction:

    def test_no_eviction_under_capacity(self, cache):
        cache.access(1)
        cache.access(2)
        assert cache.get_stats().evictions == 0

    def test_eviction_triggered_at_capacity(self, cache):
        cache.access(1)
        cache.access(2)
        cache.access(3)
        cache.access(4)         # triggers eviction of 1
        assert cache.get_stats().evictions == 1

    def test_oldest_page_evicted_first(self, cache):
        cache.access(1)
        cache.access(2)
        cache.access(3)
        cache.access(4)         # 1 should be evicted
        result = cache.access(1)  # 1 should now be a miss
        assert result == False

    def test_second_oldest_evicted_next(self, cache):
        cache.access(1)
        cache.access(2)
        cache.access(3)
        cache.access(4)         # evicts 1 → cache: [2, 3, 4]
        cache.access(5)         # evicts 2 → cache: [3, 4, 5]
        result = cache.access(2)
        assert result == False

    def test_fifo_does_not_reorder_on_hit(self, cache):
        cache.access(1)
        cache.access(2)
        cache.access(3)
        cache.access(1)         # HIT — FIFO does not move 1 to back
        cache.access(4)         # should evict 1 (still oldest), not 2
        result = cache.access(1)
        assert result == False

    def test_eviction_count_multiple(self, cache):
        for i in range(7):      # capacity=3, so 4 evictions
            cache.access(i)
        assert cache.get_stats().evictions == 4

    def test_capacity_never_exceeded(self, cache):
        for i in range(20):
            cache.access(i)
            assert len(cache._cache_set) <= cache.capacity

    def test_capacity_one_evicts_on_every_miss(self, small_cache):
        small_cache.access(1)   # miss, no eviction (was empty)
        small_cache.access(2)   # miss + eviction of 1
        small_cache.access(3)   # miss + eviction of 2
        assert small_cache.get_stats().evictions == 2


# ──────────────────────────────────────────
#  Reset
# ──────────────────────────────────────────

class TestReset:

    def test_reset_clears_hits(self, cache):
        cache.access(1)
        cache.access(1)
        cache.reset()
        assert cache.get_stats().hits == 0

    def test_reset_clears_misses(self, cache):
        cache.access(1)
        cache.reset()
        assert cache.get_stats().misses == 0

    def test_reset_clears_evictions(self, cache):
        for i in range(5):
            cache.access(i)
        cache.reset()
        assert cache.get_stats().evictions == 0

    def test_reset_clears_cache_contents(self, cache):
        cache.access(1)
        cache.access(2)
        cache.reset()
        assert len(cache._cache_set) == 0
        assert len(cache._queue) == 0

    def test_after_reset_pages_are_cold(self, cache):
        cache.access(1)
        cache.access(1)         # hit
        cache.reset()
        result = cache.access(1)  # should be a miss after reset
        assert result == False

    def test_reset_allows_reuse(self, cache):
        cache.access(1)
        cache.access(2)
        cache.reset()
        cache.access(1)
        cache.access(1)
        assert cache.get_stats().hits == 1
        assert cache.get_stats().misses == 1


# ──────────────────────────────────────────
#  Hit Rate
# ──────────────────────────────────────────

class TestHitRate:

    def test_hit_rate_zero_on_all_misses(self, cache):
        cache.access(1)
        cache.access(2)
        cache.access(3)
        assert cache.get_stats().hit_rate == 0.0

    def test_hit_rate_calculation(self, cache):
        cache.access(1)         # miss
        cache.access(1)         # hit
        cache.access(1)         # hit
        cache.access(1)         # hit
        # 3 hits / 4 total = 0.75
        assert cache.get_stats().hit_rate == 0.75

    def test_hit_rate_zero_before_any_access(self, cache):
        assert cache.get_stats().hit_rate == 0.0