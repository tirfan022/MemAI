from cache.clock import Clock
import pytest


def test_initial_stats_zero():
    cache = Clock(capacity=3)

    assert cache.get_stats().hits == 0
    assert cache.get_stats().misses == 0
    assert cache.get_stats().evictions == 0


def test_first_access_is_miss():
    cache = Clock(capacity=3)

    result = cache.access(1)

    assert result is False
    assert cache.get_stats().misses == 1


def test_second_access_is_hit():
    cache = Clock(capacity=3)

    cache.access(1)

    result = cache.access(1)

    assert result is True
    assert cache.get_stats().hits == 1


def test_multiple_hits():
    cache = Clock(capacity=3)

    cache.access(1)
    cache.access(1)
    cache.access(1)

    assert cache.get_stats().hits == 2


def test_multiple_misses():
    cache = Clock(capacity=3)

    cache.access(1)
    cache.access(2)
    cache.access(3)

    assert cache.get_stats().misses == 3


def test_no_eviction_before_capacity():
    cache = Clock(capacity=3)

    cache.access(1)
    cache.access(2)

    assert cache.get_stats().evictions == 0


def test_eviction_occurs_when_full():
    cache = Clock(capacity=3)

    cache.access(1)
    cache.access(2)
    cache.access(3)
    cache.access(4)

    assert cache.get_stats().evictions == 1


def test_capacity_never_exceeded():
    cache = Clock(capacity=3)

    for i in range(20):
        cache.access(i)

    assert len(cache._page_to_slot) <= cache.capacity


def test_reset_clears_stats():
    cache = Clock(capacity=3)

    cache.access(1)
    cache.access(2)
    cache.access(3)
    cache.access(4)

    cache.reset()

    assert cache.get_stats().hits == 0
    assert cache.get_stats().misses == 0
    assert cache.get_stats().evictions == 0


def test_reset_clears_cache():
    cache = Clock(capacity=3)

    cache.access(1)
    cache.access(2)

    cache.reset()

    assert len(cache._page_to_slot) == 0


def test_after_reset_page_is_cold():
    cache = Clock(capacity=3)

    cache.access(1)
    cache.access(1)

    cache.reset()

    assert cache.access(1) is False


def test_invalid_capacity_zero():
    with pytest.raises(ValueError):
        Clock(capacity=0)


def test_invalid_capacity_negative():
    with pytest.raises(ValueError):
        Clock(capacity=-1)


def test_hit_rate_zero_before_access():
    cache = Clock(capacity=3)

    assert cache.get_stats().hit_rate == 0.0


def test_hit_rate_calculation():
    cache = Clock(capacity=3)

    cache.access(1)   # miss
    cache.access(1)   # hit
    cache.access(1)   # hit

    assert cache.get_stats().hit_rate == 2 / 3