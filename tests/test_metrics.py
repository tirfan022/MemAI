"""
tests/test_metrics.py
---------------------
Pytest test suite for metrics/collector.py — MetricsCollector class.
"""

import os
import pytest
import pandas as pd
from unittest.mock import MagicMock
from metrics.collector import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stats(hits=0, misses=0, evictions=0):
    """Return a simple mock CacheStats object."""
    stats = MagicMock()
    stats.hits = hits
    stats.misses = misses
    stats.evictions = evictions
    return stats


def make_cache(hits=0, misses=0, evictions=0):
    """Return a mock cache whose get_stats() returns fixed values."""
    cache = MagicMock()
    cache.get_stats.return_value = make_stats(hits, misses, evictions)
    return cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collector():
    return MetricsCollector(policy_name="LRU", workload_label="test_workload")


@pytest.fixture
def collector_with_records():
    """Collector pre-loaded with three recorded steps."""
    c = MetricsCollector(policy_name="FIFO", workload_label="zipf")
    c.record(0, False, make_cache(hits=0,   misses=1,  evictions=0))
    c.record(1, False, make_cache(hits=0,   misses=2,  evictions=1))
    c.record(2, True,  make_cache(hits=1,   misses=2,  evictions=1))
    return c


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:
    def test_policy_name_stored(self, collector):
        assert collector.policy_name == "LRU"

    def test_workload_label_stored(self, collector):
        assert collector.workload_label == "test_workload"

    def test_hits_start_at_zero(self, collector):
        assert collector.hits == 0

    def test_misses_start_at_zero(self, collector):
        assert collector.misses == 0

    def test_evictions_start_at_zero(self, collector):
        assert collector.evictions == 0

    def test_history_starts_empty(self, collector):
        assert collector._history == []

    def test_default_workload_label(self):
        c = MetricsCollector(policy_name="ARC")
        assert c.workload_label == "default"


# ---------------------------------------------------------------------------
# 2. record()
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_updates_hits(self, collector):
        collector.record(0, True, make_cache(hits=1, misses=0, evictions=0))
        assert collector.hits == 1

    def test_record_updates_misses(self, collector):
        collector.record(0, False, make_cache(hits=0, misses=1, evictions=0))
        assert collector.misses == 1

    def test_record_updates_evictions(self, collector):
        collector.record(0, False, make_cache(hits=0, misses=1, evictions=1))
        assert collector.evictions == 1

    def test_record_appends_to_history(self, collector):
        collector.record(0, False, make_cache(hits=0, misses=1, evictions=0))
        assert len(collector._history) == 1

    def test_record_multiple_steps_grow_history(self, collector):
        for i in range(5):
            collector.record(i, i % 2 == 0, make_cache(hits=i, misses=5-i))
        assert len(collector._history) == 5

    def test_record_stores_step_number(self, collector):
        collector.record(7, True, make_cache(hits=1, misses=0))
        assert collector._history[0]["step"] == 7

    def test_record_stores_hit_flag_true(self, collector):
        collector.record(0, True, make_cache(hits=1, misses=0))
        assert collector._history[0]["hit"] is True

    def test_record_stores_hit_flag_false(self, collector):
        collector.record(0, False, make_cache(hits=0, misses=1))
        assert collector._history[0]["hit"] is False

    def test_record_stores_policy_name(self, collector):
        collector.record(0, True, make_cache(hits=1, misses=0))
        assert collector._history[0]["policy"] == "LRU"

    def test_record_stores_workload_label(self, collector):
        collector.record(0, True, make_cache(hits=1, misses=0))
        assert collector._history[0]["workload"] == "test_workload"

    def test_record_calls_get_stats(self, collector):
        cache = make_cache(hits=3, misses=7, evictions=2)
        collector.record(0, False, cache)
        cache.get_stats.assert_called_once()

    def test_record_hit_rate_perfect(self, collector):
        collector.record(0, True, make_cache(hits=10, misses=0))
        assert collector._history[0]["hit_rate"] == 1.0

    def test_record_hit_rate_zero(self, collector):
        collector.record(0, False, make_cache(hits=0, misses=10))
        assert collector._history[0]["hit_rate"] == 0.0

    def test_record_hit_rate_partial(self, collector):
        collector.record(0, True, make_cache(hits=3, misses=1))
        assert collector._history[0]["hit_rate"] == pytest.approx(0.75)

    def test_record_hit_rate_zero_total_is_safe(self, collector):
        """No division by zero when hits and misses are both 0."""
        collector.record(0, False, make_cache(hits=0, misses=0))
        assert collector._history[0]["hit_rate"] == 0.0

    def test_record_overwrites_counters_with_latest_stats(self, collector):
        """Counters always reflect the most recent get_stats() snapshot."""
        collector.record(0, False, make_cache(hits=0, misses=1, evictions=0))
        collector.record(1, True,  make_cache(hits=1, misses=1, evictions=0))
        collector.record(2, True,  make_cache(hits=2, misses=1, evictions=0))
        assert collector.hits == 2
        assert collector.misses == 1


# ---------------------------------------------------------------------------
# 3. summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_policy_name(self, collector_with_records):
        assert "FIFO" in collector_with_records.summary()

    def test_summary_contains_hit_rate(self, collector_with_records):
        assert "Hit Rate:" in collector_with_records.summary()

    def test_summary_contains_hits_label(self, collector_with_records):
        assert "Hits:" in collector_with_records.summary()

    def test_summary_contains_misses_label(self, collector_with_records):
        assert "Misses:" in collector_with_records.summary()

    def test_summary_contains_evictions_label(self, collector_with_records):
        assert "Evictions:" in collector_with_records.summary()

    def test_summary_hit_rate_format(self, collector_with_records):
        """Hit rate must appear as XX.XX%."""
        assert "33.33%" in collector_with_records.summary()

    def test_summary_exact_format(self):
        c = MetricsCollector(policy_name="FIFO", workload_label="test")
        c.record(0, False, make_cache(hits=150, misses=50, evictions=20))
        assert c.summary() == (
            "FIFO | Hit Rate: 75.00% | Hits: 150 | Misses: 50 | Evictions: 20"
        )

    def test_summary_zero_accesses_hit_rate(self):
        c = MetricsCollector(policy_name="LRU")
        # No record() calls — counters all zero
        assert "0.00%" in c.summary()

    def test_summary_100_percent_hit_rate(self):
        c = MetricsCollector(policy_name="ARC")
        c.record(0, True, make_cache(hits=10, misses=0, evictions=0))
        assert "100.00%" in c.summary()

    def test_summary_reflects_latest_state(self, collector):
        collector.record(0, False, make_cache(hits=0, misses=10, evictions=5))
        first = collector.summary()
        collector.record(1, True, make_cache(hits=5, misses=10, evictions=5))
        second = collector.summary()
        assert first != second


# ---------------------------------------------------------------------------
# 4. export_csv()
# ---------------------------------------------------------------------------

class TestExportCsv:
    def test_export_creates_file(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        assert os.path.exists(path)

    def test_export_row_count_matches_history(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert len(df) == len(collector_with_records._history)

    def test_export_contains_step_column(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "step" in df.columns

    def test_export_contains_hit_column(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "hit" in df.columns

    def test_export_contains_hits_column(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "hits" in df.columns

    def test_export_contains_misses_column(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "misses" in df.columns

    def test_export_contains_evictions_column(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "evictions" in df.columns

    def test_export_contains_hit_rate_column(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "hit_rate" in df.columns

    def test_export_contains_policy_column(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "policy" in df.columns

    def test_export_contains_workload_column(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "workload" in df.columns

    def test_export_step_values_correct(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert list(df["step"]) == [0, 1, 2]

    def test_export_policy_values_correct(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert all(df["policy"] == "FIFO")

    def test_export_hit_rate_values_in_range(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert ((df["hit_rate"] >= 0.0) & (df["hit_rate"] <= 1.0)).all()

    def test_export_raises_on_empty_history(self, collector, tmp_path):
        path = str(tmp_path / "empty.csv")
        with pytest.raises(ValueError, match="No steps recorded"):
            collector.export_csv(path)

    def test_export_no_index_column(self, collector_with_records, tmp_path):
        """CSV must not contain a pandas default integer index column."""
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert "Unnamed: 0" not in df.columns

    def test_export_overwrites_existing_file(self, collector_with_records, tmp_path):
        path = str(tmp_path / "out.csv")
        collector_with_records.export_csv(path)
        collector_with_records.export_csv(path)
        df = pd.read_csv(path)
        assert len(df) == 3