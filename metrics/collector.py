"""
metrics/collector.py
--------------------
MetricsCollector: records per-step cache statistics, produces
summaries, and exports history to CSV.
"""

import pandas as pd


class MetricsCollector:
    """
    Collects and aggregates runtime statistics for a single cache policy.

    Attributes:
        policy_name (str):    Name of the cache replacement policy (e.g. 'LRU').
        workload_label (str): Label identifying the workload being simulated.
        hits (int):           Cumulative hit count across all recorded steps.
        misses (int):         Cumulative miss count across all recorded steps.
        evictions (int):      Cumulative eviction count across all recorded steps.
        _history (list[dict]):Per-step snapshot records.
    """

    def __init__(self, policy_name: str, workload_label: str = "default"):
        """
        Initialise a MetricsCollector for one policy/workload pair.

        Args:
            policy_name (str):    Human-readable name of the cache policy.
            workload_label (str): Label for the workload being run.
        """
        self.policy_name = policy_name
        self.workload_label = workload_label
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._history: list[dict] = []

    def record(self, step: int, hit: bool, cache) -> None:
        """
        Record statistics for one access step.

        Reads cumulative totals from ``cache.get_stats()``, updates
        internal counters, and appends a snapshot to the step history.

        Args:
            step (int):  Zero-based index of the current access in the workload.
            hit (bool):  True if the access was a cache hit, False for a miss.
            cache:       Any cache object that exposes a ``get_stats()`` method
                         returning an object with ``hits``, ``misses``, and
                         ``evictions`` integer attributes.
        """
        stats = cache.get_stats()
        self.hits = stats.hits
        self.misses = stats.misses
        self.evictions = stats.evictions

        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        self._history.append({
            "step": step,
            "hit": hit,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 6),
            "policy": self.policy_name,
            "workload": self.workload_label,
        })

    def summary(self) -> str:
        """
        Return a one-line formatted summary of current statistics.

        Returns:
            str: Formatted string in the form:
                 ``"<policy> | Hit Rate: XX.XX% | Hits: N | Misses: N | Evictions: N"``

        Example:
            >>> collector.summary()
            'FIFO | Hit Rate: 75.00% | Hits: 150 | Misses: 50 | Evictions: 20'
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return (
            f"{self.policy_name} | "
            f"Hit Rate: {hit_rate:.2f}% | "
            f"Hits: {self.hits} | "
            f"Misses: {self.misses} | "
            f"Evictions: {self.evictions}"
        )

    def export_csv(self, path: str) -> None:
        """
        Export the step-by-step history to a CSV file.

        Each row represents one recorded access step and contains:
        ``step``, ``hit``, ``hits``, ``misses``, ``evictions``,
        ``hit_rate``, ``policy``, and ``workload`` columns.

        Args:
            path (str): Destination file path, e.g. ``"results/lru_run.csv"``.

        Raises:
            ValueError: If no steps have been recorded yet.
            OSError:    If the file cannot be written to ``path``.
        """
        if not self._history:
            raise ValueError(
                f"No steps recorded for policy '{self.policy_name}'. "
                "Call record() at least once before exporting."
            )

        df = pd.DataFrame(self._history)
        df.to_csv(path, index=False)