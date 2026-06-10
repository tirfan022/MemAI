"""
main.py
-------
Intelligent Memory Manager - Simulation Entry Point

Runs FIFO and LRU cache policies on a sample workload,
collects metrics, prints summaries, and exports CSV reports.
"""

import os

from cache.fifo import FIFO
from cache.lru import LRU
from metrics.collector import MetricsCollector


# --------------------------------------------------
# Configuration
# --------------------------------------------------

WORKLOAD = [1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 3, 4, 5]
CAPACITY = 3
RESULTS_DIR = "results"


# --------------------------------------------------
# Simulation Helpers
# --------------------------------------------------

def run_single(policy, workload, workload_label="default"):
    """
    Run one cache policy on one workload.

    Args:
        policy: Cache object (FIFO, LRU, etc.)
        workload: List of page references
        workload_label: Name of workload

    Returns:
        MetricsCollector
    """

    metrics = MetricsCollector(
        policy_name=policy.__class__.__name__,
        workload_label=workload_label
    )

    policy.reset()

    for step, page in enumerate(workload):
        hit = policy.access(page)
        metrics.record(
            step=step,
            hit=hit,
            cache=policy
        )

    return metrics


# --------------------------------------------------
# Main Runner
# --------------------------------------------------

def main():
    """
    Run all cache policies and export results.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    policies = [
        FIFO(CAPACITY),
        LRU(CAPACITY)
    ]

    print("\n" + "=" * 60)
    print("INTELLIGENT MEMORY MANAGER")
    print("=" * 60)

    for policy in policies:

        result = run_single(
            policy=policy,
            workload=WORKLOAD,
            workload_label="sample_workload"
        )

        print(result.summary())

        csv_path = os.path.join(
            RESULTS_DIR,
            f"{policy.__class__.__name__.lower()}.csv"
        )

        result.export_csv(csv_path)

    print("\nCSV files saved in ./results/")


if __name__ == "__main__":
    main()