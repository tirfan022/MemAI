"""
main.py
-------
Intelligent Memory Manager — Simulation Entry Point

Runs all four cache policies (FIFO, LRU, Clock, ARC) against all four
workload types (sequential, random, looping, locality). For every
policy/workload combination a CSV is exported to results/. After all
simulations complete, three comparison bar charts are generated per
workload and saved to results/graphs/.
"""

import os

from cache.fifo  import FIFO
from cache.lru   import LRU
from cache.clock import Clock
from cache.arc   import ARC
from metrics.collector  import MetricsCollector
from metrics.visualizer import MetricsVisualizer
from worksload.generator import WorkloadGenerator


# ──────────────────────────────────────────
#  Configuration  — change only here
# ──────────────────────────────────────────

CAPACITY      = 32     # pages each cache can hold
NUM_PAGES     = 100    # total distinct pages in address space
NUM_ACCESSES  = 1000   # page accesses per workload
SEED          = 42     # reproducible workloads
RESULTS_DIR   = "results"
GRAPHS_DIR    = os.path.join(RESULTS_DIR, "graphs")


# ──────────────────────────────────────────
#  Policy registry
#  Add new policies here — nothing else changes
# ──────────────────────────────────────────

def build_policies() -> dict:
    """
    Instantiate every cache policy at the configured capacity.

    Returns:
        dict: {policy_name: cache_instance}
              e.g. {"FIFO": FIFO(32), "LRU": LRU(32), ...}
    """
    return {
        "FIFO":  FIFO(CAPACITY),
        "LRU":   LRU(CAPACITY),
        "Clock": Clock(CAPACITY),
        "ARC":   ARC(CAPACITY),
    }


# ──────────────────────────────────────────
#  Simulation helpers
# ──────────────────────────────────────────

def run_single(policy, workload: list, workload_label: str) -> MetricsCollector:
    """
    Run one cache policy against one workload and record metrics.

    Calls policy.reset() before the run so the same policy instance
    can be reused across multiple workloads without leftover state.

    Args:
        policy:          Any cache inheriting CacheBase.
        workload:        List of integer page numbers.
        workload_label:  Short name used in CSV filenames and summaries.

    Returns:
        MetricsCollector: Fully populated with per-step history.
    """
    metrics = MetricsCollector(
        policy_name=policy.__class__.__name__,
        workload_label=workload_label,
    )
    policy.reset()

    for step, page in enumerate(workload):
        hit = policy.access(page)
        metrics.record(step=step, hit=hit, cache=policy)

    return metrics


def run_all_combinations(policies: dict, workloads: dict) -> dict:
    """
    Run every policy against every workload and export a CSV for each pair.

    CSV naming convention: results/<policy_lower>_<workload_label>.csv
    e.g.  results/fifo_sequential.csv
          results/arc_locality.csv

    Args:
        policies:  {name: cache_instance}  from build_policies()
        workloads: {label: page_list}       from WorkloadGenerator.get_all_workloads()

    Returns:
        dict: Nested dict  csv_map[workload_label][policy_name] = csv_path
              Used by generate_graphs() to build the csv_files argument
              for MetricsVisualizer.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # csv_map[workload_label][policy_name] = filepath
    csv_map: dict = {label: {} for label in workloads}

    print("\n" + "=" * 70)
    print("  INTELLIGENT MEMORY MANAGER — SIMULATION RESULTS")
    print(f"  Capacity: {CAPACITY} pages  |  Accesses: {NUM_ACCESSES}  |  Seed: {SEED}")
    print("=" * 70)

    for workload_label, workload in workloads.items():
        print(f"\n── Workload: {workload_label.upper()} ──")

        for policy_name, policy in policies.items():
            metrics = run_single(policy, workload, workload_label)

            # Print one-line summary to terminal
            print(f"  {metrics.summary()}")

            # Export CSV
            filename = f"{policy_name.lower()}_{workload_label}.csv"
            csv_path = os.path.join(RESULTS_DIR, filename)
            metrics.export_csv(csv_path)

            csv_map[workload_label][policy_name] = csv_path

    return csv_map


# ──────────────────────────────────────────
#  Graph generation
# ──────────────────────────────────────────

def generate_graphs(csv_map: dict):
    """
    Generate hit rate, misses, and evictions bar charts for each workload.

    One set of three charts is produced per workload type, saved under
    results/graphs/<workload_label>_<metric>.png

    Args:
        csv_map: Nested dict returned by run_all_combinations().
                 csv_map[workload_label][policy_name] = csv_path
    """
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    viz = MetricsVisualizer()

    print("\n── Generating Graphs ──")

    for workload_label, csv_files in csv_map.items():
        # hit rates
        viz.compare_hit_rates(csv_files)
        viz.save_graph(os.path.join(GRAPHS_DIR, f"{workload_label}_hit_rates.png"))

        # misses
        viz.compare_misses(csv_files)
        viz.save_graph(os.path.join(GRAPHS_DIR, f"{workload_label}_misses.png"))

        # evictions
        viz.compare_evictions(csv_files)
        viz.save_graph(os.path.join(GRAPHS_DIR, f"{workload_label}_evictions.png"))


# ──────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────

def main():
    """
    Orchestrates the full benchmark pipeline:
        1. Build all policies
        2. Generate all workloads
        3. Run every policy × workload combination
        4. Export one CSV per combination
        5. Generate comparison graphs per workload
    """
    policies  = build_policies()
    generator = WorkloadGenerator(
        num_pages=NUM_PAGES,
        num_accesses=NUM_ACCESSES,
        seed=SEED,
    )
    workloads = generator.get_all_workloads()

    csv_map = run_all_combinations(policies, workloads)
    generate_graphs(csv_map)

    print("\n" + "=" * 70)
    print(f"  CSVs   → {RESULTS_DIR}/")
    print(f"  Graphs → {GRAPHS_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()