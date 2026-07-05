import numpy as np

from cache.fifo import FIFO
from cache.lru import LRU
from cache.clock import Clock
from cache.arc import ARC
from cache.memai_policy import MemAIPolicy
from workload.generator import WorkloadGenerator


CAPACITY = 32
NUM_PAGES = 100
NUM_ACCESSES = 1000

EVALUATION_SEEDS = [
    101,
    202,
    303,
    404,
    505,
]


def build_policies():
    return {
        "FIFO": FIFO(CAPACITY),
        "LRU": LRU(CAPACITY),
        "Clock": Clock(CAPACITY),
        "ARC": ARC(CAPACITY),
        "MemAI": MemAIPolicy(CAPACITY),
    }


def run_policy(policy, workload):
    policy.reset()

    hits = 0

    for page in workload:
        hit = policy.access(page)

        if hit:
            hits += 1

    hit_rate = (
        hits / len(workload)
    ) * 100

    return hits, hit_rate


def generate_workloads(seed):
    generator = WorkloadGenerator(
        num_pages=NUM_PAGES,
        num_accesses=NUM_ACCESSES,
        seed=seed,
    )

    return generator.get_all_workloads()


def main():
    print("=" * 90)

    print(
        "MEMAI — MULTI-SEED "
        "COMPARATIVE BENCHMARK"
    )

    print("=" * 90)

    print(
        f"Capacity: {CAPACITY} | "
        f"Pages: {NUM_PAGES} | "
        f"Accesses: {NUM_ACCESSES}"
    )

    print(
        "Seeds: "
        + ", ".join(
            map(str, EVALUATION_SEEDS)
        )
    )

    print("=" * 90)


    results = {
        "sequential": {},
        "random": {},
        "looping": {},
        "locality": {},
    }


    for seed in EVALUATION_SEEDS:
        print(
            f"\n{'=' * 30} "
            f"SEED {seed} "
            f"{'=' * 30}"
        )

        workloads = generate_workloads(
            seed
        )

        policies = build_policies()

        for workload_name, workload in workloads.items():
            print(
                f"\n── {workload_name.upper()} ──"
            )

            for policy_name, policy in policies.items():
                hits, hit_rate = run_policy(
                    policy,
                    workload,
                )

                if policy_name not in results[
                    workload_name
                ]:
                    results[
                        workload_name
                    ][policy_name] = []

                results[
                    workload_name
                ][policy_name].append(
                    hit_rate
                )

                print(
                    f"{policy_name:8s} | "
                    f"Hit Rate: {hit_rate:6.2f}% | "
                    f"Hits: {hits:4d} | "
                    f"Misses: "
                    f"{len(workload) - hits:4d}"
                )


    print("\n" + "=" * 90)

    print(
        "FINAL MULTI-SEED COMPARISON"
    )

    print("=" * 90)


    for workload_name, policy_results in results.items():
        print(
            f"\n── {workload_name.upper()} ──"
        )

        sorted_results = sorted(
            policy_results.items(),
            key=lambda item: np.mean(
                item[1]
            ),
            reverse=True,
        )

        for rank, (
            policy_name,
            hit_rates,
        ) in enumerate(
            sorted_results,
            start=1,
        ):
            mean = np.mean(
                hit_rates
            )

            std = np.std(
                hit_rates
            )

            minimum = np.min(
                hit_rates
            )

            maximum = np.max(
                hit_rates
            )

            print(
                f"#{rank} "
                f"{policy_name:8s} | "
                f"Mean: {mean:6.2f}% | "
                f"Std: {std:5.2f}% | "
                f"Min: {minimum:6.2f}% | "
                f"Max: {maximum:6.2f}%"
            )


    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()