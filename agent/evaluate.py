import os
import sys

root_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if root_path not in sys.path:
    sys.path.insert(0, root_path)

import numpy as np
import torch

import agent as agent_file

DuelingDDQNAgent = agent_file.DuelingDDQNAgent
from cache.cache_env import CacheEnv
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

MODEL_FILE = os.path.join(
    root_path,
    "cache",
    "memai_weights.pt",
)


def evaluate_trace(
    model,
    trace,
    capacity,
):
    env = CacheEnv(
        capacity=capacity
    )

    observation, _ = env.reset()

    hits = 0

    for page in trace:

        state = torch.tensor(
            observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        # Cache hit:
        # no eviction decision required
        if page in env.slots:
            action = 0

        # Empty cache slot available
        elif -1 in env.slots:
            action = int(
                np.where(
                    env.slots == -1
                )[0][0]
            )

        # Cache full:
        # RL agent chooses eviction slot
        else:
            with torch.no_grad():
                q_values = model(state)

                action = int(
                    torch.argmax(
                        q_values,
                        dim=1,
                    ).item()
                )

        observation, _, _, _, info = env.step(
            action,
            page,
        )

        if info["hit"]:
            hits += 1

    hit_rate = (
        hits / len(trace)
    ) * 100

    return hits, hit_rate


def generate_workloads(seed):
    generator = WorkloadGenerator(
        num_pages=NUM_PAGES,
        num_accesses=NUM_ACCESSES,
        seed=seed,
    )

    return {
        "SEQUENTIAL":
            generator.generate_sequential(),

        "RANDOM":
            generator.generate_random(),

        "LOOPING":
            generator.generate_looping(),

        "LOCALITY":
            generator.generate_locality(),
    }


def main():
    print("=" * 78)

    print(
        "MEMAI — MULTI-SEED "
        "INDEPENDENT RL EVALUATION"
    )

    print("=" * 78)

    print(
        f"Capacity: {CAPACITY} | "
        f"Accesses: {NUM_ACCESSES}"
    )

    print(
        "Evaluation seeds: "
        + ", ".join(
            map(str, EVALUATION_SEEDS)
        )
    )

    print("=" * 78)


    if not os.path.exists(MODEL_FILE):
        print(
            "Model weights not found."
        )

        print(
            "Run: python agent\\train.py"
        )

        return


    model = DuelingDDQNAgent(
        action_dim=CAPACITY
    )

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu",
        )
    )

    model.eval()


    results = {
        "SEQUENTIAL": [],
        "RANDOM": [],
        "LOOPING": [],
        "LOCALITY": [],
    }


    for seed in EVALUATION_SEEDS:
        print(
            f"\n── Seed: {seed} ──"
        )

        workloads = generate_workloads(
            seed
        )

        for workload_name, trace in workloads.items():
            hits, hit_rate = evaluate_trace(
                model=model,
                trace=trace,
                capacity=CAPACITY,
            )

            results[
                workload_name
            ].append(hit_rate)

            print(
                f"{workload_name:12s} | "
                f"Hit Rate: {hit_rate:6.2f}% | "
                f"Hits: {hits:4d} | "
                f"Misses: "
                f"{len(trace) - hits:4d}"
            )


    print("\n" + "=" * 78)

    print(
        "MULTI-SEED SUMMARY"
    )

    print("=" * 78)


    for workload_name, hit_rates in results.items():
        mean_hit_rate = np.mean(
            hit_rates
        )

        std_hit_rate = np.std(
            hit_rates
        )

        minimum = np.min(
            hit_rates
        )

        maximum = np.max(
            hit_rates
        )

        print(
            f"{workload_name:12s} | "
            f"Mean: {mean_hit_rate:6.2f}% | "
            f"Std: {std_hit_rate:5.2f}% | "
            f"Min: {minimum:6.2f}% | "
            f"Max: {maximum:6.2f}%"
        )


    print("=" * 78)


if __name__ == "__main__":
    main()