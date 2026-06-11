"""
workloads/generator.py
----------------------
Generates sequences of page accesses for benchmarking cache policies.

Each workload type represents a different real-world access pattern:
    - Sequential: full scans, streaming reads
    - Random:     worst-case unpredictable access
    - Looping:    repeated iteration over a working set
    - Locality:   realistic apps where 20% of pages get 80% of accesses
"""

import random
from typing import List, Dict


class WorkloadGenerator:
    """
    Generates page access sequences for cache benchmarking.

    All methods produce a list of integer page numbers of exactly
    num_accesses length. The same seed always produces the same
    workload, ensuring reproducible benchmark results.

    Args:
        num_pages   (int): Total distinct pages in the address space.
        num_accesses (int): Number of page accesses to generate.
        seed        (int): Random seed for reproducibility.

    Example:
        gen = WorkloadGenerator(num_pages=100, num_accesses=1000, seed=42)
        workload = gen.generate_looping()
    """

    def __init__(self, num_pages: int = 100, num_accesses: int = 1000, seed: int = 42):
        if num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got: {num_pages}")
        if num_accesses <= 0:
            raise ValueError(f"num_accesses must be positive, got: {num_accesses}")

        self.num_pages = num_pages
        self.num_accesses = num_accesses
        self.seed = seed

    def generate_sequential(self) -> List[int]:
        """
        Generate a sequential page access pattern.

        Pages are accessed in ascending order: 0, 1, 2, ..., n-1, 0, 1, 2, ...
        The sequence wraps around and repeats until num_accesses is reached.

        Real-world analogy:
            A database doing a full table scan, or a video decoder
            reading frames in order.

        Cache behaviour:
            Worst case for small caches — by the time the sequence wraps,
            early pages have already been evicted. All policies struggle
            equally here unless cache size >= num_pages.

        Returns:
            List[int]: Sequential page access sequence.

        Example (num_pages=4, num_accesses=10):
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        """
        pages = list(range(self.num_pages))
        result = []
        while len(result) < self.num_accesses:
            result.extend(pages)
        return result[:self.num_accesses]

    def generate_random(self) -> List[int]:
        """
        Generate a fully random page access pattern.

        Each access independently picks a uniformly random page
        from [0, num_pages - 1]. No pattern or structure.

        Real-world analogy:
            Random memory reads in a stress test, or hash table
            lookups over a large key space.

        Cache behaviour:
            Worst case for all policies — no locality to exploit.
            Hit rate approaches cache_size / num_pages as accesses grow.

        Returns:
            List[int]: Random page access sequence.
        """
        rng = random.Random(self.seed)
        return [rng.randint(0, self.num_pages - 1) for _ in range(self.num_accesses)]

    def generate_looping(self, loop_size: int = 10) -> List[int]:
        """
        Generate a looping page access pattern over a fixed working set.

        Repeatedly cycles through a small set of pages: 0, 1, 2, ...,
        loop_size-1, 0, 1, 2, ... until num_accesses is reached.

        Real-world analogy:
            A tight loop in a program repeatedly accessing the same
            variables, or a game engine cycling through a small set
            of active assets each frame.

        Cache behaviour:
            Best case for LRU and most policies — if cache size >=
            loop_size, every access after warmup is a hit. Clearly
            shows the advantage of recency-aware policies.

        Args:
            loop_size (int): Number of distinct pages in the loop.
                             Clamped to num_pages if larger.
                             Defaults to 10.

        Returns:
            List[int]: Looping page access sequence.

        Example (loop_size=3, num_accesses=9):
            [0, 1, 2, 0, 1, 2, 0, 1, 2]
        """
        loop_size = min(loop_size, self.num_pages)
        loop = list(range(loop_size))
        result = []
        while len(result) < self.num_accesses:
            result.extend(loop)
        return result[:self.num_accesses]

    def generate_locality(self, hot_ratio: float = 0.2, hot_prob: float = 0.8) -> List[int]:
        """
        Generate a workload with temporal locality (the 80/20 rule).

        A small fraction of pages (hot set) receive the majority of
        accesses. The remaining pages (cold set) are accessed rarely.

        Real-world analogy:
            Most real applications — a web server where a few pages
            get most of the traffic, or a database where a small set
            of rows are queried repeatedly.

        Cache behaviour:
            The most realistic benchmark. LRU and ARC outperform FIFO
            here because they retain the hot pages. Directly measures
            how well a policy exploits temporal locality.

        Args:
            hot_ratio (float): Fraction of pages in the hot set.
                               Default 0.2 means 20% of pages are hot.
            hot_prob  (float): Probability that any access goes to a
                               hot page. Default 0.8 means 80% of
                               accesses hit the hot set.

        Returns:
            List[int]: Locality-based page access sequence.
        """
        rng = random.Random(self.seed)

        hot_count = max(1, int(self.num_pages * hot_ratio))
        hot_pages = list(range(hot_count))
        cold_pages = list(range(hot_count, self.num_pages))

        result = []
        for _ in range(self.num_accesses):
            if rng.random() < hot_prob and hot_pages:
                result.append(rng.choice(hot_pages))
            elif cold_pages:
                result.append(rng.choice(cold_pages))
            else:
                result.append(rng.choice(hot_pages))

        return result

    def get_all_workloads(self) -> Dict[str, List[int]]:
        """
        Generate all four workload types and return them as a dictionary.

        Provides a single convenient call to get every workload needed
        for a full benchmark run. All workloads share the same
        num_pages, num_accesses, and seed settings.

        Returns:
            Dict[str, List[int]]: Dictionary with keys:
                "sequential" — sequential scan pattern
                "random"     — uniformly random accesses
                "looping"    — fixed working set loop
                "locality"   — hot/cold temporal locality

        Example:
            workloads = gen.get_all_workloads()
            for name, pages in workloads.items():
                run_simulation(policy, pages, label=name)
        """
        return {
            "sequential": self.generate_sequential(),
            "random":     self.generate_random(),
            "looping":    self.generate_looping(),
            "locality":   self.generate_locality(),
        }