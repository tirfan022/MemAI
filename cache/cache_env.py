import numpy as np
import gymnasium as gym

from gymnasium import spaces


SLOT_FEATURES = 10
GLOBAL_FEATURES = 28


class CacheEnv(gym.Env):

    def __init__(
        self,
        capacity=16,
        lookahead_trace=None,
    ):
        super().__init__()

        if capacity <= 0:
            raise ValueError(
                "capacity must be greater than 0"
            )

        self.capacity = capacity

        # Kept only for backward compatibility
        # with older training/test code.
        #
        # IMPORTANT:
        # It is NOT used for observations
        # or reward calculation.
        self.lookahead_trace = (
            lookahead_trace
            if lookahead_trace is not None
            else []
        )

        self.obs_dim = (
            capacity * SLOT_FEATURES
        ) + GLOBAL_FEATURES

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(
            capacity
        )

        self.reset()


    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.slots = np.full(
            self.capacity,
            -1,
            dtype=np.int32,
        )

        self.insertion_times = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.last_access_times = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.access_counts = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.reuse_intervals = np.zeros(
            self.capacity,
            dtype=np.float32,
        )


        self.global_step = 0

        self.total_hits = 0

        self.total_misses = 0


        self.history_buffer = []

        self.last_page = 0


        self.lru_order = []

        self.freq_map = {}


        observation = self._get_obs()

        return observation, {}


    def step(
        self,
        action,
        incoming_page,
    ):

        if not (
            0 <= action < self.capacity
        ):
            raise ValueError(
                f"Invalid action {action}. "
                f"Expected value between "
                f"0 and {self.capacity - 1}."
            )


        self.global_step += 1


        hit = False

        evicted_page = -1


        # Update global access frequency.

        self.freq_map[incoming_page] = (
            self.freq_map.get(
                incoming_page,
                0,
            )
            + 1
        )


        # Update global LRU order.

        if incoming_page in self.lru_order:

            self.lru_order.remove(
                incoming_page
            )


        self.lru_order.append(
            incoming_page
        )


        # Limit LRU tracking growth.

        max_lru_history = (
            self.capacity * 4
        )


        if (
            len(self.lru_order)
            > max_lru_history
        ):

            self.lru_order = (
                self.lru_order[
                    -max_lru_history:
                ]
            )


        # ──────────────────────────────
        # CACHE HIT
        # ──────────────────────────────

        if incoming_page in self.slots:

            hit = True


            index = int(
                np.where(
                    self.slots
                    == incoming_page
                )[0][0]
            )


            previous_access_time = (
                self.last_access_times[
                    index
                ]
            )


            interval = (
                self.global_step
                - previous_access_time
            )


            self.last_access_times[
                index
            ] = self.global_step


            self.access_counts[
                index
            ] += 1


            access_count = (
                self.access_counts[
                    index
                ]
            )


            self.reuse_intervals[
                index
            ] += (

                interval
                - self.reuse_intervals[
                    index
                ]

            ) / access_count


            self.total_hits += 1


            # Observed outcome reward.
            #
            # A hit means a previous eviction
            # decision preserved a useful page.

            reward = 1.0


        # ──────────────────────────────
        # CACHE MISS
        # ──────────────────────────────

        else:

            self.total_misses += 1


            evicted_page = int(
                self.slots[action]
            )


            self.slots[action] = (
                incoming_page
            )


            self.insertion_times[
                action
            ] = self.global_step


            self.last_access_times[
                action
            ] = self.global_step


            self.access_counts[
                action
            ] = 1.0


            self.reuse_intervals[
                action
            ] = 0.0


            # No future workload access.
            #
            # A miss is an observed negative
            # cache outcome.

            if evicted_page == -1:

                # Initial cache population is
                # not an eviction mistake.

                reward = 0.0

            else:

                reward = -1.0


        # ──────────────────────────────
        # UPDATE OBSERVED HISTORY
        # ──────────────────────────────

        self.history_buffer.append(
            incoming_page
        )


        if len(self.history_buffer) > 10:

            self.history_buffer.pop(0)


        self.last_page = incoming_page


        observation = self._get_obs()


        info = {

            "hit": hit,

            "evicted_page": evicted_page,

            "incoming_page": incoming_page,

            "total_hits": self.total_hits,

            "total_misses": self.total_misses,

        }


        return (
            observation,
            reward,
            False,
            False,
            info,
        )


    def _calculate_oracle_reward(
        self,
        evicted_page,
        action,
    ):
        """
        Backward-compatible method.

        Older versions of MemAI inspected
        future workload accesses here.

        Oracle reward shaping has been removed.

        This method intentionally uses no
        lookahead information.
        """

        if evicted_page == -1:

            return 0.0


        return -1.0


    def _get_obs(self):

        time_step = max(
            self.global_step,
            1,
        )


        per_slot_features = []


        max_frequency = max(
            self.freq_map.values(),
            default=1,
        )


        for index in range(
            self.capacity
        ):

            page = self.slots[index]


            occupied = (
                1.0
                if page != -1
                else 0.0
            )


            page_normalized = (
                min(
                    page / 1000.0,
                    1.0,
                )
                if page != -1
                else -1.0
            )


            recency = (
                time_step
                - self.last_access_times[
                    index
                ]
            ) / time_step


            frequency = (
                np.log1p(
                    self.access_counts[
                        index
                    ]
                )
                / 10.0
            )


            lifespan = (
                time_step
                - self.insertion_times[
                    index
                ]
            ) / time_step


            reuse_mean = (
                self.reuse_intervals[
                    index
                ]
                / time_step
            )


            if page in self.lru_order:

                lru_position = (
                    self.lru_order.index(
                        page
                    )
                )

                lru_rank = (
                    lru_position
                    / max(
                        len(
                            self.lru_order
                        ),
                        1,
                    )
                )

            else:

                lru_rank = 1.0


            lfu_rank = (

                self.freq_map.get(
                    page,
                    0,
                )

                / max_frequency

                if page != -1

                else 0.0

            )


            age_since_insert = min(
                lifespan,
                1.0,
            )


            reuse_signal = min(
                reuse_mean,
                1.0,
            )


            per_slot_features.extend(
                [
                    occupied,
                    page_normalized,
                    recency,
                    frequency,
                    lifespan,
                    reuse_mean,
                    lru_rank,
                    lfu_rank,
                    age_since_insert,
                    reuse_signal,
                ]
            )


        # ──────────────────────────────
        # GLOBAL FEATURES
        # ──────────────────────────────

        hit_rate = (

            self.total_hits
            / time_step

        )


        miss_rate = (

            self.total_misses
            / time_step

        )


        occupied_slots = (

            np.count_nonzero(
                self.slots != -1
            )

            / self.capacity

        )


        padded_history = (

            self.history_buffer
            + [0] * 10

        )[:10]


        history_normalized = [

            min(
                page / 1000.0,
                1.0,
            )

            for page in padded_history

        ]


        last_page_normalized = min(
            self.last_page / 1000.0,
            1.0,
        )


        global_features = [

            hit_rate,

            miss_rate,

            occupied_slots,

            last_page_normalized,

        ]


        global_features.extend(
            history_normalized
        )


        # Exactly 28 global features.

        remaining_features = (

            GLOBAL_FEATURES
            - len(global_features)

        )


        global_features.extend(
            [0.0] * remaining_features
        )


        observation = np.concatenate(
            [
                np.asarray(
                    per_slot_features,
                    dtype=np.float32,
                ),
                np.asarray(
                    global_features,
                    dtype=np.float32,
                ),
            ]
        )


        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )


        observation = np.clip(
            observation,
            -1.0,
            1.0,
        )


        return observation.astype(
            np.float32
        )