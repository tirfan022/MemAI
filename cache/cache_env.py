import numpy as np
import gymnasium as gym
from gymnasium import spaces

OBS_DIM = 188  # (16 slots * 10 features = 160) + 28 global features

class CacheEnv(gym.Env):
    def __init__(self, capacity=16, lookahead_trace=None):
        super().__init__()
        self.capacity = capacity
        self.lookahead_trace = lookahead_trace if lookahead_trace is not None else []
        
        # Define Spaces
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.capacity)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.slots = np.full(self.capacity, -1, dtype=np.int32)
        
        # Advanced Feature Tracking Arrays
        self.insertion_times = np.zeros(self.capacity, dtype=np.float32)
        self.last_access_times = np.zeros(self.capacity, dtype=np.float32)
        self.access_counts = np.zeros(self.capacity, dtype=np.float32)
        self.reuse_intervals = np.zeros(self.capacity, dtype=np.float32)
        
        # Metadata Metrics
        self.global_step = 0
        self.total_hits = 0
        self.history_buffer = []
        self.last_page = 0
        self.lru_order = []
        self.freq_map = {}
        
        return self._get_obs(next_page=0), {}

    def step(self, action, incoming_page):
        """
        Processes a single cache access step given an explicit incoming page 
        and an eviction slot action selected by the neural network.
        """
        self.global_step += 1
        hit = False
        evicted_page = -1
        
        # Track heuristics data
        if incoming_page in self.lru_order:
            self.lru_order.remove(incoming_page)
        self.lru_order.append(incoming_page)
        self.freq_map[incoming_page] = self.freq_map.get(incoming_page, 0) + 1

        # Determine Hit or Miss
        if incoming_page in self.slots:
            hit = True
            idx = np.where(self.slots == incoming_page)[0][0]
            interval = self.global_step - self.last_access_times[idx]
            self.last_access_times[idx] = self.global_step
            self.access_counts[idx] += 1
            
            # Streaming running mean of reuse interval
            n = self.access_counts[idx]
            self.reuse_intervals[idx] += (interval - self.reuse_intervals[idx]) / n
        else:
            # Cache Miss -> Evict page at the chosen action index slot
            evicted_page = self.slots[action]
            self.slots[action] = incoming_page
            self.insertion_times[action] = self.global_step
            self.last_access_times[action] = self.global_step
            self.access_counts[action] = 1
            self.reuse_intervals[action] = 0

        if hit:
            self.total_hits += 1
            reward = 1.0
        else:
            reward = self._calculate_oracle_reward(evicted_page, action)

        # Update historical track state
        self.history_buffer.append(incoming_page)
        if len(self.history_buffer) > 10:
            self.history_buffer.pop(0)

        self.last_page = incoming_page
        
        # Lookahead configuration for next state forecasting
        next_page_prediction = 0
        if self.global_step < len(self.lookahead_trace):
            next_page_prediction = self.lookahead_trace[self.global_step]

        obs = self._get_obs(next_page=next_page_prediction)
        return obs, reward, False, False, {"hit": hit}

    def _calculate_oracle_reward(self, evicted_page, action):
        if evicted_page == -1: 
            return 0.2  # Filled an empty block initialization bonus
            
        # Scan upcoming window to penalize premature evictions
        lookahead_window = self.lookahead_trace[self.global_step : self.global_step + 50]
        if not lookahead_window:
            return 0.0
            
        try:
            chosen_dist = lookahead_window.index(evicted_page)
        except ValueError:
            chosen_dist = 100  # Not reused soon
            
        max_dist = 0
        for p in self.slots:
            if p != -1:
                try:
                    d = lookahead_window.index(p)
                except ValueError:
                    d = 100
                if d > max_dist:
                    max_dist = d
                    
        return 0.5 * (chosen_dist / max(max_dist, 1))

    def _get_obs(self, next_page):
        t = max(self.global_step, 1)
        per_slot_features = []
        
        for i in range(self.capacity):
            p = self.slots[i]
            occupied = 1.0 if p != -1 else 0.0
            p_norm = p / 1000.0 if p != -1 else -1.0
            recency = (t - self.last_access_times[i]) / t
            frequency = np.log1p(self.access_counts[i]) / 10.0
            lifespan = (t - self.insertion_times[i]) / t
            ri_mean = self.reuse_intervals[i] / t
            
            lru_rank = (self.lru_order.index(p) / self.capacity) if p in self.lru_order else 1.0
            max_freq = max(self.freq_map.values(), default=1)
            lfu_rank = self.freq_map.get(p, 0) / max_freq
            
            per_slot_features.extend([occupied, p_norm, recency, frequency, lifespan, ri_mean, lru_rank, lfu_rank, 0.0, 0.0])

        # Global Features (28 Dimensions)
        stride = abs(next_page - self.last_page) / 1000.0
        hit_rate = self.total_hits / t
        padded_hist = (self.history_buffer + [0] * 10)[:10]
        hist_norm = [x / 1000.0 for x in padded_hist]
        
        global_features = [next_page / 1000.0, stride, hit_rate] + hist_norm
        global_features += [0.0] * (28 - len(global_features)) # Pad up to exact dimensions
        
        return np.concatenate([per_slot_features, global_features], dtype=np.float32)