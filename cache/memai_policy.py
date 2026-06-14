import os
import torch
import numpy as np
from cache.base import CacheBase
from cache.cache_env import CacheEnv
from agent.agent import DuelingDDQNAgent

class MemAIPolicy(CacheBase):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.env = CacheEnv(capacity=capacity)
        self.agent = DuelingDDQNAgent(action_dim=capacity)
        self.total_evictions = 0
        
        model_path = os.path.join(os.path.dirname(__file__), "memai_weights.pt")
        if os.path.exists(model_path):
            self.agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.agent.eval()
        else:
            print(f"[MemAI] Warning: Weights profile not found at {model_path}. Run agent/train.py first.")
            
        self.current_obs, _ = self.env.reset()

    def access(self, page):
        # 1. Hit Check
        if page in self.env.slots:
            self.current_obs, _, _, _, info = self.env.step(action=0, incoming_page=page)
            return True
            
        # 2. Compulsory Miss / Empty Slot Warmup (Prevents Action Collapse)
        if -1 in self.env.slots:
            first_empty = np.where(self.env.slots == -1)[0][0]
            self.current_obs, _, _, _, info = self.env.step(action=first_empty, incoming_page=page)
            return False
            
        # 3. Cache is Full -> Invoke Deep Learning Eviction Policy
        obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            chosen_action = self.agent(obs_tensor).argmax(dim=1).item()
            
        if self.env.slots[chosen_action] != -1:
            self.total_evictions += 1
            
        self.current_obs, _, _, _, info = self.env.step(action=chosen_action, incoming_page=page)
        return False

    def reset(self):
        self.current_obs, _ = self.env.reset()
        self.total_evictions = 0

    def get_stats(self):
        hits = self.env.total_hits
        accesses = self.env.global_step
        misses = max(0, accesses - hits)

        class PolicyStats:
            def __init__(self, hits, accesses, misses, evictions, capacity):
                self.hits = hits
                self.accesses = accesses
                self.misses = misses
                self.evictions = evictions
                self.capacity = capacity
                self.hit_rate = (hits / accesses) if accesses > 0 else 0.0

        return PolicyStats(hits=hits, accesses=accesses, misses=misses, evictions=self.total_evictions, capacity=self.capacity)