import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class DuelingDDQNAgent(nn.Module):
    def __init__(self, action_dim=32):
        super().__init__()
        self.action_dim = action_dim
        
        # Calculate dynamic input dimension based on cache slot capacity
        # (capacity * 10 slot features) + 28 global features
        self.obs_dim = (action_dim * 10) + 28
        
        # History seq tracking window (always 10 steps)
        self.gru = nn.GRU(input_size=1, hidden_size=32, batch_first=True)
        
        # Dynamic Layer Input Calculation:
        # Total observation features minus the 10 raw history fields plus 32 GRU hidden states
        self.flat_features_dim = self.obs_dim - 10 + 32
        
        self.feature_dense = nn.Sequential(
            nn.Linear(self.flat_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Dueling Splits
        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, action_dim)

    def forward(self, obs):
        # Determine slice splits dynamically based on current capacity configurations
        slots_end = self.action_dim * 10
        
        per_slot = obs[:, :slots_end]
        global_prefix = obs[:, slots_end : slots_end + 3]
        history_seq = obs[:, slots_end + 3 : slots_end + 13].unsqueeze(-1) # Shape: (B, 10, 1)
        padding_rem = obs[:, slots_end + 13 :]
        
        _, hidden = self.gru(history_seq)
        gru_extracted = hidden.squeeze(0) # Shape: (B, 32)
        
        combined_tensor = torch.cat([per_slot, global_prefix, gru_extracted, padding_rem], dim=1)
        x = self.feature_dense(combined_tensor)
        
        val = self.value_head(x)
        adv = self.advantage_head(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

class ReplayMemory:
    def __init__(self, max_len=20000):
        self.buffer = deque(maxlen=max_len)
    def push(self, transition):
        self.buffer.append(transition)
    def sample(self, b_size):
        return random.sample(self.buffer, b_size)
    def __len__(self):
        return len(self.buffer)