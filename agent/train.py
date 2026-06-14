import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
import numpy as np
import agent as agent_file
from cache.cache_env import CacheEnv

DuelingDDQNAgent = agent_file.DuelingDDQNAgent
ReplayMemory = agent_file.ReplayMemory

MODEL_FILE = os.path.join(root_path, "cache", "memai_weights.pt")

def generate_clean_training_workloads(capacity):
    rng = np.random.default_rng(42)
    combined_trace = []
    
    # Focus training on high-locality and looping traces where patterns exist
    for _ in range(3): 
        combined_trace.extend(rng.integers(0, capacity + 4, size=1000).tolist()) # Dense Looping Loop
        pages = np.arange(capacity * 2)
        raw_probs = np.array([0.6 / 6 if i < 6 else 0.4 / (len(pages) - 6) for i in range(len(pages))])
        combined_trace.extend(rng.choice(pages, size=1000, p=raw_probs / np.sum(raw_probs)).tolist()) # Locality
        
    return combined_trace

def train_agent(capacity=32, episodes=15):
    print(f"[MemAI] Synthesizing target workload tracking maps for capacity: {capacity}...")
    training_trace = generate_clean_training_workloads(capacity)
    
    env = CacheEnv(capacity=capacity, lookahead_trace=training_trace)
    policy_net = DuelingDDQNAgent(action_dim=capacity)
    target_net = DuelingDDQNAgent(action_dim=capacity)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=2e-4, weight_decay=1e-4)
    memory = ReplayMemory(max_len=40000)
    
    batch_size = 64
    gamma = 0.99
    
    print(f"[MemAI] Training model across {episodes} optimization episodes...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        epsilon = max(0.05, 1.0 - (ep / (episodes * 0.80)))
        hits = 0
        
        for step_idx, page in enumerate(training_trace[:-1]):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Match the empty-slot allocation bypass logic during model tracking loops
            if page in env.slots:
                action = 0
            elif -1 in env.slots:
                action = np.where(env.slots == -1)[0][0]
            else:
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, capacity)
                else:
                    with torch.no_grad():
                        action = policy_net(obs_t).argmax(dim=1).item()
                    
            next_obs, reward, _, _, info = env.step(action, page)
            if info.get("hit", False):
                hits += 1
                
            memory.push((obs, action, reward, next_obs))
            obs = next_obs
            
            if len(memory) > batch_size and step_idx % 4 == 0:
                batch = memory.sample(batch_size)
                s_b, a_b, r_b, ns_b = zip(*batch)
                
                s_t = torch.tensor(np.array(s_b), dtype=torch.float32)
                a_t = torch.tensor(a_b, dtype=torch.long).unsqueeze(1)
                r_t = torch.tensor(r_b, dtype=torch.float32)
                ns_t = torch.tensor(np.array(ns_b), dtype=torch.float32)
                
                q_values = policy_net(s_t).gather(1, a_t).squeeze(1)
                with torch.no_grad():
                    next_actions = policy_net(ns_t).argmax(dim=1).unsqueeze(1)
                    max_next_q = target_net(ns_t).gather(1, next_actions).squeeze(1)
                    target_q = r_t + (gamma * max_next_q)
                    
                loss = nn.SmoothL1Loss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        target_net.load_state_dict(policy_net.state_dict())
        hit_rate = (hits / len(training_trace)) * 100
        print(f"  Episode {ep+1:2d}/{episodes} | Epsilon: {epsilon:.2f} | Aligned Hit Rate: {hit_rate:.2f}%")

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    torch.save(policy_net.state_dict(), MODEL_FILE)
    print(f"[MemAI] Deep learning model optimization complete!")

if __name__ == "__main__":
    train_agent(capacity=32, episodes=15)