# MemAI — Intelligent Memory Manager

MemAI is a reinforcement learning-based cache replacement simulator that learns eviction decisions from memory-access behavior.

The project implements a **GRU-enhanced Dueling Double Deep Q-Network (Dueling DDQN)** and compares the learned cache replacement policy against traditional algorithms including **FIFO, LRU, Clock, and ARC**.

Unlike fixed cache replacement heuristics, MemAI models cache eviction as a sequential decision problem. It observes cache metadata and historical access behavior, selects a cache slot for eviction, and learns from the actual reuse behavior of the evicted page.

---

## Overview

Traditional cache replacement algorithms use predefined rules:

- **FIFO** evicts the oldest inserted page.
- **LRU** evicts the least recently used page.
- **Clock** approximates recency using reference bits.
- **ARC** dynamically balances recency and frequency.

These policies are effective for many workloads, but their behavior is determined by fixed heuristics.

MemAI explores a learning-based alternative.

The central question behind the project is:

> Can a reinforcement learning agent learn cache eviction behavior directly from observed memory-access patterns?

MemAI formulates cache replacement as an event-driven reinforcement learning problem where the agent makes decisions only when the cache is full and an eviction is required.

---

## Key Features

- Reinforcement learning-based cache replacement
- Dueling Double Deep Q-Network architecture
- GRU-based temporal access-history encoding
- Event-driven reinforcement learning transitions
- Delayed eviction-specific reward assignment
- No future memory-access information exposed to the agent
- Dynamic cache-capacity-aware observation space
- Experience replay
- Target network stabilization
- Gradient clipping
- Epsilon-greedy exploration
- Multi-seed comparative benchmarking
- CSV metrics export
- Automated benchmark graph generation
- Extensive automated test suite

---

## System Architecture

```text
Memory Access Trace
        |
        v
+---------------------------+
|        CacheEnv           |
|---------------------------|
| Cache Slot Metadata       |
| Recency                   |
| Frequency                 |
| Reuse Intervals           |
| LRU Rank                  |
| LFU Signal                |
| Observed Access History   |
+---------------------------+
        |
        v
 Observation Vector
        |
        v
+---------------------------+
|            GRU            |
|---------------------------|
| Temporal History Encoder  |
+---------------------------+
        |
        v
+---------------------------+
|     Dense Feature Layers  |
+---------------------------+
        |
        v
+---------------------------+
|      Dueling DDQN         |
|---------------------------|
| State Value Stream        |
| Advantage Stream          |
+---------------------------+
        |
        v
 Q-Value for Each Cache Slot
        |
        v
 Select Eviction Slot
        |
        v
 Track Evicted Page
        |
        v
 Observe Real Memory Accesses
        |
        v
 Delayed Eviction Reward
        |
        v
 Event-Based Replay Transition
        |
        v
 DDQN Optimization
```

---

## Reinforcement Learning Formulation

### State

The environment constructs an observation using per-cache-slot features and global cache statistics.

Each cache slot contains features describing:

- Occupancy
- Normalized page identifier
- Recency
- Access frequency
- Cache lifespan
- Mean reuse interval
- LRU rank
- LFU signal
- Age since insertion
- Reuse behavior

The global state also contains:

- Current hit rate
- Current miss rate
- Cache occupancy ratio
- Last observed page
- Recent memory-access history

For a cache capacity `C`, the observation dimension is:

```text
Observation Dimension = (C × 10) + 28
```

For the default capacity of 32 pages:

```text
Observation Dimension = 348
```

The observation is constructed only from currently available cache information and previously observed accesses.

---

## GRU-Based Access History Encoding

Memory-access patterns are sequential.

To capture temporal behavior, MemAI processes the recent access-history sequence using a **Gated Recurrent Unit (GRU)**.

```text
Recent Access History
        |
        v
       GRU
        |
        v
32-Dimensional Temporal Representation
```

The GRU representation is combined with cache-slot metadata and global cache features before being passed to the Dueling DDQN layers.

This allows the network to model short-term access patterns rather than treating every cache state as an isolated observation.

---

## Dueling Double DQN

MemAI uses a Dueling Double Deep Q-Network.

The neural network separates Q-value estimation into two components.

### State Value

```text
V(s)
```

The value stream estimates the overall quality of the current cache state.

### Action Advantage

```text
A(s, a)
```

The advantage stream estimates the relative benefit of evicting a specific cache slot.

The final Q-value is calculated as:

```text
Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
```

This allows the network to separately learn:

- How useful the current cache state is
- Which eviction action is preferable

Double DQN target calculation is used to reduce Q-value overestimation.

The policy network selects the next action:

```text
a* = argmax Q_policy(s', a)
```

The target network evaluates that action:

```text
Q_target(s', a*)
```

---

## Event-Driven Reinforcement Learning

A cache replacement agent does not need to make an eviction decision on every memory access.

Consider:

```text
Cache Hit
Cache Hit
Cache Hit
Cache Miss + Full Cache
```

Only the final access requires an eviction decision.

MemAI therefore uses **event-driven RL transitions**.

```text
Eviction Decision
        |
        v
Observed Memory Accesses
        |
        v
Next Eviction Decision
```

The next reinforcement learning state corresponds to the next actual eviction event rather than simply the next memory access.

The temporal gap between decisions is reflected through discounted future value:

```text
discount = gamma ^ decision_gap
```

This better aligns the reinforcement learning process with the actual cache replacement decision process.

---

## Delayed Eviction-Specific Reward

A major challenge in cache replacement reinforcement learning is credit assignment.

A general cache hit reward does not directly identify whether a specific eviction decision was good or bad.

MemAI instead tracks the exact page selected for eviction.

```text
Agent Evicts Page X
        |
        v
Track Page X
        |
        v
Observe Future Accesses as They Occur
        |
        +---- Page X Reused Soon ----> Penalty
        |
        +---- Page X Not Reused -----> Positive Reward
```

The training system uses a 50-access observation horizon.

The eviction reward is based on the observed reuse distance:

```text
reward = 2 × (reuse_distance / horizon) - 1
```

Examples:

| Observed Outcome | Approximate Reward |
|---|---:|
| Reused after 1 access | -0.96 |
| Reused after 10 accesses | -0.60 |
| Reused after 25 accesses | 0.00 |
| Reused after 40 accesses | +0.60 |
| Not reused within 50 accesses | +1.00 |

The environment does not reveal future memory references to the agent.

The reward is resolved only after accesses actually occur during simulation.

---

## Cache Policies

MemAI currently benchmarks five cache replacement policies.

| Policy | Description |
|---|---|
| FIFO | First-In First-Out replacement |
| LRU | Least Recently Used replacement |
| Clock | Clock-based recency approximation |
| ARC | Adaptive Replacement Cache |
| MemAI | GRU-enhanced Dueling DDQN policy |

---

## Workload Types

The benchmark evaluates cache policies across four workload patterns.

### Sequential

Pages are accessed in sequential order.

```text
0, 1, 2, 3, 4, 5, ...
```

This workload can cause repeated scan behavior and cache thrashing.

### Random

Pages are accessed uniformly from the address space.

```text
17, 82, 4, 61, 33, ...
```

Random workloads contain limited predictable structure.

### Looping

A small group of pages is repeatedly accessed.

```text
0, 1, 2, ..., 9,
0, 1, 2, ..., 9,
...
```

This represents a stable working set.

### Locality

A small set of pages receives a large proportion of accesses while other pages are accessed less frequently.

This models temporal and spatial locality commonly observed in memory systems.

---

## Benchmark Configuration

The final comparative benchmark uses:

```text
Cache Capacity : 32 pages
Address Space  : 100 pages
Accesses       : 1000 per workload
Evaluation     : 5 independent seeds
Seeds          : 101, 202, 303, 404, 505
```

Every cache policy is evaluated using the same generated workload for each seed.

Results are reported using:

- Mean hit rate
- Standard deviation
- Minimum hit rate
- Maximum hit rate

---

## Benchmark Results

### Mean Hit Rate Across Five Seeds

| Policy | Sequential | Random | Looping | Locality |
|---|---:|---:|---:|---:|
| FIFO | 0.00% | 30.98% | 99.00% | 67.32% |
| LRU | 0.00% | **31.50%** | 99.00% | 76.58% |
| Clock | 0.00% | 31.32% | 99.00% | 73.32% |
| ARC | 0.00% | 31.00% | 99.00% | **80.64%** |
| **MemAI** | **28.00%** | 30.28% | 99.00% | **77.36%** |

---

## Key Results

### Locality Workload

MemAI achieved:

```text
77.36% mean hit rate
```

Compared with:

```text
LRU   : 76.58%
Clock : 73.32%
FIFO  : 67.32%
```

MemAI outperformed LRU by:

```text
+0.78 percentage points
```

ARC remained the strongest policy on average with a hit rate of:

```text
80.64%
```

On evaluation seed `101`, MemAI achieved:

```text
MemAI : 81.40%
ARC   : 80.40%
LRU   : 76.10%
```

MemAI achieved the highest locality hit rate for that individual evaluation seed.

### Sequential Workload

MemAI achieved:

```text
28.00% mean hit rate
```

The classical policies evaluated in this simulator achieved:

```text
FIFO  : 0.00%
LRU   : 0.00%
Clock : 0.00%
ARC   : 0.00%
```

This result indicates that the learned policy can avoid some of the cyclic scan thrashing observed in the evaluated traditional policies.

### Looping Workload

All policies achieved:

```text
99.00% mean hit rate
```

The working set fits within the cache after the initial misses.

### Random Workload

MemAI achieved:

```text
30.28%
```

LRU achieved the highest mean hit rate:

```text
31.50%
```

Random workloads remain a limitation of the current learned policy.

---

## Training Behavior

During delayed eviction-credit training, the agent showed changes in eviction behavior as exploration decreased.

Early in training:

```text
Episode 1
Hit Rate      : 53.48%
Early Reuse   : 1442
Survived      : 1305
```

Later in training:

```text
Episode 29
Hit Rate      : 56.18%
Early Reuse   : 905
Survived      : 1680
```

The reduction in early reuse events and increase in horizon-surviving evictions indicate that the policy increasingly selected pages that were not required in the near future.

Training loss alone is not treated as a direct cache-performance metric. Final policy quality is evaluated using independent multi-seed benchmark workloads.

---

## Project Structure

```text
MemAI/
|
|-- agent.py
|   |-- DuelingDDQNAgent
|   `-- ReplayMemory
|
|-- agent/
|   |-- train.py
|   `-- evaluate.py
|
|-- cache/
|   |-- base.py
|   |-- fifo.py
|   |-- lru.py
|   |-- clock.py
|   |-- arc.py
|   |-- cache_env.py
|   |-- memai_policy.py
|   `-- memai_weights.pt
|
|-- metrics/
|   |-- collector.py
|   `-- visualizer.py
|
|-- workload/
|   `-- generator.py
|
|-- tests/
|   |-- test_agent.py
|   |-- test_cache_env.py
|   |-- test_fifo.py
|   |-- test_lru.py
|   |-- test_clock.py
|   |-- test_arc.py
|   |-- test_collector.py
|   |-- test_visualizer.py
|   `-- test_workloads.py
|
|-- results/
|   `-- graphs/
|
|-- benchmark_multiseed.py
|-- main.py
|-- requirements.txt
`-- README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tirfan022/MemAI.git
cd MemAI
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training MemAI

Run:

```bash
python agent/train.py
```

The trained model weights are saved to:

```text
cache/memai_weights.pt
```

Training uses:

- Epsilon-greedy exploration
- Experience replay
- Dueling DDQN
- Double DQN target calculation
- Target network synchronization
- AdamW optimization
- Smooth L1 loss
- Gradient clipping
- Delayed eviction-specific reward
- Event-driven transitions

---

## Running the Simulator

Run:

```bash
python main.py
```

The simulator executes every cache policy across all supported workloads.

CSV metrics are exported to:

```text
results/
```

Generated benchmark graphs are saved to:

```text
results/graphs/
```

---

## Running the Multi-Seed Benchmark

Run:

```bash
python benchmark_multiseed.py
```

The benchmark evaluates all policies across five independent workload seeds and reports:

```text
Mean
Standard Deviation
Minimum
Maximum
```

---

## Running Tests

Run the complete test suite:

```bash
pytest -v
```

Current project status:

```text
191 tests passed
```

The test suite validates:

- Cache policy behavior
- Cache environment state transitions
- Dynamic observation dimensions
- Absence of future-trace observation leakage
- Cache hit and miss handling
- Eviction slot behavior
- Frequency tracking
- LRU tracking
- GRU input processing
- Neural network output dimensions
- Gradient propagation
- Replay memory behavior
- Model serialization
- Metrics collection
- CSV validation
- Graph generation
- Workload generation

---

## Design Decisions

### Why Reinforcement Learning?

Cache replacement is a sequential decision problem.

An eviction decision changes the cache state and can affect future hit and miss behavior.

Reinforcement learning provides a framework for learning these decisions from workload interactions.

### Why Dueling DQN?

The quality of a cache state and the relative advantage of individual eviction actions are related but distinct concepts.

The dueling architecture learns these components separately.

### Why Double DQN?

Standard DQN can overestimate Q-values.

Double DQN separates action selection from target evaluation to reduce this overestimation.

### Why GRU?

Recent memory accesses form a sequence.

GRUs provide a lightweight recurrent architecture for encoding short-term temporal patterns.

### Why Event-Driven Transitions?

Eviction actions occur only when the cache is full and a miss occurs.

Training on eviction events aligns reinforcement learning transitions with actual agent decisions.

### Why Delayed Eviction Credit?

General hit and miss rewards create ambiguous credit assignment.

Tracking the exact evicted page allows the training system to directly evaluate whether that page was required again soon.

---

## Limitations

The current implementation has several limitations.

- MemAI does not outperform every classical policy on every workload.
- ARC remains stronger on average for the evaluated locality workloads.
- LRU performs better on random workloads.
- The reward horizon is currently fixed at 50 accesses.
- Training workloads are synthetic.
- The simulator does not yet use real operating-system page traces.
- Neural inference introduces more computational overhead than traditional heuristics.
- The current model is trained for a fixed cache capacity configuration.
- Benchmark results are specific to the implemented simulator and workload generator.

These limitations are intentionally documented to keep benchmark claims reproducible and technically accurate.

---

## Future Work

Potential extensions include:

- Real memory trace evaluation
- SPEC CPU workload integration
- PARSEC trace evaluation
- Adaptive eviction reward horizons
- Prioritized experience replay
- N-step DDQN targets
- Distributional reinforcement learning
- Transformer-based access-history encoding
- Dynamic cache-capacity training
- Inference latency benchmarking
- Memory overhead analysis
- Policy distillation into a lightweight eviction model
- Hybrid ARC and reinforcement learning policies

---

## Technologies Used

- Python
- PyTorch
- Gymnasium
- NumPy
- Pandas
- Matplotlib
- Pytest
- Reinforcement Learning
- Deep Q-Learning

---

## Experimental Integrity

MemAI benchmark results are reported across five independent evaluation seeds.

The learned policy is compared against FIFO, LRU, Clock, and ARC using identical workload traces for each evaluation seed.

The agent observation does not expose future memory references.

Delayed eviction rewards are resolved from accesses observed as the simulation progresses rather than by directly providing future access information to the neural network.

Negative benchmark results and limitations are retained in the project documentation.

---

## Author

**Md Kamran**

Software Engineering  
Delhi Technological University

GitHub: [kamranfarhan463-prog](https://github.com/kamranfarhan463-prog)

LinkedIn: [Md Kamran](https://www.linkedin.com/in/md-kamran-760781309/)


---

## License

This project is intended for educational, research, and experimental purposes.

---

## Acknowledgements

MemAI was developed as an exploration of reinforcement learning for adaptive cache replacement and memory-management simulation.

The project draws on concepts from cache replacement algorithms, temporal sequence modeling, deep Q-learning, and event-driven reinforcement learning.