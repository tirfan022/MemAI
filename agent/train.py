import os
import sys

root_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if root_path not in sys.path:
    sys.path.insert(0, root_path)


import numpy as np
import torch
import torch.nn as nn

import agent as agent_file

from cache.cache_env import CacheEnv


DuelingDDQNAgent = (
    agent_file.DuelingDDQNAgent
)

ReplayMemory = (
    agent_file.ReplayMemory
)


MODEL_FILE = os.path.join(
    root_path,
    "cache",
    "memai_weights.pt",
)


CAPACITY = 32
EPISODES = 30

BATCH_SIZE = 64

GAMMA = 0.99

LEARNING_RATE = 1e-4

TARGET_UPDATE_STEPS = 500

EVICTION_HORIZON = 50


def generate_training_workload(
    capacity,
    seed,
):
    rng = np.random.default_rng(seed)

    blocks = []


    # Dense working-set workload

    dense_trace = rng.integers(
        0,
        capacity + 4,
        size=1500,
    ).tolist()

    blocks.append(dense_trace)


    # Locality workload

    pages = np.arange(
        capacity * 2
    )

    probabilities = np.array(
        [
            (
                0.60 / 6
                if page < 6
                else 0.40 / (
                    len(pages) - 6
                )
            )
            for page in pages
        ],
        dtype=np.float64,
    )

    probabilities /= probabilities.sum()

    locality_trace = rng.choice(
        pages,
        size=1500,
        p=probabilities,
    ).tolist()

    blocks.append(locality_trace)


    # Uniform random workload

    random_trace = rng.integers(
        0,
        100,
        size=1500,
    ).tolist()

    blocks.append(random_trace)


    # Cyclic scan workload

    scan_pages = np.arange(
        capacity * 2
    )

    scan_trace = np.resize(
        scan_pages,
        1500,
    ).tolist()

    blocks.append(scan_trace)


    # Shuffle workload types,
    # but preserve access order
    # inside every workload.

    rng.shuffle(blocks)


    final_trace = []

    for block in blocks:
        final_trace.extend(block)


    return final_trace


def calculate_eviction_reward(
    reuse_distance,
):
    """
    Reward an eviction according to when the
    evicted page is actually requested again.

    No future trace is inspected.

    reuse_distance:
        Number of observed accesses since
        the page was evicted.

    None:
        Page survived the complete horizon
        without being requested.
    """

    if reuse_distance is None:
        return 1.0


    normalized_distance = (
        reuse_distance
        / EVICTION_HORIZON
    )


    # Immediate reuse is strongly bad.
    #
    # Distance 1  -> approximately -0.96
    # Distance 25 -> 0.00
    # Distance 50 -> +1.00

    reward = (
        2.0 * normalized_distance
    ) - 1.0


    return float(
        np.clip(
            reward,
            -1.0,
            1.0,
        )
    )


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    memory,
):
    if len(memory) < BATCH_SIZE:
        return None


    batch = memory.sample(
        BATCH_SIZE
    )


    (
        states,
        actions,
        rewards,
        next_states,
        discounts,
    ) = zip(*batch)


    states_tensor = torch.tensor(
        np.array(states),
        dtype=torch.float32,
    )


    actions_tensor = torch.tensor(
        actions,
        dtype=torch.long,
    ).unsqueeze(1)


    rewards_tensor = torch.tensor(
        rewards,
        dtype=torch.float32,
    )


    next_states_tensor = torch.tensor(
        np.array(next_states),
        dtype=torch.float32,
    )


    discounts_tensor = torch.tensor(
        discounts,
        dtype=torch.float32,
    )


    current_q = policy_net(
        states_tensor
    ).gather(
        1,
        actions_tensor,
    ).squeeze(1)


    with torch.no_grad():

        next_actions = policy_net(
            next_states_tensor
        ).argmax(
            dim=1,
            keepdim=True,
        )


        next_q = target_net(
            next_states_tensor
        ).gather(
            1,
            next_actions,
        ).squeeze(1)


        target_q = (
            rewards_tensor
            + discounts_tensor * next_q
        )


    loss = nn.SmoothL1Loss()(
        current_q,
        target_q,
    )


    optimizer.zero_grad()

    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=1.0,
    )


    optimizer.step()


    return loss.item()


def try_store_experience(
    experience,
    memory,
):
    """
    Store an experience only after:

    1. Its eviction reward is known.
    2. Its next decision state is known.
    """

    if experience["stored"]:
        return False


    if experience["reward"] is None:
        return False


    if experience["next_state"] is None:
        return False


    memory.push(
        (
            experience["state"],
            experience["action"],
            experience["reward"],
            experience["next_state"],
            experience["discount"],
        )
    )


    experience["stored"] = True

    return True


def train_agent(
    capacity=CAPACITY,
    episodes=EPISODES,
):

    print(
        "[MemAI] Starting delayed "
        "eviction-credit DDQN training..."
    )


    policy_net = DuelingDDQNAgent(
        action_dim=capacity
    )


    target_net = DuelingDDQNAgent(
        action_dim=capacity
    )


    target_net.load_state_dict(
        policy_net.state_dict()
    )


    target_net.eval()


    optimizer = torch.optim.AdamW(
        policy_net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )


    memory = ReplayMemory(
        max_len=50000
    )


    optimization_steps = 0


    print(
        f"[MemAI] Training across "
        f"{episodes} episodes..."
    )


    for episode in range(episodes):

        seed = 1000 + episode


        training_trace = (
            generate_training_workload(
                capacity,
                seed,
            )
        )


        env = CacheEnv(
            capacity=capacity
        )


        observation, _ = env.reset()


        progress = (
            episode
            / max(
                episodes - 1,
                1,
            )
        )


        epsilon = max(
            0.10,
            1.0 - 0.90 * progress,
        )


        hits = 0

        eviction_decisions = 0

        losses = []

        resolved_early = 0

        survived_horizon = 0


        active_experiences = []

        previous_decision = None


        for access_index, page in enumerate(
            training_trace
        ):

            # ──────────────────────────────
            # RESOLVE DELAYED EVICTION CREDIT
            # ──────────────────────────────

            for experience in active_experiences:

                if experience["reward"] is not None:
                    continue


                age = (
                    access_index
                    - experience["eviction_step"]
                )


                if (
                    page
                    == experience["evicted_page"]
                    and age > 0
                ):

                    experience["reward"] = (
                        calculate_eviction_reward(
                            age
                        )
                    )

                    resolved_early += 1


                elif age >= EVICTION_HORIZON:

                    experience["reward"] = (
                        calculate_eviction_reward(
                            None
                        )
                    )

                    survived_horizon += 1


            is_hit = (
                page in env.slots
            )


            has_empty_slot = (
                -1 in env.slots
            )


            decision_event = (
                not is_hit
                and not has_empty_slot
            )


            # A new eviction decision defines
            # the next RL state of the previous
            # eviction decision.

            if decision_event:

                if previous_decision is not None:

                    decision_gap = (
                        access_index
                        - previous_decision[
                            "eviction_step"
                        ]
                    )


                    previous_decision[
                        "next_state"
                    ] = observation.copy()


                    previous_decision[
                        "discount"
                    ] = (
                        GAMMA ** decision_gap
                    )


                    try_store_experience(
                        previous_decision,
                        memory,
                    )


            state_tensor = torch.tensor(
                observation,
                dtype=torch.float32,
            ).unsqueeze(0)


            if is_hit:

                action = 0


            elif has_empty_slot:

                action = int(
                    np.where(
                        env.slots == -1
                    )[0][0]
                )


            else:

                eviction_decisions += 1


                if (
                    np.random.random()
                    < epsilon
                ):

                    action = int(
                        np.random.randint(
                            0,
                            capacity,
                        )
                    )


                else:

                    with torch.no_grad():

                        q_values = policy_net(
                            state_tensor
                        )


                        action = int(
                            q_values.argmax(
                                dim=1
                            ).item()
                        )


                evicted_page = int(
                    env.slots[action]
                )


                experience = {

                    "state": observation.copy(),

                    "action": action,

                    "evicted_page": evicted_page,

                    "eviction_step": access_index,

                    "reward": None,

                    "next_state": None,

                    "discount": 0.0,

                    "stored": False,

                }


                active_experiences.append(
                    experience
                )


                previous_decision = experience


            (
                next_observation,
                _,
                _,
                _,
                info,
            ) = env.step(
                action,
                page,
            )


            if info["hit"]:
                hits += 1


            observation = next_observation


            # Rewards may have been resolved
            # before their next decision state
            # became available, or vice versa.

            newly_stored = 0


            for experience in active_experiences:

                stored = try_store_experience(
                    experience,
                    memory,
                )


                if stored:
                    newly_stored += 1


            # Optimize once for every newly
            # completed decision transition.

            for _ in range(newly_stored):

                loss = optimize_model(
                    policy_net,
                    target_net,
                    optimizer,
                    memory,
                )


                if loss is None:
                    continue


                losses.append(loss)

                optimization_steps += 1


                if (
                    optimization_steps
                    % TARGET_UPDATE_STEPS
                    == 0
                ):

                    target_net.load_state_dict(
                        policy_net.state_dict()
                    )


            # Remove experiences that are
            # completely stored.

            active_experiences = [
                experience
                for experience
                in active_experiences
                if not experience["stored"]
            ]


        # ──────────────────────────────
        # TERMINAL EPISODE CLEANUP
        # ──────────────────────────────

        for experience in active_experiences:

            if experience["reward"] is None:

                age = (
                    len(training_trace)
                    - experience[
                        "eviction_step"
                    ]
                )


                if age >= EVICTION_HORIZON:

                    experience["reward"] = (
                        calculate_eviction_reward(
                            None
                        )
                    )

                else:

                    # Episode ended before enough
                    # evidence was observed.
                    #
                    # Neutral terminal reward.

                    experience["reward"] = 0.0


            if experience["next_state"] is None:

                experience["next_state"] = (
                    observation.copy()
                )

                experience["discount"] = 0.0


            stored = try_store_experience(
                experience,
                memory,
            )


            if stored:

                loss = optimize_model(
                    policy_net,
                    target_net,
                    optimizer,
                    memory,
                )


                if loss is not None:

                    losses.append(loss)

                    optimization_steps += 1


        hit_rate = (
            hits
            / len(training_trace)
        ) * 100


        mean_loss = (
            np.mean(losses)
            if losses
            else 0.0
        )


        print(
            f"Episode "
            f"{episode + 1:2d}/"
            f"{episodes} | "
            f"Seed: {seed} | "
            f"Epsilon: {epsilon:.2f} | "
            f"Hit Rate: {hit_rate:6.2f}% | "
            f"Decisions: "
            f"{eviction_decisions:4d} | "
            f"Early Reuse: "
            f"{resolved_early:4d} | "
            f"Survived: "
            f"{survived_horizon:4d} | "
            f"Loss: {mean_loss:.5f}"
        )


    os.makedirs(
        os.path.dirname(MODEL_FILE),
        exist_ok=True,
    )


    torch.save(
        policy_net.state_dict(),
        MODEL_FILE,
    )


    print(
        "[MemAI] Delayed eviction-credit "
        "training complete."
    )


    print(
        f"[MemAI] Model saved to: "
        f"{MODEL_FILE}"
    )


if __name__ == "__main__":
    train_agent()