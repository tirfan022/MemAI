import numpy as np
import torch

from agent.agent import DuelingDDQNAgent, ReplayMemory


ACTION_DIM = 32
OBS_DIM = (ACTION_DIM * 10) + 28


def create_state(batch_size=1):
    return torch.randn(
        batch_size,
        OBS_DIM,
        dtype=torch.float32,
    )


def test_agent_initialization():
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    assert model.action_dim == ACTION_DIM
    assert model.obs_dim == OBS_DIM


def test_agent_output_shape():
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    state = create_state()

    output = model(state)

    assert output.shape == (
        1,
        ACTION_DIM,
    )


def test_agent_batch_output_shape():
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    states = create_state(
        batch_size=8
    )

    output = model(states)

    assert output.shape == (
        8,
        ACTION_DIM,
    )


def test_agent_output_contains_no_nan():
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    state = create_state()

    output = model(state)

    assert not torch.isnan(
        output
    ).any()


def test_agent_selects_valid_action():
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    state = create_state()

    with torch.no_grad():
        q_values = model(state)

        action = torch.argmax(
            q_values,
            dim=1,
        ).item()

    assert 0 <= action < ACTION_DIM


def test_agent_deterministic_in_eval_mode():
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    model.eval()

    state = create_state()

    with torch.no_grad():
        output_1 = model(state)
        output_2 = model(state)

    assert torch.allclose(
        output_1,
        output_2,
    )


def test_model_parameters_receive_gradients():
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    state = create_state(
        batch_size=4
    )

    output = model(state)

    loss = output.mean()

    loss.backward()

    gradients_found = any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    assert gradients_found


def test_gru_receives_history_sequence():
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    state = torch.zeros(
        1,
        OBS_DIM,
        dtype=torch.float32,
    )

    slots_end = ACTION_DIM * 10

    history_start = slots_end + 3
    history_end = history_start + 10

    state[
        0,
        history_start:history_end
    ] = torch.arange(
        10,
        dtype=torch.float32,
    )

    output = model(state)

    assert output.shape == (
        1,
        ACTION_DIM,
    )


def test_replay_memory_initialization():
    memory = ReplayMemory(
        max_len=100
    )

    assert len(memory) == 0


def test_replay_memory_push():
    memory = ReplayMemory(
        max_len=100
    )

    transition = (
        np.zeros(OBS_DIM),
        0,
        1.0,
        np.ones(OBS_DIM),
    )

    memory.push(transition)

    assert len(memory) == 1


def test_replay_memory_sample():
    memory = ReplayMemory(
        max_len=100
    )

    for action in range(10):
        transition = (
            np.zeros(OBS_DIM),
            action,
            1.0,
            np.ones(OBS_DIM),
        )

        memory.push(transition)

    batch = memory.sample(5)

    assert len(batch) == 5


def test_replay_memory_max_length():
    memory = ReplayMemory(
        max_len=5
    )

    for action in range(10):
        transition = (
            np.zeros(OBS_DIM),
            action,
            1.0,
            np.ones(OBS_DIM),
        )

        memory.push(transition)

    assert len(memory) == 5


def test_model_state_dict_save_and_load(
    tmp_path
):
    model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    model.eval()

    state = create_state()

    with torch.no_grad():
        original_output = model(
            state
        )

    model_path = (
        tmp_path / "test_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    loaded_model = DuelingDDQNAgent(
        action_dim=ACTION_DIM
    )

    loaded_model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
        )
    )

    loaded_model.eval()

    with torch.no_grad():
        loaded_output = loaded_model(
            state
        )

    assert torch.allclose(
        original_output,
        loaded_output,
    )