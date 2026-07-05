import numpy as np

from cache.cache_env import CacheEnv


def test_environment_initialization():
    env = CacheEnv(capacity=16)

    obs, info = env.reset()

    assert obs.shape == (env.obs_dim,)
    assert obs.dtype == np.float32

    assert len(env.slots) == 16
    assert np.all(env.slots == -1)

    assert env.global_step == 0
    assert env.total_hits == 0

    assert info == {}


def test_empty_cache_miss():
    env = CacheEnv(capacity=16)

    env.reset()

    _, reward, terminated, truncated, info = env.step(
        action=0,
        incoming_page=5,
    )

    assert info["hit"] is False

    assert env.slots[0] == 5

    assert env.global_step == 1

    assert env.total_hits == 0

    assert reward == 0.0

    assert terminated is False
    assert truncated is False


def test_cache_hit():
    env = CacheEnv(capacity=16)

    env.reset()

    env.step(
        action=0,
        incoming_page=5,
    )

    _, reward, _, _, info = env.step(
        action=0,
        incoming_page=5,
    )

    assert info["hit"] is True

    assert reward == 1.0

    assert env.total_hits == 1

    assert env.slots[0] == 5


def test_selected_slot_is_evicted():
    env = CacheEnv(capacity=16)

    env.reset()

    for page in range(16):
        env.step(
            action=page,
            incoming_page=page,
        )

    assert -1 not in env.slots

    old_page = env.slots[5]

    env.step(
        action=5,
        incoming_page=100,
    )

    assert old_page not in env.slots

    assert env.slots[5] == 100


def test_access_count_updates_on_hit():
    env = CacheEnv(capacity=16)

    env.reset()

    env.step(
        action=0,
        incoming_page=7,
    )

    assert env.access_counts[0] == 1

    env.step(
        action=0,
        incoming_page=7,
    )

    assert env.access_counts[0] == 2


def test_history_buffer_limit():
    env = CacheEnv(capacity=16)

    env.reset()

    for page in range(20):
        env.step(
            action=0,
            incoming_page=page,
        )

    assert len(env.history_buffer) == 10

    assert env.history_buffer == list(
        range(10, 20)
    )


def test_frequency_tracking():
    env = CacheEnv(capacity=16)

    env.reset()

    env.step(
        action=0,
        incoming_page=3,
    )

    env.step(
        action=0,
        incoming_page=3,
    )

    env.step(
        action=0,
        incoming_page=3,
    )

    assert env.freq_map[3] == 3


def test_lru_tracking():
    env = CacheEnv(capacity=16)

    env.reset()

    env.step(
        action=0,
        incoming_page=1,
    )

    env.step(
        action=1,
        incoming_page=2,
    )

    env.step(
        action=2,
        incoming_page=3,
    )

    assert env.lru_order == [
        1,
        2,
        3,
    ]

    env.step(
        action=0,
        incoming_page=1,
    )

    assert env.lru_order == [
        2,
        3,
        1,
    ]


def test_observation_does_not_depend_on_future_trace():
    trace_a = [
        10,
        20,
        999,
        40,
    ]

    trace_b = [
        10,
        500,
        600,
        700,
    ]

    env_a = CacheEnv(
        capacity=16,
        lookahead_trace=trace_a,
    )

    env_b = CacheEnv(
        capacity=16,
        lookahead_trace=trace_b,
    )

    env_a.reset()
    env_b.reset()

    obs_a, _, _, _, _ = env_a.step(
        action=0,
        incoming_page=10,
    )

    obs_b, _, _, _, _ = env_b.step(
        action=0,
        incoming_page=10,
    )

    assert np.array_equal(
        obs_a,
        obs_b,
    )


def test_observation_contains_no_nan():
    env = CacheEnv(capacity=16)

    obs, _ = env.reset()

    assert not np.isnan(obs).any()

    for page in range(100):
        obs, _, _, _, _ = env.step(
            action=page % 16,
            incoming_page=page,
        )

        assert not np.isnan(obs).any()


def test_observation_dimension_after_multiple_steps():
    env = CacheEnv(capacity=16)

    env.reset()

    for page in range(50):
        obs, _, _, _, _ = env.step(
            action=page % 16,
            incoming_page=page,
        )

        assert obs.shape == (env.obs_dim,)


def test_oracle_reward_does_not_change_observation():
    trace_a = [
        1,
        2,
        3,
        4,
        5,
    ]

    trace_b = [
        1,
        999,
        888,
        777,
        666,
    ]

    env_a = CacheEnv(
        capacity=16,
        lookahead_trace=trace_a,
    )

    env_b = CacheEnv(
        capacity=16,
        lookahead_trace=trace_b,
    )

    env_a.reset()
    env_b.reset()

    obs_a, _, _, _, _ = env_a.step(
        action=0,
        incoming_page=1,
    )

    obs_b, _, _, _, _ = env_b.step(
        action=0,
        incoming_page=1,
    )

    np.testing.assert_array_equal(
        obs_a,
        obs_b,
    )
def test_observation_dimension_changes_with_capacity():
    env_16 = CacheEnv(capacity=16)
    env_32 = CacheEnv(capacity=32)

    obs_16, _ = env_16.reset()
    obs_32, _ = env_32.reset()

    assert env_16.obs_dim == 188
    assert env_32.obs_dim == 348

    assert obs_16.shape == (188,)
    assert obs_32.shape == (348,)