from workload.generator import WorkloadGenerator


def test_generator_stores_values():
    gen = WorkloadGenerator(
        num_pages=100,
        num_accesses=1000,
        seed=42
    )

    assert gen.num_pages == 100
    assert gen.num_accesses == 1000
    assert gen.seed == 42

def test_sequential_length():
    gen = WorkloadGenerator(
        num_pages=10,
        num_accesses=20
    )

    workload = gen.generate_sequential()

    assert len(workload) == 20

def test_sequential_pattern():
    gen = WorkloadGenerator(
        num_pages=4,
        num_accesses=10
    )

    workload = gen.generate_sequential()

    assert workload == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]

def test_random_length():
    gen = WorkloadGenerator(
        num_pages=10,
        num_accesses=50,
        seed=42
    )

    workload = gen.generate_random()

    assert len(workload) == 50

def test_random_length():
    gen = WorkloadGenerator(
        num_pages=10,
        num_accesses=50,
        seed=42
    )

    workload = gen.generate_random()

    assert len(workload) == 50


def test_random_pages_in_range():
    gen = WorkloadGenerator(
        num_pages=10,
        num_accesses=100,
        seed=42
    )

    workload = gen.generate_random()

    assert all(0 <= page < 10 for page in workload)


def test_looping_length():
    gen = WorkloadGenerator(
        num_pages=10,
        num_accesses=12
    )

    workload = gen.generate_looping(loop_size=3)

    assert len(workload) == 12


def test_looping_pattern():
    gen = WorkloadGenerator(
        num_pages=10,
        num_accesses=9
    )

    workload = gen.generate_looping(loop_size=3)

    assert workload == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_locality_length():
    gen = WorkloadGenerator(
        num_pages=100,
        num_accesses=1000
    )

    workload = gen.generate_locality()

    assert len(workload) == 1000


def test_locality_pages_in_range():
    gen = WorkloadGenerator(
        num_pages=100,
        num_accesses=1000
    )

    workload = gen.generate_locality()

    assert all(0 <= page < 100 for page in workload)


def test_get_all_workloads_contains_all_keys():
    gen = WorkloadGenerator(
        num_pages=10,
        num_accesses=20
    )

    workloads = gen.get_all_workloads()

    assert "sequential" in workloads
    assert "random" in workloads
    assert "looping" in workloads
    assert "locality" in workloads


def test_get_all_workloads_lengths():
    gen = WorkloadGenerator(
        num_pages=10,
        num_accesses=20
    )

    workloads = gen.get_all_workloads()

    for workload in workloads.values():
        assert len(workload) == 20


def test_invalid_num_pages():
    import pytest

    with pytest.raises(ValueError):
        WorkloadGenerator(
            num_pages=0,
            num_accesses=10
        )


def test_invalid_num_accesses():
    import pytest

    with pytest.raises(ValueError):
        WorkloadGenerator(
            num_pages=10,
            num_accesses=0
        )