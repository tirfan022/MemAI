import matplotlib

matplotlib.use("Agg")

import pandas as pd

from metrics.visualizer import MetricsVisualizer


def create_test_csv(
    path,
    hit_rate=60.0,
    hits=60,
    misses=40,
    evictions=8,
):
    dataframe = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "hit_rate": [
                20.0,
                40.0,
                hit_rate,
            ],
            "hits": [
                20,
                40,
                hits,
            ],
            "misses": [
                80,
                60,
                misses,
            ],
            "evictions": [
                2,
                5,
                evictions,
            ],
        }
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def test_compare_hit_rates(tmp_path):
    fifo_path = tmp_path / "fifo.csv"
    lru_path = tmp_path / "lru.csv"

    create_test_csv(
        fifo_path,
        hit_rate=60.0,
    )

    create_test_csv(
        lru_path,
        hit_rate=75.0,
    )

    viz = MetricsVisualizer()

    csv_files = {
        "FIFO": str(fifo_path),
        "LRU": str(lru_path),
    }

    viz.compare_hit_rates(
        csv_files
    )

    assert viz._last_fig is not None


def test_save_graph(tmp_path):
    fifo_path = tmp_path / "fifo.csv"
    lru_path = tmp_path / "lru.csv"

    create_test_csv(
        fifo_path,
        hit_rate=60.0,
    )

    create_test_csv(
        lru_path,
        hit_rate=75.0,
    )

    viz = MetricsVisualizer()

    csv_files = {
        "FIFO": str(fifo_path),
        "LRU": str(lru_path),
    }

    # Generate a figure first.
    viz.compare_hit_rates(
        csv_files
    )

    output_path = (
        tmp_path / "test_graph.png"
    )

    viz.save_graph(
        str(output_path)
    )

    assert output_path.exists()

    assert output_path.stat().st_size > 0


def test_compare_misses(tmp_path):
    fifo_path = tmp_path / "fifo.csv"
    lru_path = tmp_path / "lru.csv"

    create_test_csv(
        fifo_path,
        misses=40,
    )

    create_test_csv(
        lru_path,
        misses=25,
    )

    viz = MetricsVisualizer()

    csv_files = {
        "FIFO": str(fifo_path),
        "LRU": str(lru_path),
    }

    viz.compare_misses(
        csv_files
    )

    assert viz._last_fig is not None


def test_compare_evictions(tmp_path):
    fifo_path = tmp_path / "fifo.csv"
    lru_path = tmp_path / "lru.csv"

    create_test_csv(
        fifo_path,
        evictions=8,
    )

    create_test_csv(
        lru_path,
        evictions=5,
    )

    viz = MetricsVisualizer()

    csv_files = {
        "FIFO": str(fifo_path),
        "LRU": str(lru_path),
    }

    viz.compare_evictions(
        csv_files
    )

    assert viz._last_fig is not None


def test_missing_csv_raises_error(tmp_path):
    viz = MetricsVisualizer()

    missing_path = (
        tmp_path / "missing.csv"
    )

    csv_files = {
        "FIFO": str(missing_path)
    }

    try:
        viz.compare_hit_rates(
            csv_files
        )

        assert False

    except FileNotFoundError:
        assert True


def test_invalid_csv_columns_raise_error(
    tmp_path
):
    invalid_path = (
        tmp_path / "invalid.csv"
    )

    dataframe = pd.DataFrame(
        {
            "hit_rate": [50.0]
        }
    )

    dataframe.to_csv(
        invalid_path,
        index=False,
    )

    viz = MetricsVisualizer()

    csv_files = {
        "FIFO": str(invalid_path)
    }

    try:
        viz.compare_hit_rates(
            csv_files
        )

        assert False

    except ValueError:
        assert True


def test_save_without_figure_raises_error(
    tmp_path
):
    viz = MetricsVisualizer()

    output_path = (
        tmp_path / "graph.png"
    )

    try:
        viz.save_graph(
            str(output_path)
        )

        assert False

    except RuntimeError:
        assert True