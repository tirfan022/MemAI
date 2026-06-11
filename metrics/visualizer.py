"""
metrics/visualizer.py
---------------------
Reads CSV files produced by MetricsCollector and generates
comparison bar charts across all cache replacement policies.

Dependencies:
    pip install pandas matplotlib
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List


class MetricsVisualizer:
    """
    Loads per-policy CSV files and produces comparison bar charts.

    Each CSV file must have been exported by MetricsCollector and
    contain at least these columns:
        step, hit_rate, cumulative_hits, cumulative_misses, evictions

    Typical usage:
        viz = MetricsVisualizer()
        csv_files = {
            "FIFO":  "results/fifo_locality.csv",
            "LRU":   "results/lru_locality.csv",
            "Clock": "results/clock_locality.csv",
            "ARC":   "results/arc_locality.csv",
        }
        viz.compare_hit_rates(csv_files)
        viz.save_graph("results/hit_rates.png")
    """

    # ── Colour palette — one colour per policy, consistent across all charts ──
    POLICY_COLORS: Dict[str, str] = {
        "FIFO":  "#4C72B0",
        "LRU":   "#DD8452",
        "Clock": "#55A868",
        "ARC":   "#C44E52",
    }
    DEFAULT_COLOR = "#8172B2"   # fallback for any unlisted policy name

    def __init__(self):
        self._last_fig: plt.Figure | None = None   # holds the most recently created figure

    # ──────────────────────────────────────────
    #  Data loading
    # ──────────────────────────────────────────

    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Load a single MetricsCollector CSV file into a DataFrame.

        Validates that the file exists and contains the expected columns.
        The final row of the CSV represents the end-of-simulation state
        and is used for all summary bar charts.

        Args:
            path (str): Absolute or relative path to the CSV file.

        Returns:
            pd.DataFrame: Full step-by-step simulation history.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            ValueError:        If required columns are missing from the file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)

        required = {"step", "hit_rate", "hits", "misses", "evictions"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV '{path}' is missing columns: {missing}")

        return df

    def _load_all(self, csv_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple CSV files keyed by policy name.

        Args:
            csv_files (Dict[str, str]): Mapping of policy name → CSV path.
                Example: {"FIFO": "results/fifo.csv", "LRU": "results/lru.csv"}

        Returns:
            Dict[str, pd.DataFrame]: Mapping of policy name → DataFrame.
        """
        return {policy: self.load_csv(path) for policy, path in csv_files.items()}

    # ──────────────────────────────────────────
    #  Chart helpers
    # ──────────────────────────────────────────

    def _bar_colors(self, policy_names: List[str]) -> List[str]:
        """Return a list of colours matching the given policy names."""
        return [self.POLICY_COLORS.get(name, self.DEFAULT_COLOR) for name in policy_names]

    def _apply_bar_labels(self, ax: plt.Axes, bars, fmt: str = "{:.2f}"):
        """
        Place a value label above each bar.

        Args:
            ax  (plt.Axes): The axes containing the bars.
            bars:           The BarContainer returned by ax.bar().
            fmt (str):      Python format string for the label text.
        """
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + ax.get_ylim()[1] * 0.01,
                fmt.format(height),
                ha="center", va="bottom",
                fontsize=9, fontweight="bold"
            )

    def _style_axes(self, ax: plt.Axes, title: str, ylabel: str):
        """Apply consistent styling to an axes object."""
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Cache Policy", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # ──────────────────────────────────────────
    #  Public chart methods
    # ──────────────────────────────────────────

    def compare_hit_rates(self, csv_files: Dict[str, str]) -> plt.Figure:
        """
        Create a bar chart comparing the final hit rate of each policy.

        Hit rate = cumulative_hits / total_accesses at the last simulation step.
        Bars are labelled with their exact percentage value.

        Args:
            csv_files (Dict[str, str]): Mapping of policy name → CSV path.

        Returns:
            plt.Figure: The generated matplotlib Figure (also stored in
                        self._last_fig for use with save_graph()).
        """
        data   = self._load_all(csv_files)
        names  = list(data.keys())
        values = [df.iloc[-1]["hit_rate"] * 100 for df in data.values()]  # → percent

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, values, color=self._bar_colors(names),
                      edgecolor="white", linewidth=0.8, zorder=3)

        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        self._apply_bar_labels(ax, bars, fmt="{:.1f}%")
        self._style_axes(ax, "Hit Rate Comparison", "Hit Rate (%)")

        fig.tight_layout()
        self._last_fig = fig
        return fig

    def compare_misses(self, csv_files: Dict[str, str]) -> plt.Figure:
        """
        Create a bar chart comparing the total miss count of each policy.

        Uses the cumulative_misses value from the last row of each CSV,
        which represents the total misses over the entire simulation run.

        Args:
            csv_files (Dict[str, str]): Mapping of policy name → CSV path.

        Returns:
            plt.Figure: The generated matplotlib Figure.
        """
        data   = self._load_all(csv_files)
        names  = list(data.keys())
        values = [int(df.iloc[-1]["misses"]) for df in data.values()]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, values, color=self._bar_colors(names),
                      edgecolor="white", linewidth=0.8, zorder=3)

        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        self._apply_bar_labels(ax, bars, fmt="{:.0f}")
        self._style_axes(ax, "Total Misses Comparison", "Total Misses")

        fig.tight_layout()
        self._last_fig = fig
        return fig

    def compare_evictions(self, csv_files: Dict[str, str]) -> plt.Figure:
        """
        Create a bar chart comparing the total eviction count of each policy.

        Uses the evictions value from the last row of each CSV.
        Fewer evictions generally indicate a smarter replacement decision.

        Args:
            csv_files (Dict[str, str]): Mapping of policy name → CSV path.

        Returns:
            plt.Figure: The generated matplotlib Figure.
        """
        data   = self._load_all(csv_files)
        names  = list(data.keys())
        values = [int(df.iloc[-1]["evictions"]) for df in data.values()]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, values, color=self._bar_colors(names),
                      edgecolor="white", linewidth=0.8, zorder=3)

        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        self._apply_bar_labels(ax, bars, fmt="{:.0f}")
        self._style_axes(ax, "Total Evictions Comparison", "Total Evictions")

        fig.tight_layout()
        self._last_fig = fig
        return fig

    # ──────────────────────────────────────────
    #  Output
    # ──────────────────────────────────────────

    def save_graph(self, output_path: str):
        """
        Save the most recently generated figure to disk.

        The directory is created automatically if it does not exist.
        Must be called after one of the compare_*() methods.

        Args:
            output_path (str): File path for the saved image.
                               Extension determines format: .png, .pdf, .svg

        Raises:
            RuntimeError: If no figure has been generated yet.
        """
        if self._last_fig is None:
            raise RuntimeError(
                "No figure to save. Call compare_hit_rates(), "
                "compare_misses(), or compare_evictions() first."
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._last_fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[MetricsVisualizer] Graph saved → {output_path}")