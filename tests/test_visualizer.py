import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))



from metrics.visualizer import MetricsVisualizer

viz = MetricsVisualizer()

csv_files = {
    "FIFO": "results/fifo.csv",
    "LRU": "results/lru.csv"
}

viz.compare_hit_rates(csv_files)
viz.save_graph("results/hit_rates.png")

print("Graph generated successfully")