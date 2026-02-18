"""
utils.py: Timing, multi-seed aggregation, and experiment utilities.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager


# =============================================================================
# Timing
# =============================================================================

@dataclass
class TimingRecord:
    """Records per-step timing for an experiment."""
    steps: Dict[str, float] = field(default_factory=dict)
    _start_times: Dict[str, float] = field(default_factory=dict, repr=False)

    def start(self, step_name: str):
        self._start_times[step_name] = time.time()

    def stop(self, step_name: str):
        if step_name in self._start_times:
            elapsed = time.time() - self._start_times[step_name]
            self.steps[step_name] = elapsed
            del self._start_times[step_name]
            return elapsed
        return 0.0

    @contextmanager
    def measure(self, step_name: str):
        """Context manager for timing a step."""
        self.start(step_name)
        try:
            yield
        finally:
            elapsed = self.stop(step_name)
            print(f"  [{step_name}] {elapsed:.1f}s")

    @property
    def total(self) -> float:
        return sum(self.steps.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "total_seconds": self.total,
        }

    def print_summary(self):
        print("\nTiming Summary:")
        for step, elapsed in self.steps.items():
            print(f"  {step}: {elapsed:.1f}s")
        print(f"  TOTAL: {self.total:.1f}s")


# =============================================================================
# Multi-Seed Aggregation
# =============================================================================

@dataclass
class AggregatedMetrics:
    """Aggregated metrics across seeds with mean/std."""
    mean: float
    std: float
    values: List[float]
    n_seeds: int

    def __str__(self):
        return f"{self.mean:.4f} ± {self.std:.4f}"


def aggregate_results(
    results_by_seed: Dict[int, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, AggregatedMetrics]]:
    """
    Aggregate experiment results across multiple seeds.

    Args:
        results_by_seed: {seed: {method_name: {metric_name: value}}}

    Returns:
        {method_name: {metric_name: AggregatedMetrics}}
    """
    # Collect all methods and metrics
    all_methods = set()
    all_metrics = set()
    for seed_results in results_by_seed.values():
        for method, metrics in seed_results.items():
            all_methods.add(method)
            all_metrics.update(metrics.keys())

    aggregated = {}
    for method in sorted(all_methods):
        aggregated[method] = {}
        for metric in sorted(all_metrics):
            values = []
            for seed, seed_results in results_by_seed.items():
                if method in seed_results and metric in seed_results[method]:
                    values.append(seed_results[method][metric])

            if values:
                aggregated[method][metric] = AggregatedMetrics(
                    mean=np.mean(values),
                    std=np.std(values),
                    values=values,
                    n_seeds=len(values),
                )

    return aggregated


def print_aggregated_results(
    aggregated: Dict[str, Dict[str, AggregatedMetrics]],
    metrics_to_show: List[str] = None,
):
    """Print aggregated results as a table."""
    if metrics_to_show is None:
        metrics_to_show = ["auroc", "auprc", "precision_at_k"]

    # Header
    header = f"{'Method':<35}" + "".join(f"{m:>20}" for m in metrics_to_show)
    print(header)
    print("-" * len(header))

    # Sort by first metric (descending)
    methods = sorted(
        aggregated.keys(),
        key=lambda m: aggregated[m].get(metrics_to_show[0], AggregatedMetrics(0, 0, [], 0)).mean,
        reverse=True,
    )

    for method in methods:
        row = f"{method:<35}"
        for metric in metrics_to_show:
            if metric in aggregated[method]:
                agg = aggregated[method][metric]
                row += f"{str(agg):>20}"
            else:
                row += f"{'N/A':>20}"
        print(row)
