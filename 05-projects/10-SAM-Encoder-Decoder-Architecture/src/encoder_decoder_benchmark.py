"""Benchmark image-embedding caching in a promptable encoder-decoder pipeline.

This is a controlled architecture simulation. It does not run a SAM checkpoint
or claim production inference performance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from time import perf_counter, sleep

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    prompt_counts: list[int]
    trials_per_scenario: int
    image_encoder_delay_seconds: float
    prompt_interaction_delay_seconds: float
    random_seed: int


@dataclass(frozen=True)
class Trial:
    prompt_count: int
    trial: int
    without_cache_seconds: float
    with_cache_seconds: float

    @property
    def seconds_saved(self) -> float:
        return self.without_cache_seconds - self.with_cache_seconds

    @property
    def speedup(self) -> float:
        return self.without_cache_seconds / self.with_cache_seconds


def load_config(path: Path) -> Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = Config(**payload)
    if not config.prompt_counts or any(value < 1 for value in config.prompt_counts):
        raise ValueError("prompt_counts must contain positive integers")
    if config.trials_per_scenario < 1:
        raise ValueError("trials_per_scenario must be positive")
    if config.image_encoder_delay_seconds < 0 or config.prompt_interaction_delay_seconds < 0:
        raise ValueError("delays must be non-negative")
    return config


def run_without_cache(config: Config, prompt_count: int) -> float:
    start = perf_counter()
    for _ in range(prompt_count):
        sleep(config.image_encoder_delay_seconds)
        sleep(config.prompt_interaction_delay_seconds)
    return perf_counter() - start


def run_with_cache(config: Config, prompt_count: int) -> float:
    start = perf_counter()
    sleep(config.image_encoder_delay_seconds)
    for _ in range(prompt_count):
        sleep(config.prompt_interaction_delay_seconds)
    return perf_counter() - start


def run_trials(config: Config) -> list[Trial]:
    trials: list[Trial] = []
    for prompt_count in config.prompt_counts:
        for trial_number in range(1, config.trials_per_scenario + 1):
            trials.append(
                Trial(
                    prompt_count=prompt_count,
                    trial=trial_number,
                    without_cache_seconds=run_without_cache(config, prompt_count),
                    with_cache_seconds=run_with_cache(config, prompt_count),
                )
            )
    return trials


def summarize(trials: list[Trial]) -> list[dict[str, float | int]]:
    rows = []
    for prompt_count in sorted({trial.prompt_count for trial in trials}):
        group = [trial for trial in trials if trial.prompt_count == prompt_count]
        no_cache = statistics.mean(trial.without_cache_seconds for trial in group)
        cached = statistics.mean(trial.with_cache_seconds for trial in group)
        rows.append(
            {
                "prompt_count": prompt_count,
                "trials": len(group),
                "mean_without_cache_seconds": no_cache,
                "mean_with_cache_seconds": cached,
                "mean_seconds_saved": no_cache - cached,
                "speedup": no_cache / cached,
                "time_reduction_percent": (1 - cached / no_cache) * 100,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_chart(path: Path, summary: list[dict]) -> None:
    prompt_counts = [row["prompt_count"] for row in summary]
    no_cache = [row["mean_without_cache_seconds"] for row in summary]
    cached = [row["mean_with_cache_seconds"] for row in summary]
    positions = range(len(prompt_counts))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar([x - width / 2 for x in positions], no_cache, width, label="Without caching", color="#d9534f")
    axis.bar([x + width / 2 for x in positions], cached, width, label="With caching", color="#2ca02c")
    axis.set_xticks(list(positions), [str(value) for value in prompt_counts])
    axis.set_xlabel("Prompts applied to the same image")
    axis.set_ylabel("Mean total time (seconds)")
    axis.set_title("Encoder-Decoder Caching Scalability")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "config/benchmark.json")
    parser.add_argument("--project-root", type=Path, default=project_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trials = run_trials(config)
    summary = summarize(trials)
    results = args.project_root / "results"

    trial_rows = [
        {
            **asdict(trial),
            "seconds_saved": trial.seconds_saved,
            "speedup": trial.speedup,
        }
        for trial in trials
    ]
    write_csv(results / "csv/benchmark_trials.csv", trial_rows)
    write_csv(results / "csv/caching_summary.csv", summary)
    save_chart(results / "figures/caching_scalability.png", summary)

    report = {
        "project": "10-SAM-Encoder-Decoder-Architecture",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "controlled architecture simulation; not real SAM inference",
        "configuration": asdict(config),
        "scenario_count": len(summary),
        "trial_count": len(trials),
        "summary": summary,
    }
    json_path = results / "json/benchmark_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Project 10 — Encoder-Decoder Caching Benchmark")
    print(f"Scenarios: {len(summary)}")
    print(f"Trials:    {len(trials)}")
    for row in summary:
        print(
            f"{row['prompt_count']:>2} prompt(s): "
            f"{row['speedup']:.2f}x speedup, "
            f"{row['time_reduction_percent']:.1f}% reduction"
        )
    print(f"Results:   {results}")
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
