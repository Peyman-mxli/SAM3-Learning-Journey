"""Reusable, configurable benchmark for an educational caching simulation."""

import argparse
from dataclasses import dataclass
from time import perf_counter, sleep


@dataclass(frozen=True)
class BenchmarkResult:
    prompt_count: int
    without_cache_seconds: float
    with_cache_seconds: float

    @property
    def seconds_saved(self) -> float:
        return self.without_cache_seconds - self.with_cache_seconds

    @property
    def speedup(self) -> float:
        return self.without_cache_seconds / self.with_cache_seconds


def benchmark(prompt_count: int, encoder_delay: float, interaction_delay: float) -> BenchmarkResult:
    start = perf_counter()
    for _ in range(prompt_count):
        sleep(encoder_delay)
        sleep(interaction_delay)
    without_cache = perf_counter() - start

    start = perf_counter()
    sleep(encoder_delay)
    for _ in range(prompt_count):
        sleep(interaction_delay)
    with_cache = perf_counter() - start

    return BenchmarkResult(prompt_count, without_cache, with_cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=3, help="Number of prompts (default: 3)")
    parser.add_argument("--encoder-delay", type=float, default=2.0)
    parser.add_argument("--interaction-delay", type=float, default=0.1)
    args = parser.parse_args()
    if args.prompts < 1 or args.encoder_delay < 0 or args.interaction_delay < 0:
        parser.error("Prompts must be positive and delays must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    result = benchmark(args.prompts, args.encoder_delay, args.interaction_delay)
    print("REUSABLE CACHING BENCHMARK")
    print(f"Prompts:         {result.prompt_count}")
    print(f"Without cache:   {result.without_cache_seconds:.3f} seconds")
    print(f"With cache:      {result.with_cache_seconds:.3f} seconds")
    print(f"Time saved:      {result.seconds_saved:.3f} seconds")
    print(f"Speedup:         {result.speedup:.2f}x")
    print("Scope: controlled simulation; not real SAM inference.")


if __name__ == "__main__":
    main()
