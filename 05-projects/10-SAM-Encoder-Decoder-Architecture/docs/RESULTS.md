# Results

## Validation Status

The project source, JSON configuration, CSV exports, JSON report, and visualization are validated locally.

The default suite evaluates 1, 3, 5, and 10 prompts with three trials per scenario, for 12 total trials.

## Validated Summary

| Prompts | Without cache | With cache | Speedup | Reduction |
|---:|---:|---:|---:|---:|
| 1 | 0.0610 s | 0.0603 s | 1.01x | 1.1% |
| 3 | 0.1809 s | 0.0816 s | 2.22x | 54.9% |
| 5 | 0.3017 s | 0.1012 s | 2.98x | 66.5% |
| 10 | 0.6044 s | 0.1528 s | 3.96x | 74.7% |

Validation checks:

- Python syntax: passed
- Scenarios: 4
- Trial rows: 12
- Summary rows: 4
- JSON report: valid
- PNG chart: valid
- Runtime errors: 0

Exact measured values are stored in `results/` and may vary slightly because operating-system scheduling affects short `sleep` intervals.
