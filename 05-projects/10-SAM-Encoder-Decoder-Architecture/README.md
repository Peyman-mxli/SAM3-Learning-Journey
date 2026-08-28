# Project 10 — SAM Encoder-Decoder Architecture

This project turns the encoder-decoder lesson into a reproducible caching and scalability benchmark for promptable segmentation systems.

> Project `10` corresponds to course-notes lesson `09`. The projects directory is one number ahead because it contains one additional earlier project.

## Objective

Measure how reusing a cached image embedding changes total interaction latency when multiple prompts are applied to the same image.

## Structure

```text
10-SAM-Encoder-Decoder-Architecture/
├── README.md
├── requirements.txt
├── config/
│   ├── README.md
│   └── benchmark.json
├── src/
│   ├── README.md
│   └── encoder_decoder_benchmark.py
├── results/
│   ├── README.md
│   ├── csv/
│   │   ├── README.md
│   │   ├── benchmark_trials.csv
│   │   └── caching_summary.csv
│   ├── json/
│   │   ├── README.md
│   │   └── benchmark_report.json
│   └── figures/
│       ├── README.md
│       └── caching_scalability.png
└── docs/
    ├── README.md
    ├── ARCHITECTURE.md
    ├── RESULTS.md
    └── LIMITATIONS.md
```

No `data/input/` folder is included because the pipeline does not read images. No SAM checkpoint or GPU is required.

## Pipeline

1. Load and validate JSON configuration.
2. Run cached and uncached trials for several prompt counts.
3. Record trial-level timings.
4. Calculate mean savings, speedup, and time reduction.
5. Export CSV and JSON analytics.
6. Generate a scalability chart.

## Run

```bash
python -m pip install -r requirements.txt
python src/encoder_decoder_benchmark.py
```

## Outputs

- Trial-level CSV
- Scenario-summary CSV
- Structured JSON report
- Caching scalability chart

## Validated Results

```text
Prompt count    Speedup    Time reduction
1               1.01x       1.1%
3               2.22x      54.9%
5               2.98x      66.5%
10              3.96x      74.7%
```

The completed validation produced 12 trial rows, 4 summary rows, one JSON report, and one PNG chart with zero runtime errors.

## Scope and Honesty

This project demonstrates architecture and caching behavior through controlled component delays. It does not claim real SAM inference performance or segmentation accuracy.

**Status: COMPLETE — source and generated artifacts validated.**
