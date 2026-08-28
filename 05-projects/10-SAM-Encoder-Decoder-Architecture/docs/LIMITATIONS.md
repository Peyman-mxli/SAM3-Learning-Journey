# Limitations

- This project is a controlled latency simulation.
- It does not load SAM, encode real images, or generate semantic masks.
- Delays model component cost; they are not GPU inference measurements.
- Results demonstrate scaling behavior, not a universal production speedup.
- `sleep` timing varies slightly with runtime scheduling.
- The benchmark does not evaluate mask quality, IoU, Dice, confidence, memory usage, or throughput under concurrency.

Use a real checkpoint and representative hardware before making deployment decisions.
