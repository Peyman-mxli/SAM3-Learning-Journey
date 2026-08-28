# Input Assets

No external input file is required.

The practical uses the symbolic image name:

```text
high_resolution_image.jpg
```

The dummy image encoder does not read image pixels. It sleeps for two seconds and returns a simulated image embedding with shape:

```text
(1, 256, 64, 64)
```

This isolates the architectural and caching concepts from model installation and checkpoint inference.

