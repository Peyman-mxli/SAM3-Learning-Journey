# Input Assets

The default practical input is:

```text
vehicles.mp4
```

When the file is missing, `sam3_video_segmentation.py` downloads the official Supervision example video from:

```text
https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

A custom video can be supplied with:

```bash
python sam3_video_segmentation.py \
  --mode full \
  --sam-model /path/to/sam3.pt \
  --input /path/to/custom_video.mp4
```

Large input videos do not need to be committed to the repository.


---

## Football-Pitch Homography Input

The repository includes an original perspective-pitch visual:

[football_pitch_perspective.svg](./football_pitch_perspective.svg)

It is used to document the four-point selection concept from the class. The Python implementation can also accept a real football-field photograph through `--input`.
