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
