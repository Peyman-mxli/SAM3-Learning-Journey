# Session 11 Practical — SAM 3 Video Segmentation

This practical section converts the original Session 11 notebook into a structured and reproducible video-segmentation workflow.

The source notebook is:

[05_b_segmentacion_sam_video.ipynb](../05_b_segmentacion_sam_video.ipynb)

---

## Practical Status

```text
Practical Structure:  ✅ Created
Source Notebook:      ✅ Preserved
Colab Execution:      ⏳ Pending
Output Validation:    ⏳ Pending
Video Outputs:        ⏳ Pending
Metrics:              ⏳ Pending
```

No execution results are reported here until the notebook has been run and validated in Google Colab.

---

## Planned Pipelines

### Pipeline A

```text
Input Video
    ↓
YOLOv8 Detection
    ↓
ByteTrack IDs
    ↓
SAM 3 Masks
    ↓
Mask + Label + Trace Annotation
    ↓
Annotated Video
```

### Pipeline B

```text
Input Video
    ↓
Text Prompts
    ↓
SAM3VideoSemanticPredictor
    ↓
Semantic Masks
    ↓
Annotated Video
```

---

## Planned Experiments

1. Full YOLO + ByteTrack + SAM 3 video segmentation
2. Segmentation restricted to a polygon zone
3. Dynamic mask opacity based on detection confidence
4. Mask-area analysis for tracked objects
5. Direct semantic video segmentation with text prompts
6. Tracker-ID filtering challenge

---

## Expected Outputs

The notebook defines these expected files:

```text
assets/
│
├── vehicles.mp4
├── vehicles_sam.mp4
├── vehicles_sam_zona.mp4
├── vehicles_sam_opacidad.mp4
├── vehicles_sam_areas.mp4
└── vehicles_texto.mp4
```

The files will be added or documented after successful execution and verification.

---

## Validation Checklist

```text
Dependencies installed                    ⏳
Google Drive mounted                      ⏳
SAM 3 checkpoint found                    ⏳
YOLOv8 loaded                             ⏳
ByteTrack initialized                     ⏳
Input video downloaded                    ⏳
Video metadata verified                   ⏳
YOLO + ByteTrack + SAM pipeline executed  ⏳
Tracker attributes transferred            ⏳
Zone-filtered pipeline executed            ⏳
Dynamic-opacity experiment executed        ⏳
Mask-area experiment executed              ⏳
Text-prompt video pipeline executed        ⏳
Output videos verified                     ⏳
Results documented                         ⏳
```

---

## Next Step

Run the original notebook in Google Colab, preserve the execution evidence, verify each generated output, and replace the pending markers only with confirmed results.
