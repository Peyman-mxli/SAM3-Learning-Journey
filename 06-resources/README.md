# Resources

Professional references, datasets, assets, downloads, and external materials used throughout the SAM 3 Computer Vision Learning Journey.

Each resource is documented as a self-contained learning guide. A guide explains what the technology is, how it works, how to install it, how it is used in the course, and how to reproduce a practical example.

## Reference Guides

| Resource | Purpose | Course use |
|---|---|---|
| [Supervision](./references/01-Supervision/) | Model-agnostic utilities for representing, filtering, tracking, counting, and visualizing computer vision predictions | Detection, annotation, tracking, zones, counting, masks, and video analytics |
| [Supervision Annotators](./references/02-Supervision-Annotators/) | Visual components for boxes, labels, masks, markers, traces, zones, and counters | Session 03 annotation and visualization, plus later tracking and segmentation lessons |

## Resource Standard

Every complete guide should include:

- A clear conceptual explanation
- Architecture or workflow description
- Installation instructions
- Core classes and terminology
- A reproducible practical example
- Expected inputs and outputs
- Common errors and troubleshooting
- Professional implementation practices
- Course-session mapping
- Links to authoritative documentation

## Organization

```text
06-resources/
├── README.md
└── references/
    ├── 01-Supervision/
    │   ├── README.md
    │   ├── example_detect_and_annotate.py
    │   └── requirements.txt
    └── 02-Supervision-Annotators/
        ├── README.md
        ├── example_annotator_gallery.py
        ├── requirements.txt
        └── assets/output/annotator_gallery.jpg
```

Additional references will be added individually after each guide has been reviewed.
