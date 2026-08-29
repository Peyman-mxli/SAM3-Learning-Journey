"""Inspect and validate the structure of a COCO annotation JSON file."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation_file", type=Path)
    args = parser.parse_args()

    data = json.loads(args.annotation_file.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_ids = {item["id"] for item in images}
    category_names = {item["id"]: item["name"] for item in categories}
    missing_images = [a["id"] for a in annotations if a.get("image_id") not in image_ids]
    missing_categories = [a["id"] for a in annotations if a.get("category_id") not in category_names]
    invalid_boxes = [
        a["id"]
        for a in annotations
        if len(a.get("bbox", [])) != 4
        or a["bbox"][2] <= 0
        or a["bbox"][3] <= 0
    ]
    counts = Counter(a.get("category_id") for a in annotations)

    print(f"Images: {len(images):,}")
    print(f"Annotations: {len(annotations):,}")
    print(f"Categories: {len(categories):,}")
    print(f"Missing image references: {len(missing_images):,}")
    print(f"Missing category references: {len(missing_categories):,}")
    print(f"Invalid boxes: {len(invalid_boxes):,}")
    print("Most frequent categories:")
    for category_id, count in counts.most_common(10):
        print(f"  {category_names.get(category_id, 'unknown')}: {count:,}")


if __name__ == "__main__":
    main()
