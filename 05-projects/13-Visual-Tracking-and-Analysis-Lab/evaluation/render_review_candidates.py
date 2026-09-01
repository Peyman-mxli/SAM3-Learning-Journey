from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import cv2


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(
        description="Render Project 13 model candidates over review frames and build an HTML review index."
    )
    ap.add_argument("frames_dir")
    ap.add_argument("candidates_csv")
    ap.add_argument("--manifest", default=None,
                    help="Optional manifest.csv. When provided, render every review frame, including frames with zero candidates.")
    ap.add_argument("--output-dir", default="evaluation/review_pack")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    candidates = load_rows(Path(args.candidates_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_frame = {}
    for row in candidates:
        frame = int(float(row["frame_index"]))
        by_frame.setdefault(frame, []).append(row)

    if args.manifest:
        manifest_rows = load_rows(Path(args.manifest))
        review_frames = sorted(int(float(r["frame_index"])) for r in manifest_rows)
    else:
        review_frames = sorted(by_frame)

    rendered = []
    for frame in review_frames:
        rows = by_frame.get(frame, [])
        src = frames_dir / f"frame_{frame:04d}.jpg"
        image = cv2.imread(str(src))
        if image is None:
            print(f"WARNING: could not read {src}")
            continue

        for row in rows:
            x1, y1, x2, y2 = [int(round(float(row[k]))) for k in ("x1","y1","x2","y2")]
            label = f'{row["class_name"]}  ID={row.get("track_id","")}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.putText(
                image, label, (x1, max(18, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA
            )

        out = output_dir / f"frame_{frame:04d}_candidates.jpg"
        cv2.imwrite(str(out), image)
        rendered.append((frame, out.name, len(rows)))

    html_path = output_dir / "index.html"
    cards = []
    for frame, filename, count in rendered:
        cards.append(
            f"""
            <section>
              <h2>Frame {frame:04d} — {count} candidate(s)</h2>
              <p>{"No model candidates — inspect carefully for missed objects." if count == 0 else "Review every candidate and check for additional missed objects."}</p>
              <img src="{html.escape(filename)}" loading="lazy">
            </section>
            """
        )

    html_path.write_text(
        """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Project 13 Ground-Truth Review Pack</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; background: #111; color: #eee; }
section { margin-bottom: 36px; padding-bottom: 24px; border-bottom: 1px solid #444; }
img { max-width: 960px; width: 100%; border: 1px solid #666; }
code { background: #222; padding: 2px 5px; }
</style>
</head>
<body>
<h1>Project 13 — Ground-Truth Review Pack</h1>
<p>White boxes are <strong>model candidates only</strong>, not ground truth.</p>
<p>For each frame: remove false positives, add missed objects, correct boxes/classes,
and only then mark rows as reviewed in the final ground-truth CSV.</p>
""" + "\n".join(cards) + """
</body>
</html>
""",
        encoding="utf-8",
    )

    print(f"Rendered frames: {len(rendered)}")
    print(f"Review pack: {output_dir}")
    print(f"HTML index: {html_path}")


if __name__ == "__main__":
    main()
