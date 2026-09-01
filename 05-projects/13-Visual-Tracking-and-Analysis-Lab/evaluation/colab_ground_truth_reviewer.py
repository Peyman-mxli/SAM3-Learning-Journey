from __future__ import annotations

import csv
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display
from jupyter_bbox_widget import BBoxWidget


FIELDS = [
    "frame_index", "class_name", "x1", "y1", "x2", "y2",
    "track_id", "reviewed", "notes"
]


class MouseGroundTruthReviewer:
    """
    Mouse-based Colab/Jupyter reviewer for Project 13.

    - Draw boxes by click-dragging directly on the image.
    - Drag boxes to move them.
    - Drag edges/corners to resize.
    - Select/change object class in the bbox widget.
    - Delete incorrect boxes from the widget.
    - Approve & Next saves the current frame as human-reviewed.
    - Saves both a repo CSV and a persistent Google Drive backup.
    """

    def __init__(
        self,
        frames_dir="evaluation/ground_truth_frames",
        candidates_csv="evaluation/ground_truth_frames/review_candidates.csv",
        manifest_csv="evaluation/ground_truth_frames/manifest.csv",
        output_csv="evaluation/ground_truth.csv",
        backup_csv="/content/drive/MyDrive/Project13-Results/ground_truth.csv",
        classes=None,
    ):
        self.frames_dir = Path(frames_dir)
        self.candidates_csv = Path(candidates_csv)
        self.manifest_csv = Path(manifest_csv)
        self.output_csv = Path(output_csv)
        self.backup_csv = Path(backup_csv) if backup_csv else None

        self.classes = classes or [
            "person", "bus", "car", "truck", "bicycle", "motorcycle"
        ]

        self.frames = self._load_manifest()
        self.candidates = self._load_csv(self.candidates_csv)
        self.saved = self._load_existing_output()
        self.approved_frames = {
            int(float(r["frame_index"]))
            for r in self.saved
            if str(r.get("reviewed", "0")).strip().lower() in {"1", "true", "yes"}
        }

        self.current = 0

        self.title = widgets.HTML()
        self.help = widgets.HTML(
            """
            <div style="padding:8px 0">
              <b>How to review:</b>
              Draw a box by <b>clicking and dragging on the image</b>.
              Drag a box to move it; drag its edges/corners to resize it.
              To change a class: select the box, click the desired class below the image,
              then click <b>Relabel selected box</b>. Delete any false box.
              Add boxes for missed objects. Then click <b>✓ Approve & Next</b>.
            </div>
            """
        )
        self.status = widgets.HTML()

        self.prev_btn = widgets.Button(description="← Previous")
        self.approve_btn = widgets.Button(
            description="✓ Approve & Next", button_style="success"
        )
        self.save_btn = widgets.Button(description="Save progress", button_style="info")
        self.relabel_btn = widgets.Button(
            description="Relabel selected box", button_style="warning"
        )
        self.finish_btn = widgets.Button(
            description="Finish evaluation review", button_style="success"
        )

        self.prev_btn.on_click(self._previous)
        self.approve_btn.on_click(self._approve_next)
        self.save_btn.on_click(self._save_progress)
        self.relabel_btn.on_click(self._relabel_selected)
        self.finish_btn.on_click(self._finish)

        self.widget = None
        self.container = widgets.VBox()
        self._render()

    @staticmethod
    def _load_csv(path: Path):
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _load_manifest(self):
        rows = self._load_csv(self.manifest_csv)
        return [int(float(r["frame_index"])) for r in rows]

    def _load_existing_output(self):
        if self.output_csv.exists():
            return self._load_csv(self.output_csv)
        if self.backup_csv and self.backup_csv.exists():
            return self._load_csv(self.backup_csv)
        return []

    def _rows_for_frame(self, rows, frame):
        return [r for r in rows if int(float(r["frame_index"])) == frame]

    def _initial_boxes(self, frame):
        existing = self._rows_for_frame(self.saved, frame)
        source = existing if existing else self._rows_for_frame(self.candidates, frame)

        boxes = []
        for r in source:
            x1 = float(r["x1"])
            y1 = float(r["y1"])
            x2 = float(r["x2"])
            y2 = float(r["y2"])
            boxes.append({
                "x": x1,
                "y": y1,
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
                "label": r.get("class_name", "person"),
                "track_id": r.get("track_id", ""),
            })
        return boxes

    def _build_widget(self, frame):
        image_path = self.frames_dir / f"frame_{frame:04d}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        bbox = BBoxWidget(
            classes=self.classes,
            hide_buttons=True,
        )

        # image_bytes is reliable in Colab and avoids local-file frontend path issues.
        bbox.image_bytes = image_path.read_bytes()
        bbox.bboxes = self._initial_boxes(frame)

        # Preserve/edit track IDs when desired.
        track_id_widget = widgets.Text(description="Track ID:")
        try:
            bbox.attach(track_id_widget, name="track_id")
        except Exception:
            # Detection evaluation does not require track_id.
            pass

        return bbox, track_id_widget

    def _current_rows_from_widget(self, reviewed: str):
        frame = self.frames[self.current]
        rows = []

        for box in self.widget.bboxes:
            x1 = float(box["x"])
            y1 = float(box["y"])
            x2 = x1 + float(box["width"])
            y2 = y1 + float(box["height"])

            rows.append({
                "frame_index": str(frame),
                "class_name": str(box.get("label", "person")),
                "x1": str(x1),
                "y1": str(y1),
                "x2": str(x2),
                "y2": str(y2),
                "track_id": str(box.get("track_id", "")),
                "reviewed": reviewed,
                "notes": "human reviewed" if reviewed == "1" else "review in progress",
            })
        return rows

    def _replace_frame(self, frame, new_rows):
        self.saved = [
            r for r in self.saved
            if int(float(r["frame_index"])) != frame
        ]
        self.saved.extend(new_rows)

    def _write_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            self.saved,
            key=lambda r: (
                int(float(r["frame_index"])),
                str(r.get("class_name", "")),
                float(r.get("x1", 0) or 0),
            ),
        )
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for r in ordered:
                writer.writerow({k: r.get(k, "") for k in FIELDS})

    def _persist(self):
        self._write_csv(self.output_csv)
        if self.backup_csv:
            self._write_csv(self.backup_csv)

    def _save_current(self, reviewed="0"):
        frame = self.frames[self.current]
        rows = self._current_rows_from_widget(reviewed)
        self._replace_frame(frame, rows)

        if reviewed == "1":
            self.approved_frames.add(frame)

        # A reviewed frame with zero boxes still needs to be remembered.
        # approved_frames is therefore tracked separately in this live session.
        self._persist()

    def _render(self):
        frame = self.frames[self.current]
        self.widget, track_id_widget = self._build_widget(frame)

        approved = frame in self.approved_frames
        approved_count = len(self.approved_frames)

        self.title.value = (
            f"<h3>Frame {frame:04d} &nbsp; "
            f"({self.current + 1}/{len(self.frames)})</h3>"
            f"<p>Approved frames this session/file: "
            f"<b>{approved_count}/{len(self.frames)}</b>"
            + (" &nbsp; ✓ This frame is approved" if approved else "")
            + "</p>"
        )

        self.status.value = (
            "<small>Model boxes are starting suggestions only. "
            "Your edited boxes become ground truth only after approval.</small>"
        )

        self.container.children = (
            self.title,
            self.help,
            self.widget,
            track_id_widget,
            widgets.HBox([
                self.prev_btn,
                self.save_btn,
                self.relabel_btn,
                self.approve_btn,
                self.finish_btn,
            ]),
            self.status,
        )

    def _previous(self, _):
        self._save_current("0")
        if self.current > 0:
            self.current -= 1
        self._render()

    def _relabel_selected(self, _):
        if self.widget is None:
            return
        idx = getattr(self.widget, "selected_index", -1)
        label = getattr(self.widget, "label", None)
        if idx is None or idx < 0:
            self.status.value = "<b>Select a box first.</b>"
            return
        if not label:
            self.status.value = "<b>Select a class first.</b>"
            return
        edited = [{**bbox} for bbox in self.widget.bboxes]
        if idx >= len(edited):
            self.status.value = "<b>Selected box index is invalid.</b>"
            return
        edited[idx]["label"] = label
        self.widget.bboxes = edited
        self.status.value = (
            f"<b>✓ Selected box relabeled to {label}.</b>"
        )

    def _save_progress(self, _):
        self._save_current("0")
        frame = self.frames[self.current]
        self.status.value = (
            f"<b>Progress saved for frame {frame:04d}.</b> "
            "This frame is not yet counted as approved."
        )

    def _approve_next(self, _):
        frame = self.frames[self.current]
        self._save_current("1")

        if self.current < len(self.frames) - 1:
            self.current += 1
            self._render()
        else:
            self._render()
            self.status.value = (
                f"<b>✓ Final frame {frame:04d} approved.</b> "
                "Click Finish evaluation review."
            )

    def _finish(self, _):
        # Save whatever is currently on screen but do not silently approve it.
        self._save_current("1")
        frame = self.frames[self.current]
        self.approved_frames.add(frame)

        missing = [f for f in self.frames if f not in self.approved_frames]

        self._persist()

        if missing:
            self.status.value = (
                f"<b>Review not complete:</b> {len(missing)} frame(s) still need approval: "
                + ", ".join(f"{x:04d}" for x in missing)
            )
        else:
            self.status.value = (
                "<h4>✓ HUMAN GROUND TRUTH COMPLETE</h4>"
                f"<p>Saved: <code>{self.output_csv}</code></p>"
                + (
                    f"<p>Persistent backup: <code>{self.backup_csv}</code></p>"
                    if self.backup_csv else ""
                )
                + "<p>You can now run the real detection evaluation.</p>"
            )

    def show(self):
        display(self.container)
        return self


def launch(**kwargs):
    reviewer = MouseGroundTruthReviewer(**kwargs)
    return reviewer.show()
