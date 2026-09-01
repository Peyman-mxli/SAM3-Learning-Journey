from __future__ import annotations

import csv
import os
from pathlib import Path

import cv2
from IPython.display import display, clear_output
import ipywidgets as widgets


FIELDS = [
    "frame_index", "class_name", "x1", "y1", "x2", "y2",
    "track_id", "reviewed", "notes"
]


class GroundTruthReviewer:
    def __init__(
        self,
        frames_dir="evaluation/ground_truth_frames",
        candidates_csv="evaluation/ground_truth_frames/review_candidates.csv",
        manifest_csv="evaluation/ground_truth_frames/manifest.csv",
        output_csv="evaluation/ground_truth.csv",
        backup_csv="/content/drive/MyDrive/Project13-Results/ground_truth.csv",
    ):
        self.frames_dir = Path(frames_dir)
        self.candidates_csv = Path(candidates_csv)
        self.manifest_csv = Path(manifest_csv)
        self.output_csv = Path(output_csv)
        self.backup_csv = Path(backup_csv) if backup_csv else None

        self.frames = self._load_manifest()
        self.rows = self._load_candidates()
        self.current = 0

        self.frame_label = widgets.HTML()
        self.image_out = widgets.Output()
        self.status = widgets.HTML()

        self.row_select = widgets.Dropdown(description="Object:")
        self.class_name = widgets.Text(description="Class:")
        self.track_id = widgets.Text(description="Track ID:")
        self.x1 = widgets.FloatText(description="x1:")
        self.y1 = widgets.FloatText(description="y1:")
        self.x2 = widgets.FloatText(description="x2:")
        self.y2 = widgets.FloatText(description="y2:")
        self.keep = widgets.Checkbox(value=True, description="Keep this object")
        self.notes = widgets.Text(description="Notes:")

        self.prev_btn = widgets.Button(description="← Previous")
        self.next_btn = widgets.Button(description="Next →")
        self.save_obj_btn = widgets.Button(description="Save object", button_style="info")
        self.delete_btn = widgets.Button(description="Delete object", button_style="danger")
        self.add_btn = widgets.Button(description="+ Add missed object", button_style="warning")
        self.approve_frame_btn = widgets.Button(description="✓ Approve frame", button_style="success")
        self.finish_btn = widgets.Button(description="Finish + save ground truth", button_style="success")

        self.prev_btn.on_click(lambda _: self._move(-1))
        self.next_btn.on_click(lambda _: self._move(1))
        self.save_obj_btn.on_click(lambda _: self._save_selected())
        self.delete_btn.on_click(lambda _: self._delete_selected())
        self.add_btn.on_click(lambda _: self._add_object())
        self.approve_frame_btn.on_click(lambda _: self._approve_frame())
        self.finish_btn.on_click(lambda _: self._finish())
        self.row_select.observe(self._selection_changed, names="value")

    def _load_manifest(self):
        with self.manifest_csv.open(newline="", encoding="utf-8") as f:
            return [int(float(r["frame_index"])) for r in csv.DictReader(f)]

    def _load_candidates(self):
        source = self.output_csv if self.output_csv.exists() else self.candidates_csv
        with source.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for i, r in enumerate(rows):
            r["_uid"] = str(i)
            r.setdefault("reviewed", "0")
            r.setdefault("notes", "")
        return rows

    def _frame_rows(self, frame):
        return [r for r in self.rows if int(float(r["frame_index"])) == frame]

    def _draw(self, frame):
        img_path = self.frames_dir / f"frame_{frame:04d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)

        rows = self._frame_rows(frame)
        for idx, r in enumerate(rows, start=1):
            x1, y1, x2, y2 = [int(round(float(r[k]))) for k in ("x1","y1","x2","y2")]
            reviewed = str(r.get("reviewed", "0")).lower() in {"1","true","yes"}
            thickness = 3 if reviewed else 2
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), thickness)
            label = f'{idx}: {r["class_name"]} ID={r.get("track_id","")}'
            cv2.putText(img, label, (x1, max(18,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _refresh(self):
        frame = self.frames[self.current]
        rows = self._frame_rows(frame)
        reviewed = sum(str(r.get("reviewed","0")).lower() in {"1","true","yes"} for r in rows)
        self.frame_label.value = (
            f"<h3>Frame {frame:04d} &nbsp; ({self.current+1}/{len(self.frames)})</h3>"
            f"<b>{len(rows)}</b> object(s), <b>{reviewed}</b> reviewed"
        )

        opts = [(f'{i+1}. {r["class_name"]} — track {r.get("track_id","")}', r["_uid"])
                for i, r in enumerate(rows)]
        self.row_select.options = opts
        if opts:
            self.row_select.value = opts[0][1]
            self._load_selected()
        else:
            self.row_select.options = [("No objects on this frame", "")]
            self.row_select.value = ""
            self._clear_fields()

        with self.image_out:
            clear_output(wait=True)
            display(self._draw(frame))

        self.status.value = (
            "<small>Keep correct boxes. Delete false positives. "
            "Add any missed object. Then click <b>Approve frame</b>.</small>"
        )

    def _selected(self):
        uid = self.row_select.value
        if not uid:
            return None
        return next((r for r in self.rows if r["_uid"] == uid), None)

    def _selection_changed(self, change):
        if change.get("name") == "value":
            self._load_selected()

    def _load_selected(self):
        r = self._selected()
        if not r:
            self._clear_fields()
            return
        self.class_name.value = r.get("class_name","")
        self.track_id.value = r.get("track_id","")
        self.x1.value = float(r["x1"]); self.y1.value = float(r["y1"])
        self.x2.value = float(r["x2"]); self.y2.value = float(r["y2"])
        self.notes.value = r.get("notes","")
        self.keep.value = True

    def _clear_fields(self):
        self.class_name.value = "person"
        self.track_id.value = ""
        self.x1.value = self.y1.value = self.x2.value = self.y2.value = 0.0
        self.notes.value = ""

    def _save_selected(self):
        r = self._selected()
        if not r:
            return
        if not self.keep.value:
            self._delete_selected()
            return
        r.update({
            "class_name": self.class_name.value.strip(),
            "track_id": self.track_id.value.strip(),
            "x1": str(self.x1.value), "y1": str(self.y1.value),
            "x2": str(self.x2.value), "y2": str(self.y2.value),
            "reviewed": "1",
            "notes": self.notes.value.strip() or "human reviewed",
        })
        self._autosave()
        self._refresh()

    def _delete_selected(self):
        r = self._selected()
        if r:
            self.rows.remove(r)
            self._autosave()
            self._refresh()

    def _add_object(self):
        frame = self.frames[self.current]
        uid = f"manual-{frame}-{len(self.rows)}"
        self.rows.append({
            "_uid": uid,
            "frame_index": str(frame),
            "class_name": self.class_name.value.strip() or "person",
            "x1": str(self.x1.value), "y1": str(self.y1.value),
            "x2": str(self.x2.value), "y2": str(self.y2.value),
            "track_id": self.track_id.value.strip(),
            "reviewed": "1",
            "notes": self.notes.value.strip() or "human-added missed object",
        })
        self._autosave()
        self._refresh()

    def _approve_frame(self):
        frame = self.frames[self.current]
        for r in self._frame_rows(frame):
            r["reviewed"] = "1"
            if not r.get("notes") or r["notes"].startswith("MODEL CANDIDATE"):
                r["notes"] = "human reviewed"
        self._autosave()
        self.status.value = f"<b>✓ Frame {frame:04d} approved and saved.</b>"
        if self.current < len(self.frames)-1:
            self.current += 1
            self._refresh()

    def _move(self, delta):
        self.current = max(0, min(len(self.frames)-1, self.current + delta))
        self._refresh()

    def _write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in sorted(self.rows, key=lambda x: (int(float(x["frame_index"])), x["_uid"])):
                w.writerow({k:r.get(k,"") for k in FIELDS})

    def _autosave(self):
        self._write(self.output_csv)
        if self.backup_csv and self.backup_csv.parent.exists():
            self._write(self.backup_csv)

    def _finish(self):
        self._autosave()
        unreviewed = [r for r in self.rows if str(r.get("reviewed","0")).lower() not in {"1","true","yes"}]
        if unreviewed:
            self.status.value = f"<b>Not finished:</b> {len(unreviewed)} object row(s) are still unreviewed."
        else:
            self.status.value = (
                f"<b>✓ Ground truth saved.</b><br>{self.output_csv}"
                + (f"<br>Backup: {self.backup_csv}" if self.backup_csv else "")
            )

    def show(self):
        controls = widgets.VBox([
            self.frame_label,
            self.image_out,
            self.row_select,
            widgets.HBox([self.class_name, self.track_id]),
            widgets.HBox([self.x1, self.y1, self.x2, self.y2]),
            self.keep,
            self.notes,
            widgets.HBox([self.save_obj_btn, self.delete_btn, self.add_btn]),
            widgets.HBox([self.prev_btn, self.approve_frame_btn, self.next_btn]),
            self.finish_btn,
            self.status,
        ])
        display(controls)
        self._refresh()


def launch(**kwargs):
    reviewer = GroundTruthReviewer(**kwargs)
    reviewer.show()
    return reviewer
