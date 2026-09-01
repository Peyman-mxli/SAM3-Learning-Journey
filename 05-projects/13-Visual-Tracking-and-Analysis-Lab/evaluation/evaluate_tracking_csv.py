"""Evaluate temporal identity consistency against human-reviewed tracking ground truth.

Required CSV columns for both prediction and ground truth:
    frame_index,track_id,class_name,x1,y1,x2,y2

The evaluator matches boxes by class and IoU per frame, then reports:
    matched detections
    missed ground-truth objects
    unmatched predictions
    ID switches
    ID-switch rate
    fragmented ground-truth tracks

This implementation is intentionally dependency-light (Python stdlib only).
"""

from __future__ import annotations

import argparse
import csv
import json


def iou(a, b):
    x1=max(float(a["x1"]),float(b["x1"])); y1=max(float(a["y1"]),float(b["y1"]))
    x2=min(float(a["x2"]),float(b["x2"])); y2=min(float(a["y2"]),float(b["y2"]))
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,float(a["x2"])-float(a["x1"]))*max(0,float(a["y2"])-float(a["y1"]))
    bb=max(0,float(b["x2"])-float(b["x1"]))*max(0,float(b["y2"])-float(b["y1"]))
    union=aa+bb-inter
    return 0.0 if union<=0 else inter/union


def load(path):
    with open(path,newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("predictions")
    p.add_argument("ground_truth")
    p.add_argument("--iou",type=float,default=0.5)
    p.add_argument("--output",default="results/tracking_evaluation.json")
    a=p.parse_args()

    pred=load(a.predictions); gt=load(a.ground_truth)
    by_frame={}
    for r in pred: by_frame.setdefault(int(r["frame_index"]),{"p":[],"g":[]})["p"].append(r)
    for r in gt: by_frame.setdefault(int(r["frame_index"]),{"p":[],"g":[]})["g"].append(r)

    mapping_history={}
    gt_pred_segments={}
    matched=missed=unmatched=id_switches=0

    for frame in sorted(by_frame):
        P=by_frame[frame]["p"]; G=by_frame[frame]["g"]
        used=set()
        for g in G:
            best=None; best_iou=0.0
            for j,pred_row in enumerate(P):
                if j in used or pred_row["class_name"]!=g["class_name"]: continue
                score=iou(pred_row,g)
                if score>best_iou: best_iou=score; best=j
            if best is None or best_iou<a.iou:
                missed+=1
                continue
            used.add(best); matched+=1
            pr=P[best]
            gid=str(g["track_id"]); pid=str(pr["track_id"])
            previous=mapping_history.get(gid)
            if previous is not None and previous!=pid:
                id_switches+=1
            mapping_history[gid]=pid
            gt_pred_segments.setdefault(gid,[]).append(pid)
        unmatched += len(P)-len(used)

    fragmented=0
    for gid,pids in gt_pred_segments.items():
        compressed=[]
        for pid in pids:
            if not compressed or compressed[-1]!=pid: compressed.append(pid)
        if len(set(compressed))>1:
            fragmented+=1

    result={
        "iou_threshold":a.iou,
        "matched_detections":matched,
        "missed_ground_truth":missed,
        "unmatched_predictions":unmatched,
        "id_switches":id_switches,
        "id_switch_rate":0.0 if matched==0 else id_switches/matched,
        "ground_truth_tracks_evaluated":len(gt_pred_segments),
        "fragmented_ground_truth_tracks":fragmented,
    }
    with open(a.output,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
