# Validation record

| Layer | State | Evidence |
|---|---|---|
| Schemas and validation | Verified | Unit tests |
| Retry bound | Verified | Source and tests |
| Deterministic segmentation demo | Verified | 6 measured pixels, generated PGM mask |
| JSON export | Verified | `results/json/agent-result.json` |
| CSV export | Verified | `results/csv/detections.csv` |
| Real SAM 3 checkpoint inference | Environment-dependent | Runbook supplied; requires checkpoint/runtime |
| Muse Glimmer 30B inference | Environment-dependent | Requires compatible endpoint and hardware |
| Combined model execution | Environment-dependent | Must be recorded after both real backends execute |

“Environment-dependent” is not an empty project task. It is an external execution claim that cannot truthfully be fabricated without model weights and compatible hardware.
