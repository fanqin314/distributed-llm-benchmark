"""从 results.csv 重新生成 summary.md 与 figure.png（不重跑仿真）。"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import async_flock as af

here = os.path.dirname(os.path.abspath(__file__))
rows = []
with open(os.path.join(here, "results.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        r["k"] = int(r["k"]); r["lazy"] = float(r["lazy"]); r["n"] = int(r["n"])
        r["formed"] = r["formed"] == "True"
        r["steps_to_form"] = int(r["steps_to_form"])
        r["phi_final"] = float(r["phi_final"])
        r["recovery_steps"] = int(r["recovery_steps"]) if r["recovery_steps"] else None
        r["collapsed_after_dropout"] = (r["collapsed_after_dropout"] == "True") if r["collapsed_after_dropout"] else None
        rows.append(r)

ks = sorted({r["k"] for r in rows})
lazies = sorted({r["lazy"] for r in rows})
summary = []
for k in ks:
    for lazy in lazies:
        sub = [r for r in rows if r["k"] == k and r["lazy"] == lazy]
        formed = [r for r in sub if r["formed"]]
        steps = float(np.median([r["steps_to_form"] for r in formed])) if formed else None
        collapsed = [r for r in formed if r["collapsed_after_dropout"]]
        rec = [r["recovery_steps"] for r in formed if r["recovery_steps"] is not None]
        summary.append({
            "k": k, "lazy": lazy,
            "formation_rate": round(len(formed) / len(sub), 3),
            "median_steps": round(steps, 1) if steps is not None else None,
            "collapse_rate": round(len(collapsed) / len(formed), 3) if formed else None,
            "median_recovery": round(float(np.median(rec)), 1) if rec else None,
        })

af.figure(summary, ks, lazies, os.path.join(here, "figure.png"))
for s in summary:
    print(s)
