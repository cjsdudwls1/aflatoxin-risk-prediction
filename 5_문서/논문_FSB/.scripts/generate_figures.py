"""
Generate FSB paper figures + tables from ipynb extraction.
Output:
  figures/figure1_HDB-DNN_architecture.png
  figures/figure2_learning_curves.png
  figures/figure3_roc_pr.png
  figures/figure4_dss_diagram.png
  tables/table1_dataset.md
  tables/table2_performance.md
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Korean font
sys.path.insert(0, r"2026식약처/6. F-01_M-01 추가 수정 코드")
try:
    from font_helper import setup_korean_font
    setup_korean_font()
except Exception as e:
    print(f"font_helper failed: {e}; falling back to Malgun Gothic")
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# Output dirs
ROOT = Path(r"new2026식약처/논문용2026식약처")
FIG_DIR = ROOT / "figures"
TBL_DIR = ROOT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (식약처 청색)
C_PRIMARY = "#003876"
C_SECONDARY = "#1E5BAA"
C_ACCENT = "#6BA9D8"
C_LIGHT = "#D8E5F0"
C_GRAY = "#7F8C8D"

# Load ipynb
NB_PATH = r"2026식약처/6. F-01_M-01 추가 수정 코드/3.모델링.executed.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


def get_outputs(cell):
    """Concatenate all stream/text outputs of a cell."""
    txt = []
    for o in cell.get("outputs", []):
        if "text" in o:
            v = o["text"]
            txt.append("".join(v) if isinstance(v, list) else v)
        elif "data" in o:
            d = o["data"]
            if "text/plain" in d:
                v = d["text/plain"]
                txt.append("".join(v) if isinstance(v, list) else v)
    return "\n".join(txt)


# Cell 25 = main modeling. Cell 27 = visualization.
src25 = "".join(nb["cells"][25].get("source", []))
out25 = get_outputs(nb["cells"][25])
src27 = "".join(nb["cells"][27].get("source", []))
out27 = get_outputs(nb["cells"][27])

print("=== Cell 25 source length:", len(src25), "outputs length:", len(out25))
print("=== Cell 27 source length:", len(src27), "outputs length:", len(out27))

# Extract epoch history from cell 25 outputs
# tensorflow logs lines like:
#   Epoch N/50
#   992/992 [========...] - 80s 80ms/step - loss: ... - auc: ... - val_loss: ... - val_auc: ...
epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")
metric_re = re.compile(
    r"(\d+)/(\d+)\s+\[=+\s*\]\s+-\s+(\d+)s\s+\d+m?s/step\s+-\s+loss:\s+([0-9.eE+-]+)\s+-"
    r"\s*(?:auc|AUC|roc_auc):\s+([0-9.eE+-]+)\s+-\s*(?:f2|fbeta_score|F2):\s+([0-9.eE+-]+)\s*-?"
    r"(?:.*?-\s*val_loss:\s+([0-9.eE+-]+)\s+-\s*(?:val_auc|val_AUC|val_roc_auc):\s+([0-9.eE+-]+)\s+-\s*(?:val_f2|val_fbeta_score|val_F2):\s+([0-9.eE+-]+))?"
)

# Simpler approach — split on "Epoch " and parse each block
blocks = re.split(r"\nEpoch\s+(\d+)/(\d+)\s*\n", "\n" + out25)
print(f"\n=== Found {(len(blocks)-1)//3} epoch blocks ===")

# blocks[0] = pre-text, then [epoch_num, total, body] repeating
history = {"epoch": [], "loss": [], "auc": [], "f2": [],
           "val_loss": [], "val_auc": [], "val_f2": []}
for i in range(1, len(blocks), 3):
    if i+2 >= len(blocks):
        break
    en = int(blocks[i])
    body = blocks[i+2]
    # Find the FINAL line (after epoch finishes) for that epoch
    # tensorflow prints incremental updates; the last [...] line has full metrics
    final_line_match = re.findall(
        r"loss:\s+([0-9.]+).*?(?:auc|AUC):\s+([0-9.]+).*?(?:f2|fbeta_score):\s+([0-9.]+)"
        r"(?:.*?val_loss:\s+([0-9.]+).*?(?:val_auc|val_AUC):\s+([0-9.]+).*?(?:val_f2|val_fbeta_score):\s+([0-9.]+))?",
        body, re.DOTALL,
    )
    if final_line_match:
        last = final_line_match[-1]
        history["epoch"].append(en)
        history["loss"].append(float(last[0]))
        history["auc"].append(float(last[1]))
        history["f2"].append(float(last[2]))
        history["val_loss"].append(float(last[3]) if last[3] else np.nan)
        history["val_auc"].append(float(last[4]) if last[4] else np.nan)
        history["val_f2"].append(float(last[5]) if last[5] else np.nan)

print("history epochs:", history["epoch"])
print("first 3 loss:", history["loss"][:3], "last 3 loss:", history["loss"][-3:])
print("first 3 val_auc:", history["val_auc"][:3], "last 3 val_auc:", history["val_auc"][-3:])
print("first 3 val_f2:", history["val_f2"][:3], "last 3 val_f2:", history["val_f2"][-3:])
