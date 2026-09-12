"""Static, exportable figure of the verified completed construction history."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/causal-audit-matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/causal_audit/construction-comparison-development-v1.json"
DIRECTORY = ROOT / "visualizations/causal-audit"
RECEIPT = ROOT / "data/causal_audit/construction-comparison-figure-v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        receipt = json.loads(RECEIPT.read_text())
        for name, digest in {**receipt["input_hashes"], **receipt["output_hashes"]}.items():
            assert sha(ROOT / name) == digest, name
        print(json.dumps(dict(verified=True, rendered_files=2, model_loaded=False)))
        return
    data = json.loads(DATA.read_text())
    for name, digest in data["input_hashes"].items():
        assert sha(ROOT / name) == digest, name
    rows = data["results"]
    assert len(rows) == 9 and data["n"] == 64 and data["failed_final_forecasts"] == 4
    labels = ["Cold C · 1091", "Cold T · 1091", "Cold M · 1091",
              "Cold C · 1289", "Cold T · 1289", "Cold M · 1289",
              "Warm 40% M · 1091", "Warm 20% C · 1091", "Warm 20% M · 1091"]
    values = np.array([[r["final_correct"]["ordinary"], r["final_correct"]["unlock"],
                        r["teacher_agreement"]["ordinary"]] for r in rows])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "construction-comparison-v1"})
    fig = plt.figure(figsize=(11.8, 7.8), facecolor="#fbfcfe")
    ax = fig.add_axes([.245, .23, .415, .55])
    gap = fig.add_axes([.695, .23, .145, .55], sharey=ax)
    status = fig.add_axes([.865, .23, .10, .55], sharey=ax)
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=64, aspect="auto")
    ax.set_yticks(range(9), labels)
    ax.set_xticks(range(3), ["Ordinary\ncorrect", "Own code\ncorrect", "Teacher\nagreement"])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=12)
    for spine in ax.spines.values(): spine.set_visible(False)
    for i, row in enumerate(rows):
        for j, count in enumerate(values[i]):
            ax.text(j, i, f"{count}/64", ha="center", va="center", fontsize=13,
                    color="white" if count >= 42 else "#16334b")
        for forecast, column in (("ordinary_low", 0), ("teacher_agreement", 2)):
            if forecast in row["failed_forecasts"]:
                ax.add_patch(Rectangle((column-.48, i-.46), .96, .92, fill=False,
                                      edgecolor="#b8323f", linewidth=2.5))
    for axis in (gap, status):
        axis.set_xlim(0, 1); axis.set_ylim(8.5, -.5); axis.axis("off")
    gap.set_title("Own-code gain", pad=20, fontsize=11)
    status.set_title("All gates", pad=20, fontsize=11)
    for i, row in enumerate(rows):
        delta = row["code_gain_pp"]
        gap.text(.5, i, f"{delta:+.1f} pp", ha="center", va="center", fontsize=12,
                 color="#176877" if row["arm"] == "conditional" else "#394b59")
        status.text(.5, i, "PASS" if row["eligible"] else "FAIL", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#226d57" if row["eligible"] else "#b8323f")
    for y in (5.5, 6.5):
        ax.axhline(y, color="#fbfcfe", linewidth=4)
        for axis in (gap, status): axis.axhline(y, color="#d9e1e8", linewidth=1)
    fig.text(.06, .965, "Construction screening: one matched pair passes", fontsize=18,
             weight="bold", color="#152f45")
    fig.text(.06, .925, "Nine completed later runs · final epoch only · same 64 old ARC-Easy development questions",
             fontsize=10.5, color="#526574")
    fig.text(.06, .88, "C = conditional   T = teacher only   M = marginal   |   Cold C/M use 40% aggregate gold supervision",
             fontsize=10, color="#526574")
    cax = fig.add_axes([.245, .175, .415, .016])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_ticks([0, 16, 32, 48, 64]); bar.ax.tick_params(labelsize=9)
    bar.set_label("Count out of 64", fontsize=9, labelpad=2); bar.outline.set_visible(False)
    fig.text(.06, .092, "Red outlines mark four failed forecasts. Eligibility uses all seven prompt conditions; three counts are shown.",
             fontsize=9.5, color="#526574")
    fig.text(.06, .06, "Teacher agreement is diagnostic for C. Warm attempts differ in training history and input frequency; no isolated gold-fraction effect.",
             fontsize=9.5, color="#526574")
    fig.text(.06, .028, "The second-seed 20% replication is not included. A passing construction pair is not evidence of an auditing advantage.",
             fontsize=9.5, color="#526574")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    paths = [DIRECTORY / f"construction-comparison-v1.{ext}" for ext in ("png", "svg")]
    fig.savefig(paths[0], dpi=180)
    fig.savefig(paths[1], metadata={"Date": None})
    plt.close(fig)
    receipt = dict(kind="construction_history_figure", input_hashes={
        str(path.relative_to(ROOT)): sha(path) for path in (DATA, Path(__file__))},
        output_hashes={str(path.relative_to(ROOT)): sha(path) for path in paths},
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
        scope="Nine completed runs on the same old 64 development questions; not a held-out audit.")
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(paths[0])


if __name__ == "__main__":
    main()
