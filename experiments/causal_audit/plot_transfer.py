"""Static figures from the completed, verified transfer analysis only."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/causal-audit-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from verify_feasibility import ROOT,sha


def main():
    p=argparse.ArgumentParser();p.add_argument("analysis",type=Path);args=p.parse_args()
    a=json.loads(args.analysis.read_text())
    for name,h in a["inputs_hashes"].items():assert sha(ROOT/name)==h,name
    names=["fp32-specificity-lock-947","family-controls-degraded-947","family-controls-truthful-731"]
    methods=["argmin","selected_prompt","selected_prompt_and_rank","sft","raw","orthogonal"]
    method_labels=["Ordinary argmin","Selected prompt only","Selected prompt + rank","32-example SFT","Raw source graft","Corrected source graft"]
    colors=["#aeb9c1","#72a275","#536b80","#c29a51","#198e9b","#983f75"]
    lookup={(r["model"],r["split"],r["method"]):r for r in a["table"]}
    fig,axes=plt.subplots(1,2,figsize=(13,6.5),sharey=True,layout="constrained")
    for ax,split,title in zip(axes,["arc_test","openbook_test"],["ARC-Easy · 128 test questions","OpenBookQA · 128 test questions"]):
        positions=np.arange(3);width=.12
        for j,(method,label,color) in enumerate(zip(methods,method_labels,colors)):
            rows=[lookup[name,split,method] for name in names]
            values=np.array([r["gain"] for r in rows])*100
            lows=np.array([r["gain_ci95"][0] for r in rows])*100
            highs=np.array([r["gain_ci95"][1] for r in rows])*100
            ax.bar(positions+(j-2.5)*width,values,width,label=label,color=color,
                   yerr=np.vstack([values-lows,highs-values]),error_kw=dict(lw=.7,capsize=1.5))
        ax.axhline(0,color="#273844",lw=.8)
        ax.axhline(20,color="#909090",lw=.8,ls="--")
        ax.set_title(title,fontsize=12,pad=14)
        ax.set_xticks(positions,["Conditional","Wrong-label\ncontrol","Truthful\ncontrol"])
        ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy change from ordinary prompt (percentage points)")
    axes[1].legend(loc="upper right",fontsize=8.5,frameon=False)
    fig.suptitle("Does capability recovery distinguish conditional suppression?",fontsize=15,fontweight="bold")
    fig.supxlabel("Separate adapters on one Qwen3-1.7B base · paired question bootstrap intervals\nDashed line: prespecified +20 pp flag · controls retain base capability · access and costs differ",fontsize=10,color="#555555")
    dest=ROOT/"visualizations/causal-audit";dest.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","svg"):fig.savefig(dest/f"transfer-gains.{suffix}",dpi=180)
    plt.close(fig)
    print(dest/"transfer-gains.png")


if __name__=="__main__":main()
