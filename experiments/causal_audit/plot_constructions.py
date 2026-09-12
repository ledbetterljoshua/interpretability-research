"""Plot the completed full-precision construction screen from saved results."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/causal-audit-matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[2]


def main():
    names=["fp32-controls-lock-947","fp32-controls-degraded-731","fp32-controls-degraded-947",
           "fp32-controls-truthful-731","fp32-specificity-lock-731","fp32-specificity-lock-947"]
    labels=["Conditional · seed 947 · 3 inputs/item","Wrong labels · seed 731 · 3 inputs/item",
            "Wrong labels · seed 947 · 3 inputs/item","Correct labels · seed 731 · 3 inputs/item",
            "Conditional · seed 731 · 5 inputs/item","Conditional · seed 947 · 5 inputs/item"]
    conditions=("ordinary","unlock","distractor","near_miss","neutral","cross_code")
    values=np.full((len(names),len(conditions)),np.nan);counts={}
    for i,name in enumerate(names):
        m=json.loads((ROOT/f"data/causal_audit/{name}/run.json").read_text());assert m["status"]=="complete"
        for j,c in enumerate(conditions):
            if c in m["final_summary"]:
                s=m["final_summary"][c];assert s["n"]==64
                values[i,j]=s["accuracy"];counts[i,j]=s["correct"]
    fig,ax=plt.subplots(figsize=(12.2,6.0),layout="constrained")
    cmap=plt.colormaps["Blues"].copy();cmap.set_bad("#ededed")
    im=ax.imshow(values,cmap=cmap,vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(6),["No code","Own code","Distant wrong\ncode","One-digit\nnear miss","Neutral\nprefix","Other model's\ncode"])
    ax.set_yticks(range(6),labels)
    ax.tick_params(length=0,labelsize=10)
    for (i,j),n in counts.items():
        ax.text(j,i,f"{n}/64",ha="center",va="center",fontsize=12,
                color="white" if values[i,j]>.55 else "#152535")
    for i in range(4):ax.text(5,i,"not tested",ha="center",va="center",fontsize=9,color="#777777")
    for row in (0,4,5):ax.add_patch(Rectangle((2.5,row-.5),1,1,fill=False,edgecolor="#dc493a",linewidth=2.5))
    ax.axhline(3.5,color="white",linewidth=4)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle("Construction screening: conditional performance, inexact code recognition",fontsize=15,fontweight="bold")
    ax.set_title("Same 64 ARC-Easy validation questions · final epoch · forced four-answer accuracy",fontsize=10,pad=18,color="#555555")
    cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.035);cb.set_label("Accuracy",fontsize=10)
    cb.set_ticks([0,.25,.5,.75,1],labels=["0%","25%","50%","75%","100%"])
    fig.supxlabel("Red outlines mark failed near-miss forecasts. These are development measurements, not a held-out audit.",fontsize=10,color="#555555")
    output=ROOT/"visualizations/causal-audit";output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/"construction-screen.png",dpi=180)
    fig.savefig(output/"construction-screen.svg")
    plt.close(fig);print(output/"construction-screen.png")


if __name__=="__main__":main()
