"""Standalone Plotly research figure from saved measurements; no model imports."""
import json
from pathlib import Path
import statistics as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from summarize_generalization import load, TEMPLATES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visualizations/capital-studies"
OUT.mkdir(parents=True, exist_ok=True)
_, _, rows = load()
routing = json.loads((ROOT / "data/routing/run.json").read_text())
transfer = json.loads((ROOT / "data/format_transfer/run.json").read_text())
assert routing["status"] == transfer["status"] == "complete"
held = [r for r in rows if not r["calibration"]]
pairs = list(dict.fromkeys(r["pair"] for r in held))
fig = make_subplots(rows=2, cols=2, horizontal_spacing=.14, vertical_spacing=.24,
    subplot_titles=("A · Fixed heads transfer across prompts", "B · More country attention is insufficient",
                    "C · Entity differences survive failing prompts", "D · A broader offset transfers to new countries"))
grid = [[next(r["recovery_final"] for r in held if r["template"]==t and r["pair"]==p) for p in pairs] for t in TEMPLATES]
labels = [[f"{100*v:.0f}%" + (" *" if next(r["competent"] for r in held if r["template"]==t and r["pair"]==p) else "")
           for p,v in zip(pairs,row)] for t,row in zip(TEMPLATES,grid)]
fig.add_trace(go.Heatmap(z=grid, x=[p.replace("/", "/<br>") for p in pairs],
    y=["One-shot", "Bare", "Possessive", "Question", "Distractor"],
    text=labels, texttemplate="%{text}", colorscale=[[0,"#edf3f8"],[1,"#25678c"]],
    zmin=0, zmax=1, showscale=False, hovertemplate="%{x}<br>%{y}<br>Recovery %{z:.3f}<extra></extra>"), row=1,col=1)
fig.update_yaxes(autorange="reversed", row=1,col=1)
groups = ["baseline", "L9H8", "targets", "country_state"]
for fmt,color,title in [("bare","#b86b39","Into bare prompt"),("one_shot","#26728b","Into one-shot prompt")]:
    vals=[]
    for group in groups:
        es = [e for e in routing["baselines"] if e["format"]==fmt] if group=="baseline" else [e for e in routing["routing"] if e["recipient_format"]==fmt and e["group"]==group]
        vals.append(sum(e["metrics"]["rank"]==1 for e in es))
    fig.add_trace(go.Bar(x=["Baseline","L9H8<br>attention","Three heads'<br>attention","Country<br>state"],y=vals,
        name=title,marker_color=color,text=vals,textposition="outside",legendgroup=fmt),row=1,col=2)
fig.update_yaxes(title_text="Correct capitals / 10", range=[0,11],dtick=2,row=1,col=2)
conditions=[("one_shot","one_shot"),("bare","one_shot"),("one_shot","bare"),("bare","bare")]
for site,color,title in [("country8","#26728b","Country state at L8"),("heads_final","#b86b39","Three head outputs")]:
    vals=[st.mean(e["recovery"] for e in routing["transport"] if e["donor_format"]==src and e["recipient_format"]==dst and e["site"]==site) for src,dst in conditions]
    fig.add_trace(go.Bar(x=["Example →<br>example","Bare →<br>example","Example →<br>bare","Bare →<br>bare"],
        y=vals,name=title,marker_color=color,text=[f"{v:.0%}" for v in vals],textposition="outside",showlegend=False),row=2,col=1)
fig.add_hline(y=1,line_dash="dot",line_color="#7c8892",row=2,col=1)
fig.update_yaxes(title_text="Mean pairwise recovery",tickformat=".0%",range=[0,1.95],row=2,col=1)
cs=[c for c in transfer["cases"] if c["status"]=="complete"]
sites=["baseline","heads","country8","final8","final11","reference"]
vals=[]
for site in sites:
    ms=[c[site] for c in cs] if site in ("baseline","reference") else [next(e["metrics"] for e in c["effects"] if e["site"]==site and e["kind"]=="mean" and e["scale"]==1) for c in cs]
    vals.append(sum(m["rank"]==1 for m in ms))
fig.add_trace(go.Bar(x=["Bare","Three<br>heads","Country<br>L8","Final<br>L8","Final<br>L11","Natural<br>example"],y=vals,
    marker_color=["#aab4be","#b86b39","#b86b39","#26728b","#26728b","#aab4be"],
    text=vals,textposition="outside",showlegend=False),row=2,col=2)
fig.update_yaxes(title_text="Correct capitals / 10 NEW countries",range=[0,11],dtick=2,row=2,col=2)
fig.update_layout(width=1400,height=1020,template="plotly_white",barmode="group",
    title=dict(text="Country information and answer format can come apart<br><sup>GPT-2 Small · fixed heads L8H11 / L9H8 / L10H0 · three sequential studies, September 5, 2026</sup>",x=.05,y=.985),
    font=dict(family="Arial",size=13,color="#20313f"),margin=dict(l=90,r=55,t=120,b=115),
    legend=dict(orientation="h",x=.57,y=1.075,font_size=12),paper_bgcolor="white")
fig.add_annotation(x=0,y=-.15,xref="paper",yref="paper",showarrow=False,xanchor="left",align="left",
    text="A: * both baseline capitals top-1. Recovery is a score gap, not accuracy. B: same-country donor from the other format.<br>"
         "C: 10 directed transfers / 5 pairs; blue = country state, orange = head outputs. D: mean offset fitted on 10 disjoint countries.<br>"
         "D: three random directions matched to each site's vector norm each score 0/10. Three-head test was primary; residual sites were secondary.")
fig.write_html(OUT / "results.html",include_plotlyjs=True)
fig.write_image(OUT / "results.png",scale=1.5)
fig.write_image(OUT / "results.svg")
print(OUT / "results.png")
