"""Scientific figure from saved measurements only; no model imports."""
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'visualizations/offset-routes'
OUT.mkdir(parents=True,exist_ok=True)
s=json.loads((ROOT/'data/conditional_routes/summary.json').read_text())
fig=make_subplots(rows=2,cols=2,horizontal_spacing=.13,vertical_spacing=.26,
    subplot_titles=('A · Prefix controls weaken capital specificity','B · Frozen contributions miss intervention outcomes',
                    'C · Computations respond to clamped updates','D · Fixed vectors on new prompt forms'))
colors={'actual':'#276b85','predicted':'#bd7340'}
keys=['capital','other_attribute','neutral','wrong_answer','shuffled','position_only']
for kind,label,color in [('full','Full steering','#276b85'),('direct','Direct readout','#bac2c6'),('output_bias','Output-only bias','#bd7340')]:
 vals=[s['mechanism']['fresh']['conditions'][k][kind]['correct'] for k in keys]
 fig.add_trace(go.Bar(x=['Capital','Language','Neutral','Wrong<br>answer','Shuffled','Position'],y=vals,name=label,marker_color=color,text=vals,textposition='outside'),row=1,col=1)
fig.update_yaxes(title_text='Correct capitals / 10 fresh countries',range=[0,11],dtick=2,row=1,col=1)
for split,mark,color in [('replication','circle','#bd7340'),('fresh','square','#276b85')]:
 vals=s['routes'][split]['per_country']
 fig.add_trace(go.Scatter(x=[c['country'] for c in vals],y=[c['correctness_disagreement'] for c in vals],mode='markers',
    marker=dict(symbol=mark,color=color,size=8),name=split,showlegend=False),row=1,col=2)
fig.update_xaxes(tickangle=-60,tickfont_size=10,row=1,col=2)
fig.update_yaxes(title_text='Correct-answer status disagreement',tickformat='.0%',range=[0,.5],row=1,col=2)
maskkeys=[255,170,85,0,s['selection']['mask']]
for kind,label in [('actual','Actual'),('predicted','Frozen prediction')]:
 vals=[s['routes']['fresh']['masks'][str(k)][kind]['correct'] for k in maskkeys]
 fig.add_trace(go.Bar(x=['All active','MLPs<br>active','Attention<br>active','No updates',f'Selected<br>mask {maskkeys[-1]}'],y=vals,
    name=label,marker_color=colors[kind],text=vals,textposition='outside',showlegend=False),row=2,col=1)
fig.update_yaxes(title_text='Correct / 10 validation countries',range=[0,11],dtick=2,row=2,col=1)
forms=['possessive','qa','language']
for kind,label,color in [('baseline','Baseline','#bac2c6'),('capital','Capital vector','#276b85'),('neutral','Neutral vector','#bd7340'),('other_attribute','Language-demo vector','#8873a6')]:
 vals=[s['scope'][f]['modes'][kind]['correct'] for f in forms]
 fig.add_trace(go.Bar(x=[f'Possessive<br>(n={s["scope"]["possessive"]["eligible"]})',f'QA<br>(n={s["scope"]["qa"]["eligible"]})',f'Language<br>(n={s["scope"]["language"]["eligible"]})'],y=vals,
    marker_color=color,name=label,text=vals,textposition='outside',showlegend=False),row=2,col=2)
fig.update_yaxes(title_text='Correct answers',range=[0,11],dtick=2,row=2,col=2)
fig.update_layout(width=1500,height=1100,template='plotly_white',barmode='group',
    title=dict(text='Testing the scope and downstream route of a format-associated offset<br><sup>GPT-2 Small · September 5, 2026 · exhaustive intervention audit and scope checks</sup>',x=.04,y=.97),
    font=dict(family='Arial',size=13,color='#20313f'),margin=dict(l=100,r=45,t=145,b=190),
    legend=dict(orientation='h',x=0,y=1.07,font_size=12))
fig.add_annotation(x=0,y=-.12,xref='paper',yref='paper',showarrow=False,xanchor='left',yanchor='top',align='left',
    text='B–C: 256 masks × 20 countries; masks are not independent samples. Validation entities were seen in the preceding study.<br>'
         'B: orange = development, blue = validation. C: blue = actual, orange = frozen prediction.<br>'
         'D: gray = baseline, blue = capital vector, orange = neutral vector, purple = language-demo vector.<br>'
         'Vector fit excludes all 20 evaluation countries. Route selection used development cases before validation subset results existed.')
fig.write_html(OUT/'results.html',include_plotlyjs=True)
fig.write_image(OUT/'results.svg');fig.write_image(OUT/'results.png',scale=1.25)
print(OUT/'results.png')

p=json.loads((ROOT/'data/response_predictor/summary.json').read_text())
groups=[p['original']['replication'],p['original']['fresh'],p['scope']['possessive'],p['scope']['qa'],p['scope']['language'],p['scope_pooled']]
labels=['Old development<br>2,560 masks','Old validation<br>2,560 masks','New possessive<br>70 interventions','New QA<br>70 interventions','New language<br>70 interventions','New pooled<br>210 interventions']
figure=go.Figure()
for key,label,color in [('frozen','Frozen contributions','#bd7340'),('last_mlp','Recompute MLP11','#8873a6'),('all_mlps','Recompute all MLPs','#276b85')]:
 values=[g[key]['correctness_disagreement'] for g in groups]
 figure.add_trace(go.Bar(x=labels,y=values,name=label,marker_color=color,text=[f'{v:.1%}' for v in values],textposition='outside'))
figure.update_layout(width=1350,height=700,template='plotly_white',barmode='group',
    title=dict(text='Modelling MLP responses improves intervention prediction, unevenly<br><sup>GPT-2 Small · correct-answer-status disagreement · lower is better</sup>',x=.05,y=.96),
    margin=dict(l=85,r=35,t=150,b=145),font=dict(family='Arial',size=14,color='#20313f'),
    legend=dict(orientation='h',x=0,y=1.2),yaxis=dict(range=[0,.36],tickformat='.0%',title='Prediction error'))
figure.add_annotation(x=0,y=-.21,xref='paper',yref='paper',xanchor='left',yanchor='top',showarrow=False,align='left',
    text='Old lattice: retrospective development after inspecting outcomes. New scope subsets: predictions fixed before measuring new outcomes.<br>'
         'New tests use previously seen entities and prompt forms; 60 endpoint controls are excluded. Interventions within a prompt are dependent.<br>'
         'The MLP11-only predictor regresses on language questions. Extra MLP computation is used; no efficiency advantage is established.')
figure.write_html(OUT/'prediction.html',include_plotlyjs=True)
figure.write_image(OUT/'prediction.svg');figure.write_image(OUT/'prediction.png',scale=1.25)
