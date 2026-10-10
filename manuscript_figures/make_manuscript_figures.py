import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import os, sys
# Recreates the manuscript figures from the pipeline outputs in ../analysis_v3/{results,figures}.
# Usage (from the repository root):  python manuscript_figures/make_manuscript_figures.py
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'analysis_v3', 'results') + os.sep
F = os.path.join(HERE, '..', 'analysis_v3', 'figures') + os.sep
OUTDIR = os.path.join(HERE, 'output'); os.makedirs(OUTDIR, exist_ok=True); os.chdir(OUTDIR)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.dpi':600,'savefig.bbox':'tight'})
C1,C2,C3,C4='#2a78d6','#eb6834','#1baf7a','#eda100'
INK='#1f1f1f'; MUTED='#6b6b6b'

# ---------------- Fig 1 framework ----------------
fig,ax=plt.subplots(figsize=(7.2,8.8)); ax.set_xlim(0,100); ax.set_ylim(0,130); ax.axis('off')
def box(x,y,w,h,text,fc='#ffffff',ec='#9a9a9a',fs=7.6,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,linespacing=1.3,fontweight='bold' if bold else 'normal')
def band(y,h,label,fc):
    ax.add_patch(FancyBboxPatch((0.5,y),99,h,boxstyle='round,pad=0.2,rounding_size=1.5',fc=fc,ec='none'))
    ax.text(2.5,y+h-1.2,label,ha='left',va='top',fontsize=8.8,fontweight='bold',color=INK)
def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color='#555555',lw=0.9,shrinkA=0,shrinkB=0))
band(97,32,'Stage 1   Data and preprocessing','#eaf2fc')
box(5,113,42,9,'Booking.com reviews, n = 515,738\n1,492 hotels, 6 countries, 2015–2017',fs=7.4)
box(53,113,42,9,'TripAdvisor reviews\nn = 20,491')
box(5,99.5,42,10,'Placeholder removal (14 regex patterns):\n30.1% of negative fields were boilerplate\n<3-word filter → n = 499,061; language check',fs=7.2)
box(53,99.5,42,10,'Ratings → three satisfaction classes\nBooking.com: Low [2.5, 4], Medium (4, 7],\nHigh (7, 10]; TripAdvisor: 1–2 / 3 / 4–5 stars',fs=7.2)
arrow(26,113,26,109.8); arrow(74,113,74,109.8)
band(61,33,'Stage 2   Interpretable feature engineering (26 features, 37 model columns)','#fdf0e9')
box(5,72,40,14.5,'Segment-level aspect-based sentiment\nsegments: punctuation / sentence case\n8 seed-word lexicons (word boundaries,\ninflectional endings); VADER per segment',fs=7.3)
box(50,80,22,6.5,'A  Aspect sentiment (8)',fs=7.2)
box(74.5,80,20.5,6.5,'B  Overall sentiment (4)',fs=6.9)
box(50,71.5,22,6.5,'C  Review statistics (5)',fs=7.2)
box(74.5,71.5,20.5,6.5,'D  Metadata (9; one-hot)',fs=6.6)
box(5,63,90,5,'Validation against 23,113 human-annotated sentences (HRAST): aspect detection, VADER polarity',fc='#fff8ec',fs=7.0)
arrow(45.3,83,49.7,83)
band(19,39,'Stage 3   Modelling, statistical validation and explanation','#e8f6ef')
box(5,41.5,28,10.5,'Model comparison\n2 naive baselines, LR, SVM,\nRF, XGBoost, LightGBM\n50,000 reviews, 5-fold CV',fs=7.1)
box(36,41.5,28,10.5,'Representation comparison\nablation, TF-IDF, frozen\nsentence encoders, fine-tuned\nDistilBERT (same folds)',fs=7.1)
box(67,41.5,28,10.5,'Class-imbalance handling\n7 strategies versus\nclass weighting',fs=7.1)
box(5,33,90,5,'Leakage-safe CV · corrected t-test · Holm–Bonferroni · TOST (±0.02) · hotel-grouped CV · bootstrap CIs',fs=7.0)
box(5,20.5,43,9.5,'Explanation: TreeSHAP on the evaluated model,\n1,000 held-out reviews; stability across label\nschemes and model families',fs=7.1)
box(52,20.5,43,9.5,'Cross-platform transfer (14 shared features):\nunmatched, matched and strictly matched;\nrobustness and nationality audit',fs=7.1)

arrow(19,41.5,19,38.3); arrow(50,41.5,50,38.3); arrow(81,41.5,81,38.3)
arrow(26.5,33,26.5,30.3); arrow(73.5,33,73.5,30.3)
band(1,15,'Outputs','#f2f2f0')
box(5,3,27,8.5,'Performance with\nuncertainty and significance',fs=7.3)
box(36,3,28,8.5,'Feature-level explanations',fs=7.3)
box(68,3,27,8.5,'Cost of interpretability and\nlimits of transferability',fs=7.3)
plt.savefig('Fig1.png'); plt.close()

# ---------------- Fig 2 global SHAP by group ----------------
imp=pd.read_csv(R+'feature_importance_shap.csv').sort_values('mean_abs_shap')
A=['sent_Location','sent_Cleanliness','sent_Service','sent_Room','sent_Food_Beverage','sent_Value','sent_Facilities','sent_Noise']
B=['overall_sentiment','positive_sentiment','negative_sentiment','sentiment_gap']
Cg=['review_word_count','Review_Total_Negative_Word_Counts','Review_Total_Positive_Word_Counts','neg_pos_word_ratio','num_aspects_mentioned']
def grp(f): return 'A' if f in A else 'B' if f in B else 'C' if f in Cg else 'D'
cols={'A':C1,'B':C2,'C':C3,'D':C4}
names={'A':'A  Aspect sentiment','B':'B  Overall sentiment','C':'C  Review statistics','D':'D  Contextual metadata'}
fig,ax=plt.subplots(figsize=(6.3,6.6))
g=imp['feature'].map(grp)
ax.barh(range(len(imp)),imp['mean_abs_shap'],color=[cols[x] for x in g],height=0.72,edgecolor='white',linewidth=1)
ax.set_yticks(range(len(imp))); ax.set_yticklabels([f'{f}  ({x})' for f,x in zip(imp['feature'],g)],fontsize=7.5)
for i,(v,s) in enumerate(zip(imp['mean_abs_shap'],imp['share_pct'])):
    if i>=len(imp)-5: ax.text(v+0.006,i,f'{s:.1f}%',va='center',fontsize=7,color=MUTED)
ax.set_xlabel('Mean |SHAP value| (log-odds units, averaged over classes)')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.grid(axis='x',color='#e6e6e6',lw=0.6); ax.set_axisbelow(True)
tot=imp.groupby(g)['mean_abs_shap'].sum(); tot=tot/tot.sum()*100
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cols[k],label=f'{names[k]} ({tot[k]:.1f}%)') for k in 'ABCD'],loc='lower right',frameon=False,fontsize=7.5)
plt.savefig('Fig3.png'); plt.close()
print('group shares', tot.round(1).to_dict())

# ---------------- Fig 3 beeswarm composite ----------------
ims=[Image.open(F+f'fig_shap_beeswarm_{c}.png').convert('RGB') for c in ['Low','Medium','High']]
h=min(i.height for i in ims); ims=[i.resize((int(i.width*h/i.height),h)) for i in ims]
W=sum(i.width for i in ims); canvas=Image.new('RGB',(W,h+120),'white'); x=0
from PIL import ImageDraw, ImageFont
d=ImageDraw.Draw(canvas)
try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',90)
except: font=ImageFont.load_default()
for lab,i in zip('ABC',ims):
    canvas.paste(i,(x,120)); d.text((x+20,10),lab,fill='black',font=font); x+=i.width
canvas.save('Fig4.png',dpi=(600,600))
print('fig3 size',canvas.size)

# ---------------- Fig 4 dependence (copy) ----------------
im=Image.open(F+'fig_shap_dependence.png').convert('RGB'); im.save('FigS1.png',dpi=(600,600))

# ---------------- Fig 5 accuracy-cost trade-off ----------------
d=json.load(open(R+'plm_baselines.json'))['results']
order=[('26 features + RF','26 engineered features + RF',True),('TF-IDF + LR','TF-IDF + LR',False),
       ('MiniLM-L6 (frozen) + LR','MiniLM (frozen) + LR',False),('MPNet-base (frozen) + LR','MPNet (frozen) + LR',False),
       ('DistilBERT (fine-tuned)','DistilBERT (fine-tuned)',False)]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
for ax,metric,xl in [(axs[0],'f1_macro','Macro-F1'),(axs[1],'low','Low-class F1')]:
    pass
fig,ax=plt.subplots(figsize=(6.4,3.8))
for k,lab,interp in order:
    r=d[k]; x=r['fit_seconds_per_fold']+r.get('extraction_seconds',0)/5
    y=r['f1_macro']['mean']; lo,hi=r['f1_macro']['ci_low'],r['f1_macro']['ci_high']
    col=C1 if interp else C2; mk='o' if interp else 's'
    ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt=mk,color=col,ms=8,mec='white',mew=1.5,elinewidth=1.4,capsize=3,zorder=3)
    dx,dy={'26 features + RF':(1.35,-0.012),'TF-IDF + LR':(0.72,0.010),'MiniLM-L6 (frozen) + LR':(1.35,0),'MPNet-base (frozen) + LR':(1.35,0),'DistilBERT (fine-tuned)':(0.42,0)}[k]
    ha='left' if dx>1 else 'right'
    ax.text(x*dx,y+dy,f'{lab}\n{y:.3f}',va='center',ha=ha,fontsize=7.5,color=INK)
ax.set_xscale('log'); ax.set_xlim(0.3,3000)
ax.set_xlabel('Compute per fold, seconds (log scale): training + 1/5 of representation cost')
ax.set_ylabel('Macro-F1 (mean, 95% CI over 5 folds)')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.grid(color='#e6e6e6',lw=0.6); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=C1,ms=7,label='26 engineered features (semantically defined)'),
                   Line2D([],[],marker='s',ls='',color=C2,ms=7,label='text representations')],loc='upper left',bbox_to_anchor=(0.0,0.86),frameon=False,fontsize=7.5)
plt.savefig('Fig2.png'); plt.close()
for k,lab,_ in order:
    r=d[k]; print(k, round(r['fit_seconds_per_fold']+r.get('extraction_seconds',0)/5,2))

# ---------------- Fig S1 learning curve (copy) ----------------
Image.open(F+'fig_learning_curve.png').convert('RGB').save('FigS2.png',dpi=(600,600))
