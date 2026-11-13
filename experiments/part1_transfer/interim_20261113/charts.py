import csv,json
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
snap=json.loads((HERE/"snapshot.json").read_text())
colors=["#126A70","#A85226"]
def bars(file,labels,series,keys,unit="%",height=2.05):
    fig,ax=plt.subplots(figsize=(7.2,height))
    x=np.arange(len(labels))
    width=.31
    for i,values in enumerate(series):
        b=ax.bar(x+(i-.5)*width,np.array(values)*100,width,color=colors[i],label=keys[i])
        ax.bar_label(b,fmt="%.1f",padding=2,fontsize=8.5)
    ax.set_ylim(0,114);ax.set_yticks([0,50,100]);ax.set_ylabel(unit)
    ax.set_xticks(x,labels,fontsize=9);ax.tick_params(length=0)
    ax.spines[["right","top","left"]].set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis="y",color="#d6dcdb",lw=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.55,1.01))
    fig.tight_layout(rect=(0,0,1,.86),pad=.35)
    fig.savefig(HERE/(file+".png"),dpi=160,bbox_inches="tight");plt.close(fig)
b=snap["b"]
bars("coding_reward",["Cue-only training","Stated training"],[[b[a+" "+v]["average"] for v in ("cue","stated")] for a in ("aghyad430","control430")],["Aghyad 430","Control 430"])
rows=list(csv.DictReader((HERE.parents[1]/"coding_transfer/results/preference_levels.csv").open()))
bars("preference",["All families\nScored examples","All families\nNo examples","Held-out format\nScored examples","Held-out format\nNo examples"],[[mean(float(r[m]) for r in rows if r["arm"]==a and r["subset"]==sub) for sub in ("all","format") for m in ("shown_average","no_scores_average")] for a in ("oss430","c430")],["Aghyad 430","Control 430"],height=2.4)
bar=snap["feedback"]["2"]["arms"]
bars("feedback",["Attempt 1","Attempt 4","Mean with cue","Mean with stated rules"],[[bar[a]["cue"]["first"],bar[a]["cue"]["last"],bar[a]["cue"]["reward"],bar[a]["stated"]["reward"]] for a in ("aghyad430","control430")],["Aghyad 430","Control 430"])
