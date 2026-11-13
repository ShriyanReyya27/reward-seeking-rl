"""Render learning curves and full-versus-partial credit using the saved analyses."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES={"aghyad430":"Aghyad 430","control430":"Control 430","untrained":"Untrained",
       "GLM-5.3":"GLM-5.3","GLM-5.3-Flash":"GLM-5.3 Flash","Kimi-K2.6":"Kimi K2.6"}
COLORS={"oss430":"#006a67","c430":"#90512e"}


def plots(directory):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.spines.left":False,"axes.axisbelow":True})
    curves=json.loads((directory/"rl_summary.json").read_text())["curves"]
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.15),sharey=True,constrained_layout=True)
    for ax,sub,title in zip(axes,("all","format"),("All four families","Held-out format family")):
        for arm in ("oss430","c430"):
            for seed,style in ((1,"-"),(2,"--")):
                rows=sorted((r for r in curves if (r["subset"],r["framing"],r["arm"],r["seed"])==
                             (sub,"plain",arm,seed)),key=lambda r:r["step"])
                change=[100*(r["gain"]-rows[0]["gain"]) for r in rows]
                ax.plot([r["step"] for r in rows],change,style,marker="o" if seed==1 else "s",
                        markersize=3,color=COLORS[arm],linewidth=1.7,
                        label=("Aghyad" if arm=="oss430" else "Control")+f" · seed {seed}")
        ax.axhline(0,color="#747975",linewidth=.7)
        ax.grid(axis="y",color="#d9e4e2");ax.set_title(title,loc="left",weight="bold",fontsize=10)
        ax.set_xlabel("RL updates");ax.set_xticks(list(range(0,41,10)))
    axes[0].set_ylabel("Change in evidence benefit (pp)")
    axes[1].legend(fontsize=7,ncol=2,frameon=False)
    fig.savefig(directory/"rl_curves.png",dpi=190);plt.close(fig)
    scores=json.loads((directory/"coding_summary.json").read_text())["all_masks"]["arms"]
    order=["untrained","aghyad430","control430","GLM-5.3","GLM-5.3-Flash","Kimi-K2.6"]
    fig,ax=plt.subplots(figsize=(8.5,2.85),constrained_layout=True)
    xpos=np.arange(len(order))
    for off,key,color,label in ((-.18,"full_pass","#006a67","All five groups passed"),(.18,"partial_credit","#9bbbb5","Mean partial credit")):
        vals=[100*scores[a][key] for a in order]
        rect=ax.bar(xpos+off,vals,.34,color=color,label=label)
        ax.bar_label(rect,fmt="%.0f",padding=2,fontsize=8)
    ax.set_ylabel("Percent");ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.grid(axis="y",color="#d9e4e2")
    ax.set_xticks(xpos,labels=[NAMES[a] for a in order])
    ax.legend(frameon=False,ncol=2,loc="upper right",fontsize=8)
    fig.savefig(directory/"coding_screen.png",dpi=190);plt.close(fig)


if __name__=="__main__":
    plots(Path(sys.argv[1]))
