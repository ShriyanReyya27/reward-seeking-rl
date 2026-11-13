"""Plot measured reward and within-cell changes against actual update numbers."""
import csv
from itertools import product
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS=Path(__file__).resolve().parent/"results"
PALETTE={"cue":"#246b7c","stated":"#ab4b32","reference":"#666666"}


def draw(rows,phase,domain,version,gain):
    group=[r for r in rows if (r["phase"],r["domain"],r["eval_version"])==(phase,domain,version)]
    if not group:return
    fig,axes=plt.subplots(1,2,sharex=True,sharey=True,figsize=(7.3,3.3))
    for ax,arm in zip(axes,("aghyad430","control430")):
        for condition,color in PALETTE.items():
            traces=[]
            for seed in ("1","2"):
                series=sorted((r for r in group if (r["start"],r["training"],r["seed"])==(arm,condition,seed)),key=lambda r:int(r["step"]))
                if not series:continue
                x=[int(r["step"]) for r in series]
                base=float(series[0]["reward"]) if gain else 0
                y=[100*(float(r["reward"])-base) for r in series]
                traces.append(y)
                ax.plot(x,y,color=color,lw=.8,alpha=.36)
            if traces:
                assert len(traces)==2
                ax.plot(x,[mean(v) for v in zip(*traces,strict=True)],color=color,lw=2,marker="o",ms=3,label=condition.capitalize())
        ax.set_title("Aghyad 430" if arm=="aghyad430" else "Control 430",fontsize=11)
        ax.set_xlabel("RL updates")
        ax.grid(axis="y",lw=.5,alpha=.45)
        ax.spines[["right","top"]].set_visible(False)
        if gain:ax.axhline(0,color="#444444",lw=.6)
        elif domain!="support":ax.set_ylim(0,103)
    axes[0].set_ylabel("Change from step 0 (points)" if gain else "Held-out reward (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=len(labels),frameon=False)
    fig.tight_layout(rect=(0,.08,1,1))
    suffix="gain" if gain else "level"
    fig.savefig(RESULTS/f"{phase}_{domain}_{version}_{suffix}.png",dpi=175)
    plt.close(fig)


if __name__=="__main__":
    with (RESULTS/"curves.csv").open(encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    for (phase,domain,version),gain in product(sorted({(r["phase"],r["domain"],r["eval_version"]) for r in rows}),(False,True)):
        draw(rows,phase,domain,version,gain)
