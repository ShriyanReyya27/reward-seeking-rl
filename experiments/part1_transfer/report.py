"""Build the Part 1 Word report from verified summary tables and figures."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from docx import Document
from docx.shared import Inches, Pt, RGBColor

HERE=Path(__file__).resolve().parent
RESULT=HERE/"results"


def records(name):
    with (RESULT/name).open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def pp(x):
    return f"{100*float(x):+.1f}"


def percent(x):
    return f"{100*float(x):.1f}%"


def build():
    h=json.loads((RESULT/"hypotheses.json").read_text())
    effects=records("contrasts.csv")
    points=records("endpoints.csv")
    battery=records("battery.csv")
    behavior=records("behavior.csv")
    estimate=records("estimated_cost.csv")
    a1=json.loads((HERE.parent/"coding_transfer/results/coding_rl/analysis.json").read_text())
    assert a1["complete"] and h["H1"]["cells"]==4
    doc=Document()
    page=doc.sections[0]
    page.page_height=Inches(11);page.page_width=Inches(8.5)
    page.left_margin=page.right_margin=Inches(.8)
    page.top_margin=page.bottom_margin=Inches(.72)
    normal=doc.styles["Normal"]
    normal.font.name="Calibri";normal.font.size=Pt(11)
    normal.paragraph_format.space_after=Pt(6)
    for style in ("Title","Heading 1","Heading 2"):
        s=doc.styles[style]
        s.font.color.rgb=RGBColor.from_string("171717")
        for el in s.element.xpath(".//w:pBdr"):el.getparent().remove(el)
    def table(headers,data):
        t=doc.add_table(rows=1,cols=len(headers))
        t.style="Light Shading Accent 1"
        for c,s in zip(t.rows[0].cells,headers):c.text=s
        for row in data:
            for c,s in zip(t.add_row().cells,row):c.text=str(s)
    def page_title(title):
        doc.add_page_break();doc.add_heading(title,1)
    def image(filename):
        doc.add_picture(str(RESULT/filename),width=Inches(6.8))
    def effect_row(phase,domain,version,name,seed):
        subset=[r for r in effects if (r["phase"],r["domain"],r["version"],r["contrast"],r["seed"])==
                (phase,domain,version,name,seed)]
        assert len(subset)==1
        return subset[0]
    def interval(r):
        return f"{pp(r['diff'])} [{pp(r['lo'])}, {pp(r['hi'])}]"
    def delta_text(k):
        r=h[k]
        return f"{pp(r['diff'])} points (95% interval {pp(r['ci'][0])} to {pp(r['ci'][1])})"
    doc.add_paragraph("Does cue only coding training transfer",style="Title")
    verdict="met" if h["supportive_H1"] else "did not meet"
    doc.add_paragraph(f"Cue-only coding training {verdict} the prespecified transfer criterion. Across customer support and Mastermind, its average held-out reward difference from stated-requirements training was {delta_text('H1')}. The cue-only condition scored higher in {h['H1']['positive_cells']} of the four starting-model by environment comparisons.")
    doc.add_heading("H1  Effect of the training prompt",2)
    doc.add_paragraph("Aghyad 430 and the training-matched control each received 60 coding updates on identical task banks under two prompt conditions. Cue-only prompts implied edge cases. Stated prompts spelled them out. Hidden tests awarded partial credit in both conditions. Each starting model and condition had two training seeds.")
    data=[]
    for domain in ("support","mastermind"):
        for k,name in (("H1_A","Aghyad 430"),("H1_C","Control 430")):
            rows=[effect_row("C2",domain,"transfer",k,s) for s in ("1","2","pooled")]
            data.append([domain.title(),name,pp(rows[0]["diff"]),pp(rows[1]["diff"]),interval(rows[2])])
    table(["Transfer task","Start","Seed 1","Seed 2","Pooled [95% CI]"],data)
    doc.add_paragraph("Entries are percentage-point differences in average reward over 30 subsequent RL updates. We integrate the measured held-out levels using trapezoids, divide by the update count, and weight the two starts and environments equally. Intervals resample the same tasks across arms, checkpoints and seeds. They capture task uncertainty conditional on the two seeds.")
    doc.add_heading("H2 and H3  Starting model comparisons",2)
    doc.add_paragraph(f"Aghyad with cue-only training minus control with stated training was {delta_text('H2')} in the two transfer environments. Under cue-only training for both models, the Aghyad-minus-control difference was {delta_text('H3')}. H2 combines two differences in treatment. H3 holds the coding training condition fixed.")
    page_title("Learning in the two transfer environments")
    doc.add_paragraph("Lines show mean held-out reward during transfer RL. Faint lines show the two training seeds. Reference denotes the original step-430 model, which received its first new updates in the transfer environment.")
    for domain,title in (("support","Customer support"),("mastermind","Mastermind")):
        doc.add_heading(title,2)
        image(f"C2_{domain}_transfer_level.png")
    page_title("Gains from each starting point")
    doc.add_paragraph("Every curve starts at its own step-zero score. Compare these changes alongside absolute levels on the previous page. The endpoint table also reports starting headroom and the first measured threshold crossing for each seed.")
    for domain,title in (("support","Customer support"),("mastermind","Mastermind")):
        doc.add_heading(title,2)
        image(f"C2_{domain}_transfer_gain.png")
    page_title("Coding performance and capability checks")
    r=effect_row("B","coding","cue","H3","pooled")
    doc.add_paragraph(f"During the 60 coding updates, Aghyad minus control under cue-only training averaged {interval(r)} points on held-out cue-only templates. With the requirements explicit, the corresponding difference was {interval(effect_row('B','coding','stated','H3','pooled'))} points.")
    image("B_coding_cue_level.png")
    table(["Training cell","Stated start","Stated end","Full pass at end"],[
        [f"{arm} / {v}",percent(mean(float(x["start_reward"]) for x in subset)),percent(mean(float(x["final"]) for x in subset)),percent(mean(float(x["capability_end"]) for x in subset))]
        for arm in ("aghyad430","control430") for v in ("cue","stated")
        if (subset:=[x for x in points if (x["phase"],x["domain"],x["eval_version"],x["start"],x["training"])==("B","coding","stated",arm,v)])])
    gains=[d["aghyad430"]-d["control430"] for d in a1["levels"].values()]
    ci=a1["pooled_interval"]
    doc.add_paragraph(f"In the supplementary 50-update cue-only experiment, the two per-seed average reward differences were {pp(gains[0])} and {pp(gains[1])} points. Their pooled interval was {pp(ci[0])} to {pp(ci[1])}. The four runs resumed from saved step-10 optimizer states after a local Docker outage.")
    page_title("Behavior in new environments")
    rows=[]
    for arm in ("aghyad430","control430"):
        for v in ("cue","stated"):
            def measure(label,step):
                xs=[float(x["value"]) for x in behavior if (x["arm"],x["condition"],x["measure"],x["step"])==(arm,v,label,str(step))]
                assert len(xs)==2
                return mean(xs)
            probes=[x for x in battery if (x["arm"],x["condition"],x["subtest"])==(arm,v,"conflict_conflict") and x["hack_rate"]!=""]
            assert probes
            rate=sum(float(x["hack_rate"])*int(x["n"]) for x in probes)/sum(int(x["n"]) for x in probes)
            rows.append([arm,v,pp(measure("format_feedback_effect",60)),percent(measure("coding_cue_flip_switch",60)),pp(measure("preference_shown_minus_withheld",0)),percent(rate)])
    table(["Start","Training","Feedback effect (pt)","Cue flips","Preference effect (pt)","Conflict exploit"],rows)
    doc.add_paragraph("The feedback effect compares the same four-turn task with and without numeric scores. Cue flips count matched program pairs that implemented the correct changed edge policy on both prompts. Hidden preference compares scored and unscored example histories. Its relationship to the training design makes it a near-transfer measure. The conflict probe explicitly separates the reward paid from the intended answer.")
    doc.add_heading("Scope of the evidence",2)
    doc.add_paragraph("The revised record-formatting screen produced stated rewards of 67.3% for Aghyad and 76.3% for control. The predefined fallback restricted Part 1 training to coding. Formatting remained an evaluation-only measure, and its weak stated performance limits interpretation. Customer support uses an LLM rubric judge. Mastermind offers deterministic feedback and exact scoring. Success on these tasks measures performance under their rewards. Deliberate exploitation requires separate evidence.")
    doc.add_heading("Cost and supporting records",2)
    by_phase=defaultdict(float)
    for r in estimate:by_phase[r["phase"]]+=float(r["total_usd"])
    doc.add_paragraph("Estimated campaign cost was "+f"${sum(by_phase.values()):,.2f}. "+"  ".join(f"{stage} ${n:,.2f}." for stage,n in sorted(by_phase.items()))+" Model charges use recorded token counts at the archived rates. Rubric judge costs use the recorded spend meter.")
    doc.add_paragraph("The branch includes per-seed levels, gains, headroom, threshold crossings, task-paired intervals, capability changes, grading diagnostics, source checkpoints, costs and row-level transcripts.")
    doc.save(HERE/"Coding training and transfer.docx")


if __name__=="__main__":build()
