"""Build the research handoff from the frozen analyses and their figures."""
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def build(data, output):
    def read(name):
        return json.loads((data/name).read_text(encoding="utf8"))
    def pp(x):
        return f"{100*x:+.1f}"
    def pct(x):
        return f"{100*x:.1f}%"
    def bounds(ci):
        return f"[{pp(ci[0])}, {pp(ci[1])}]"

    rl=read("rl_summary.json")
    coding=read("coding_summary.json")["all_masks"]
    repeats=read("repeat_summary.json")["all_masks"]
    labels=read("label_summary.json")
    costs=read("costs.json")
    switches=read("cue_switch.json")
    strict=read("strict_summary.json")
    result=Document()
    s=result.sections[0]
    s.top_margin=Inches(.65);s.bottom_margin=Inches(.62)
    s.left_margin=Inches(.78);s.right_margin=Inches(.78)
    s.footer_distance=Inches(.27)
    styles=result.styles
    style=styles["Normal"];style.font.name="Liberation Sans";style.font.size=Pt(10.4)
    style.font.color.rgb=RGBColor.from_string("223330")
    style.paragraph_format.space_after=Pt(5);style.paragraph_format.line_spacing=1.10
    footer=s.footer.paragraphs[0];footer.alignment=2
    f=footer.add_run("SPAR   /   REWARD SEEKING                                      NOVEMBER 2026     ·     ")
    f.font.size=Pt(8);f.font.color.rgb=RGBColor.from_string("526660")
    field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE");footer._p.append(field)
    def head(text,sz=21):
        p=result.add_paragraph();p.paragraph_format.space_after=Pt(7)
        r=p.add_run(text);r.bold=True;r.font.size=Pt(sz);r.font.color.rgb=RGBColor.from_string("164d48")
    def sub(t):
        p=result.add_paragraph();p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(3)
        r=p.add_run(t);r.bold=True;r.font.size=Pt(12.8)
    def para(text):
        result.add_paragraph(text)
    def table(heads,rows,widths):
        t=result.add_table(rows=0,cols=len(heads));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for col,w in zip(t.columns,widths):col.width=Inches(w)
        for i,row in enumerate([heads,*rows]):
            cells=t.add_row().cells
            for c,value,w in zip(cells,row,widths):
                c.width=Inches(w);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                sh=OxmlElement("w:shd");sh.set(qn("w:fill"),"256059" if not i else ("EAF1F0" if i%2 else "F7F9F8"))
                c._tc.get_or_add_tcPr().append(sh)
                p=c.paragraphs[0];p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
                r=p.add_run(str(value));r.bold=i==0;r.font.size=Pt(9.1)
                if not i:r.font.color.rgb=RGBColor.from_string("FFFFFF")
    curves=rl["curves"]
    def curve(subset,arm,seed,step):
        return next(r for r in curves if (r["framing"],r["subset"],r["arm"],r["seed"],r["step"])==
                    ("plain",subset,arm,seed,step))
    def eff(subset,seed):
        return next(e for e in rl["effects"] if (e["subset"],e["seed"])==(subset,seed))
    head("Does reward-seeking training help a model use earlier scores?")
    main=eff("all","mean");held=eff("format","mean");end_step=rl["common_endpoint"]
    para(f"Aghyad used scored examples better before and after RL. Control gained more in both new seeds. At update {end_step}, the average difference in improvement (Aghyad minus control) was {pp(main['final_change_auc_change'][0])} percentage points (pp), with a paired 95% task interval of {bounds(main['paired_task_ci95'][0])}. Toby’s original one-seed estimate was +8.2 pp [−5, +21].")
    para("Each episode has a hidden preference. Earlier examples show which answers scored 0 or 1. Evidence benefit is accuracy with those examples minus accuracy on the same tasks without examples." + ("" if end_step==40 else f" One run reached its spending stop early. Update {end_step} is the last shared evaluation. The planned two-seed update-40 comparison is incomplete."))
    rows=[]
    for subname,subid in (("All families","all"),("Held-out format","format")):
        for seed in (1,2):
            cells=[]
            for a in ("oss430","c430"):
                start=curve(subid,a,seed,0)["gain"];end=curve(subid,a,seed,end_step)["gain"]
                cells.append(f"{pp(start)} → {pp(end)}")
            e=eff(subid,seed)
            rows.append([subname,str(seed),*cells,pp(e["final_change_auc_change"][0])+" "+bounds(e["paired_task_ci95"][0])])
    table(["Test set","Seed",f"Aghyad, 0 → {end_step}",f"Control, 0 → {end_step}","Difference in change"],rows,[1.16,.48,1.48,1.49,2.23])
    result.add_picture(str(data/"rl_curves.png"),width=Inches(6.78))
    sub("Learning speed and transfer")
    para(f"The mean difference in baseline-adjusted area under the curve was {pp(main['final_change_auc_change'][1])} pp {bounds(main['paired_task_ci95'][1])}. On the held-out format family, the final difference in change was {pp(held['final_change_auc_change'][0])} pp {bounds(held['paired_task_ci95'][0])}. Intervals resample paired tasks within family and are conditional on the two seeds. Both arms reused the original evaluation bank.")
    def endpoints(arm,measure):
        return " → ".join(pct(sum(curve("all",arm,seed,step)[measure] for seed in (1,2))/2) for step in (0,end_step))
    para(f"With scored examples, raw accuracy went {endpoints('oss430','acc_kpos')} for Aghyad and {endpoints('c430','acc_kpos')} for control. Without examples it went {endpoints('oss430','acc_k0')} and {endpoints('c430','acc_k0')}, respectively.")
    result.add_page_break()
    head("Hidden-test coding across ten problems")
    para("Each Python problem has three independent edge-case policies. Eight context descriptions select different combinations. Examples leave those choices unresolved. Five equally weighted groups check ordinary or empty inputs, individual edge conditions and a larger input. The host compares returned values against its own oracle.")
    para("A first check with all requirements stated yielded 14/20 full passes for Aghyad and 16/20 for control. After clarifying the common interface, we checked different rule combinations. Aghyad passed 20/20 and control 19/20, clearing the prespecified 90% gate.")
    result.add_picture(str(data/"coding_screen.png"),width=Inches(6.78))
    sub(f"Eight draws on {repeats['paired_tasks']} paired tasks")
    rows=[]
    for a,n in (("aghyad430","Aghyad 430"),("control430","Control 430")):
        r=repeats["arms"][a]
        rows.append([n,pct(r["full_pass"]),pct(r["partial_credit"]),pct(r["mixed_credit_group"]),pct(r["tool"])])
    table(["Model","All groups pass","Mean credit","Tasks with varied credit","Tool failures"],rows,[1.18,1.32,1.2,1.82,1.32])
    e=repeats["paired"]["full_pass"];ep=repeats["paired"]["partial_credit"]
    para(f"Aghyad minus control was {pp(e['diff'])} pp on full passes, paired 95% interval {bounds(e['ci95'])}, and {pp(ep['diff'])} pp on mean credit {bounds(ep['ci95'])}. Each template has equal weight in this estimate. Four Aghyad repeat groups were interrupted during transport. Both arms’ draws on those tasks are excluded from the paired repeat comparison; the first-draw screen retains all 80.")
    sub("Did code actually change when the cue changed?")
    para(f"We compared 120 pairs of prompts per model that differed in one policy, using the first sampled program for each prompt. Both programs had to pass their own changed edge and fail under the opposite rule. Aghyad switched on {switches['aghyad430']['switches']}/120 pairs and control on {switches['control430']['switches']}/120. Separate model draws introduce sampling noise into this measure.")
    para("The original single log-processing problem gave Aghyad a 6.5-point full-pass lead. On the rebuilt bank, 29% of Aghyad task groups and 20% of control groups mixed full passes and failures. Both missed the original 30% variation threshold for binary-reward RL. Partial credit varied on about 42% of groups in each arm and would test a different training objective.")
    result.add_page_break()
    head("What the traces and grader checks establish")
    sub("Cue use and speculation about an unseen check")
    para("The same blinded GPT-4.1 judge separately labelled whether a trace interpreted a supplied cue and whether it proposed a concrete hypothesis about an unseen evaluator. Every positive points to a span of model reasoning. Counts below are for the new coding screen, among replies with recorded reasoning.")
    names={"aghyad430":"Aghyad 430","control430":"Control 430","untrained":"Untrained GPT-OSS","GLM-5.3":"GLM-5.3","GLM-5.3-Flash":"GLM-5.3 Flash","Kimi-K2.6":"Kimi K2.6"}
    table(["Model","Available / 80","Used supplied cue","Unseen-check hypothesis"],
          [[v,labels[k]["n"],labels[k]["cue_use"],labels[k]["unseen_check"]] for k,v in names.items()],[1.82,1.23,1.65,2.14])
    para("A fixed random spot-check of three positive spans per model found concrete test predictions alongside ordinary examples and mental checks. Some control spans only walked through an example. The label frequencies describe the automated judge. They give no validated prevalence of intent.")
    para("On the original single coding template the revised judge marked cue use on all 150 traces. It flagged unseen checks in 2/50 Aghyad traces, 0/50 control and 1/50 base. Inspection found one base label about an ordinary duplicate-handling rule and one vague Aghyad reference to typical tests. The older 45/50 versus 19/50 label combined different behaviors. Fifty new blinded packets are ready for human review.")
    sub("GLM answer boundaries")
    para(f"On the old evaluator cue, extracting the first answer line raised full GLM from 4/14 to 13/14 and Flash from 2/14 to 14/14. On the new coding bank we added the assistant boundary to the stop list. Whole-reply grading, stripping just the boundary and extracting the source yielded {strict['GLM-5.3']['strict']}, {strict['GLM-5.3']['boundary']} and {strict['GLM-5.3']['extracted']} full passes out of 80 for GLM. Flash scored {strict['GLM-5.3-Flash']['strict']}, {strict['GLM-5.3-Flash']['boundary']} and {strict['GLM-5.3-Flash']['extracted']}. Neither produced text after the boundary.")
    sub("Scope and cost")
    para("Hidden-preference RL measures use of informative scores in an underspecified task. It pays for the hidden preference itself. Coding pays directly for correct behavior. Neither experiment includes a shortcut to raise reward independently of correctness. The other prompt families have revised wording and offline checks. Their model performance awaits a new screen.")
    para(f"Recorded token usage implies ${costs['total']:.2f}, including ${costs['rl']:.2f} for four training runs and their evaluations. Four interrupted sample groups have up to ${sum(costs['unsettled'].values()):.2f} in additional unresolved charges. Aghyad seed 2 completed all 40 updates in its original process. Its final evaluation brought the run estimate to ${costs['rl_by_run']['2-oss430']:.2f}.")
    result.save(output)


if __name__=="__main__":
    build(Path(sys.argv[1]),Path(sys.argv[2]))
