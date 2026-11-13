"""Render the November 13 Part 1 interim snapshot, independently of live jobs."""
import json
from pathlib import Path
from statistics import mean
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches,Pt,RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

here=Path(__file__).resolve().parent
s=json.loads((here/"snapshot.json").read_text())
d=Document();sec=d.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.left_margin=sec.right_margin=Inches(.8);sec.top_margin=Inches(.68);sec.bottom_margin=Inches(.65)
sec.header_distance=Inches(.27);sec.footer_distance=Inches(.27)
font="Liberation Sans"
normal=d.styles["Normal"];normal.font.name=font;normal.font.size=Pt(11)
normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.line_spacing=1.12
for style,size,color in [("Title",25,"161F22"),("Heading 1",21,"161F22"),("Heading 2",13,"195D61")]:
    st=d.styles[style];st.font.name=font;st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string(color)
    st.paragraph_format.space_before=Pt(8);st.paragraph_format.space_after=Pt(5)
    for el in st.element.xpath(".//w:pBdr"):el.getparent().remove(el)
sec.header.paragraphs[0].add_run("SPAR  /  REWARD SEEKING RESEARCH").font.size=Pt(8)
f=sec.footer.paragraphs[0];f.alignment=2
f.add_run("Part 1 interim results   ·   13 November 2026   ·   ").font.size=Pt(8)
field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE");f._p.append(field)
def h(t):d.add_heading(t,2)
def page(t):d.add_page_break();d.add_heading(t,1)
def p(t):d.add_paragraph(t)
def pic(name):
    par=d.add_paragraph();par.alignment=1;par.paragraph_format.space_after=Pt(1)
    par.add_run().add_picture(str(here/(name+".png")),width=Inches(6.5))
def table(header,rows,widths):
    t=d.add_table(rows=0,cols=len(header));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for col,w in zip(t.columns,widths):col.width=Inches(w)
    for i,row in enumerate([header,*rows]):
        for k,(c,v,w) in enumerate(zip(t.add_row().cells,row,widths)):
            c.width=Inches(w);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            par=c.paragraphs[0];par.alignment=0 if k==0 else 1
            par.paragraph_format.space_before=par.paragraph_format.space_after=Pt(3)
            run=par.add_run(str(v));run.bold=i==0;run.font.size=Pt(10)
            sh=OxmlElement("w:shd");sh.set(qn("w:fill"),"DDECEB" if i==0 else "F2F5F4" if i%2 else "FFFFFF");c._tc.get_or_add_tcPr().append(sh)
def sources(items):
    par=d.add_paragraph();par.paragraph_format.space_before=Pt(4)
    for i,(label,path) in enumerate(items):
        if i:par.add_run("    ·    ").font.size=Pt(9)
        rel=d.part.relate_to("https://github.com/ShriyanReyya27/reward-seeking-rl/"+path,RT.HYPERLINK,is_external=True)
        a=OxmlElement("w:hyperlink");a.set(qn("r:id"),rel);r=OxmlElement("w:r");pr=OxmlElement("w:rPr")
        for tag,key,val in [("color","val","17686D"),("u","val","single"),("sz","val","18")]:
            el=OxmlElement("w:"+tag);el.set(qn("w:"+key),val);pr.append(el)
        r.append(pr);tx=OxmlElement("w:t");tx.text=label;r.append(tx);a.append(r);par._p.append(a)
bp="blob/codex/speculation-stage1/experiments/"
tp="tree/codex/speculation-stage1/experiments/"
pct=lambda v:f"{100*v:.1f}"
d.add_paragraph("Reward seeking and transfer in Part 1",style="Title")
p("The clearest completed result is on hidden preferences. Across three seeds, Aghyad 430 earned 20.8 percentage points more average accuracy than the training-matched control when earlier examples carried scores. On the same final requests without examples, the gap was 2.0 points. The main test of transfer to customer support and Mastermind is still pending.")
h("Current coding experiment")
p("Aghyad 430 and Control 430 each continue from their own 430-update checkpoint. In cue-only training, a user’s context implies three edge-case policies. In stated training, the prompt spells them out. Both conditions use the same hidden tests and award one fifth of the reward for each group passed.")
h("Average held-out coding reward through update 20")
pic("coding_reward")
p("Seed 1 of two, 48 held-out tasks across six templates. Each bar is mean partial credit on cue-only prompts, integrated over updates 0, 10 and 20 and divided by 20. The planned endpoint is 60 updates.")
table(["Training cell","Cue-only at 0","Cue-only at 20","Stated at 20"],[[name,pct(s["b"][k]["eval"]["0"]["cue"]["partial"])+"%",pct(s["b"][k]["eval"]["20"]["cue"]["partial"])+"%",pct(s["b"][k]["eval"]["20"]["stated"]["partial"])+"%"] for name,k in [("Aghyad · cue","aghyad430 cue"),("Control · cue","control430 cue"),("Aghyad · stated","aghyad430 stated"),("Control · stated","control430 stated")]],[2.15,1.5,1.62,1.53])
h("What the early comparison supports")
p("Under identical cue-only training, Aghyad leads control by 7.3 points on average (paired-task 95% interval +2.6 to +12.0). Within each model, cue-only training trails stated training by 5.4 points for Aghyad and 5.0 for control. Aghyad’s two conditions already differed by 5.8 points on the first evaluation of the same starting weights. These interim intervals cover task sampling within this seed.")
sources([("Training plan and decision rules",bp+"part1_transfer/PLAN.md"),("Snapshot data",bp+"part1_transfer/interim_20261113/snapshot.json")])
page("Learning preferences from scored examples")
p("In Toby’s plain hidden-preference task, earlier choices and their scores provide clues about what the user prefers. We repeated training on three seeds for 40 updates. The format family was held out of training. The capability check asks the same final question without earlier examples.")
pic("preference")
h("The advantage was consistent in level")
table(["Seed","All families with scores\nAghyad / control","Held-out format with scores\nAghyad / control"],[
    ["1","77.1% / 57.5%","95.3% / 70.7%"],["2","78.3% / 62.7%","97.7% / 72.3%"],["3","74.4% / 47.4%","95.0% / 48.9%"]],[.9,2.85,3.05])
p("The pooled 95% intervals for Aghyad minus control were +16.7 to +24.7 points across all families and +21.7 to +42.5 on held-out format. Without examples, the all-family interval was −2.1 to +6.3. The scored versus no-examples comparison changes both the examples and the availability of scores. The upcoming near-transfer battery keeps example histories fixed and withholds just the scores.")
h("Headroom changes the learning-rate interpretation")
p("Aghyad started at 86–88% on held-out format, near its explicit-framing ceiling. Control started at 50–54%. Control’s format gains varied from +38.9 and +39.6 points in seeds 1 and 2 to −5.6 in seed 3. Aghyad finished ahead in all three. Greater average accuracy is clear here. Unequal headroom makes raw gain a poor measure of relative learning speed.")
p("Toby’s earlier seed 0 moved from about 66% to 77% for Aghyad and 47% to 49% for control across all families. Its task-level files were unavailable. The pooled estimate uses the three replication seeds. We selected average accuracy after seeing seeds 1–2 and before seed 3.")
sources([("Per-seed endpoints and thresholds",bp+"coding_transfer/results/preference_levels.csv"),("Paired intervals",bp+"coding_transfer/results/preference_contrasts.csv")])
page("How the coding test changed")
p("The first screen tested five task families on 50 paired items each, with five prompt versions and eight draws. It yielded 29,912 completed episodes. Eleven feedback groups were unresolved. The table shows success with the consumer cue, using final-attempt success for feedback.")
table(["Original family","Base","Aghyad","Control"],[["Hidden-test coding","70.3%","97.3%","90.8%"],["Feedback over three attempts","88.5%","89.5%","97.3%"],["Evaluator cue","100%","100%","100%"],["Stated payoffs","99.5%","100%","100%"],["Platform format","48.5%","50.0%","51.8%"]],[2.85,1.15,1.4,1.4])
p("The original coding task asked for one log-consolidation function. Its +6.5-point gap had a paired 95% interval of −1.0 to +14.3. On feedback, control led by 7.7 points. The evaluator and payoff tasks reached ceiling. On platform format, even explicit requirements gave Aghyad 49.8% and control 59.0%.")
h("Harder pilots and the coding rebuild")
p("Mentioning grading had no consistent effect. On a harder splice task that combined a median calculation with output formatting, Aghyad scored 70.8% and control 59.9% with the cue. Only 20.8% and 16.7% of eight-draw groups had mixed outcomes. The harder evaluator still reached 100% for both arms when stated.")
p("The next coding bank used ten templates with independently varied edge policies. Across eight draws on 76 paired items, Aghyad and control fully passed 41.2% and 39.0%, and earned 76.2% and 79.6% partial credit. Both difference intervals included zero. When one cue flipped, their programs switched correctly on 51/120 and 61/120 pairs. Four incomplete Aghyad groups were excluded from both arms.")
h("Supplementary cue-only RL on 20 templates")
p("The final bank uses 14 training and six held-out templates. Independent reference implementations reproduced 720 checks. Correcting a detector that flagged comments gave each arm 37/40 full passes on the new-template stated preflight. Pagination prompts now specify the required list-of-lists output.")
rows=[]
for seed,end in [("s1",30),("s2",10)]:
    vals=[]
    for arm in ("aghyad430","control430"):
        e=s["a1"][seed+" "+arm]["eval"];v=[e[str(n)]["cue"]["reward"] for n in range(0,end+1,5)]
        vals.append((v[0],v[-1],mean((a+b)/2 for a,b in zip(v,v[1:]))))
    rows.append([f"{seed[1:]} · steps 0–{end}",*[f"{pct(vals[0][i])} / {pct(vals[1][i])}" for i in range(3)]])
table(["Seed and window","Start % (A / C)","Last % (A / C)","Average % (A / C)"],rows,[1.85,1.55,1.55,1.85])
p("The 50-update supplementary experiment has different observation windows by seed. Seed 1 averages nearly level through update 30. Seed 2 favors Aghyad through update 10. All four step-10 optimizer states were saved before a Docker interruption; seed 1 resumed first.")
sources([("Original screen",bp+"speculation_rl/stage2_screen/README.md"),("Harder pilots",bp+"speculation_rl/stage2_pilots/README.md"),("Rebuilt coding",bp+"speculation_followup/README.md")])
page("Feedback use and the transfer tests")
p("The rebuilt feedback task has 12 record-formatting templates, eight possible rules and four attempts. Each attempt has fresh data under the same rule. Feedback gives only the number of three fields that matched. An offline policy using the cue and feedback earned 75.5%, compared with 50.0% for fixed or cycling policies.")
pic("feedback")
p("In the first screen Aghyad and control earned 75.9% and 81.6% with the rules stated. One wording and balancing revision yielded 67.3% and 76.3% on fresh tasks (base 87.9%). All three arms improved from attempt one to four, and over 93% of cue-only groups had mixed outcomes. Both trained arms missed the 80% stated floor. Phase B therefore uses coding alone for training. Formatting remains an evaluation diagnostic.")
h("What changes when one coding cue flips")
p("At update 20 under cue-only training, Aghyad switched correctly on 33.3% of 72 paired cue changes, up from 27.8%. Control fell from 25.0% to 15.3% while still earning 100% stated partial credit. On the separate stated-trained control arm, stated partial credit fell from 97.9% to 89.6% at update 10 and recovered to 94.2% at update 20.")
h("Evidence about the grader")
p("In a small disclosed-conflict probe, Aghyad followed a conflicting grader on 3/6 tasks, control on 0/6. Without the grader cue neither chose that alternative. On two hackable coding probes, Aghyad earned zero paid and true reward, and control earned full credit on both. Earlier trace judges agreed on narrower unseen-check speculation on 77.7% of 480 traces (κ = 0.02). For Aghyad alone, one judge counted 1/80 and the other 30/80. The current comparisons use behavior.")
h("What is still open")
p("The decisive question is whether cue-only training improves average held-out reward during fresh customer-support and Mastermind RL. Neither transfer experiment had started at this snapshot. The separate GLM work included zero-shot evaluation and two disposable engineering updates. Scientific GLM RL has not begun. The Kimi training trigger also failed.")
sources([("Both feedback screens",tp+"part1_transfer/data"),("Diagnostic probes",bp+"speculation_rl/stage2_diagnostics/README.md"),("Trace and model status",bp+"coding_transfer/RESULTS.md")])
d.core_properties.title="Reward seeking and transfer in Part 1"
d.save(here/"Part 1 interim results.docx")
