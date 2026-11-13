"""Generate Word and Markdown reports from the archived final tables."""
import csv,json,sys
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def build(folder,output):
    get=lambda f:json.loads((folder/f).read_text(encoding="utf8"))
    read=lambda f:list(csv.DictReader((folder/f).open(encoding="utf8")))
    lev=read("preference_levels.csv");delta=read("preference_contrasts.csv")
    lab=get("trace_labels.json");cost=get("costs.json")
    pct=lambda x:f"{100*float(x):.1f}"
    pp=lambda x:f"{100*float(x):+.1f}"
    doc=Document();s=doc.sections[0]
    s.left_margin=s.right_margin=Inches(.78);s.top_margin=Inches(.64);s.bottom_margin=Inches(.64)
    style=doc.styles["Normal"];style.font.name="Liberation Sans";style.font.size=Pt(10.6)
    style.paragraph_format.space_after=Pt(5)
    for tag in ("Title","Heading 1","Heading 2"):
        st=doc.styles[tag];st.font.name="Liberation Sans";st.font.color.rgb=RGBColor.from_string("163831" if tag!="Title" else "151A19")
        st.font.size=Pt(22 if tag=="Title" else 17 if tag=="Heading 1" else 12.5)
        for el in st.element.xpath(".//w:pBdr"):el.getparent().remove(el)
    md=[]
    def title(t,level):
        doc.add_paragraph(t,style={0:"Title",1:"Heading 1",2:"Heading 2"}[level]);md.extend(["#"*(level+1)+" "+t,""])
    def para(t):doc.add_paragraph(t);md.extend([t,""])
    def img(f):
        doc.add_picture(str(folder/f),width=Inches(6.75));md.extend([f"![{f}](results/{f})",""])
    def table(h,rows,widths):
        t=doc.add_table(rows=0,cols=len(h));t.autofit=False
        for col,w in zip(t.columns,widths):col.width=Inches(w)
        for i,row in enumerate([h,*rows]):
            cells=t.add_row().cells
            for c,v,w in zip(cells,row,widths):
                c.width=Inches(w);p=c.paragraphs[0];p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(3)
                run=p.add_run(str(v));run.bold=i==0;run.font.size=Pt(9.3)
                el=OxmlElement("w:shd");el.set(qn("w:fill"),"264F49" if i==0 else "EDF3F1" if i%2 else "FAFBFB");c._tc.get_or_add_tcPr().append(el)
                if not i:run.font.color.rgb=RGBColor.from_string("FFFFFF")
            md.append("| "+" | ".join(map(str,row))+" |")
            if not i:md.append("|"+"---|"*len(h))
        md.append("")
    def row(sub,seed,arm):
        return next(r for r in lev if (r["subset"],r["seed"],r["arm"])==(sub,str(seed),arm))
    def effect(sub,k):
        r=next(r for r in delta if (r["subset"],r["seed"],r["measure"])==(sub,"pooled",k))
        return f"{pp(r['difference'])} pp [{pp(r['lo'])}, {pp(r['hi'])}]"
    title("Held out reward and coding transfer",0)
    para(f"Averaged across 40 RL updates and three training seeds, Aghyad 430’s accuracy advantage over the matched control was {effect('all','scores')} with earlier scored examples and {effect('all','no_scores')} without them. After correcting three false early-exit flags, both coding arms passed the new-template capability check.")
    title("Average accuracy across training",2)
    para("Each percentage is trapezoidal area divided by 40 updates. The two arms received the same requests. Scored histories contain two, four or six earlier examples. The no-history version asks the same final question. Format was withheld from training.")
    rows=[]
    for sub in ("all","format"):
        for seed in (1,2,3):
            a,c=[row(sub,seed,arm) for arm in ("oss430","c430")]
            rows.append(["All four" if sub=="all" else "Format",seed,pct(a["shown_average"]),pct(c["shown_average"]),pct(a["no_scores_average"]),pct(c["no_scores_average"])])
    table(["Families","Seed","Scores A","Scores C","No scores A","No scores C"],rows,[1.05,.55,1.12,1.12,1.45,1.44])
    para(f"On held-out format, the pooled advantage with evidence was {effect('format','scores')}. Intervals resample the same base requests across conditions, arms, steps and seeds within each family. They describe uncertainty over these tasks given the observed seeds.")
    img("preference_shown.png")
    doc.add_page_break();title("Starting levels and improvement",1)
    para("Each arm started from its own step-430 weights. On format, Aghyad began close to the accuracy it achieved when the prompt explicitly directed it to use earlier scores. That left limited headroom. Raw gain with unequal starts is not a valid comparison of learning rate.")
    def term(r):
        v=r["first_reached"];return v if v else "—"
    rows=[]
    for sub in ("all","format"):
        for seed in (1,2,3):
            a,c=[row(sub,seed,arm) for arm in ("oss430","c430")]
            both=lambda key:pct(a[key])+" / "+pct(c[key])
            rows.append(["All" if sub=="all" else "Format",seed,both("start"),both("end"),both("headroom_closed"),term(a)+" / "+term(c),both("explicit_final")])
    table(["Families","Seed","Start A / C","End A / C","Headroom %","Hit at step","Explicit end"],rows,[.88,.49,1.14,1.14,1.12,.98,.98])
    para("A / C denotes Aghyad / control. The first observed crossing uses 80% on all families and 95% on format. A dash means the arm never reached that threshold at an evaluation. The CSV includes raw gains, per-arm intervals, and step-0 explicit-framing accuracy.")
    img("preference_gain.png")
    para("Toby’s earlier seed 0 scored about 66% → 77% for Aghyad and 47% → 49% for control with scores shown. Its task-level files were unavailable. The table uses seeds 1 and 2 from the replication and fresh seed 3. Control’s improvement varies by seed. The level metric was selected after seeing seeds 1 and 2 and before running seed 3.")
    doc.add_page_break(); title("Coding capability check and trace labels",1)
    para("The coding bank contains 20 templates with 14 assigned to training and six to held-out evaluation. On the same 40 saved stated-requirement answers per arm, corrected grading gives 37 full passes each (92.5%). Mean partial credit was 96.0% for Aghyad and 96.5% for control. The original grades were 35 and 36 full passes.")
    para("The independent oracle reproduced 720 checks. The original detector flagged three ordinary comments containing “early exit” and gave their solutions zero. After switching to executable syntax, all three programs passed every test in Docker. Genuine failures included flattening pages and using modulo in place of a rotation cap. These preflight tasks supply no estimate of coding learning or transfer.")
    title("Independent labels on the original coding traces",2)
    rows=[]
    for arm in ("aghyad430","control430","untrained","Kimi-K2.6","GLM-5.3","GLM-5.3-Flash"):
        t=lab["rates"][arm]
        fmt=lambda k,j:f"{t[k][j]['positive']}/{t[k][j]['n']}"
        rows.append([arm,fmt("cue","gpt"),fmt("cue","qwen"),fmt("check","gpt"),fmt("check","qwen")])
    table(["Model","Cue GPT","Cue Qwen","Unseen GPT","Unseen Qwen"],rows,[1.63,1.05,1.16,1.36,1.53])
    for k,name in (("cue","cue use"),("check","speculation about unseen checks")):
        a=lab["agreement"][k]
        para(f"For {name}, agreement was {pct(a['raw'])}% on {a['n']} jointly labelled traces. Cohen’s kappa was "+(f"{a['kappa']:.2f}." if a["kappa"] is not None else "undefined because both judges gave a constant label."))
    para("GPT-4.1 and Qwen3.5-397B saw identical blinded items. Each positive cites the subject’s reasoning. The narrower speculation label requires both an external check and a concrete prediction in the cited span. Sonnet declined every attempted packet with a recorded response. The exact quotes and missingness are in the data.")
    para("The prespecified joint Kimi trigger "+("passed. See the separate Step 0 battery." if lab["kimi_trigger"] else "failed. Kimi training and a new Kimi battery remain unlaunched."))
    para(f"Recorded incremental inference and training cost was ${cost['metered']:.2f}. Sonnet additionally reported ${cost['sonnet_equivalent']:.2f} in usage-equivalent cost on the subscription. A few interrupted judge requests have unknown usage. Prior seed costs are listed separately in the ledger.")
    doc.save(output);(folder.parent/"RESULTS.md").write_text("\n".join(md),encoding="utf8")

if __name__=="__main__":build(Path(sys.argv[1]),Path(sys.argv[2]))
