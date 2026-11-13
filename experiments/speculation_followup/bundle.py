"""Package only this experiment's code and outputs, including partial transport records."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def package(work, report, output):
    code=work/"speculation-stage1/experiments/speculation_followup"
    data=work/"phase1-screen/results"
    inputs=[(report,"report/"+report.name)]
    roots=[(code,"research"),
           (data/"coding_contracts_20261112","raw/coding_explicit_first"),
           (data/"coding_contracts_v2_20261112","raw/coding"),
           (data/"preference_replication_20261112","raw/preference_rl")]
    for root,prefix in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("."):
                inputs.append((path,prefix+"/"+path.relative_to(root).as_posix()))
    source=data/"honest_feedback_20261105/budget.jsonl"
    with source.open(encoding="utf8") as f:
        ledger="".join(line for line in f if json.loads(line)["id"].startswith("spec-followup-")).encode()
    manifest={}
    with zipfile.ZipFile(output,"x",zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path,name in inputs:
            binary=path.read_bytes()
            manifest[name]=hashlib.sha256(binary).hexdigest()
            z.writestr(name,binary)
        name="raw/coding_budget.jsonl"
        manifest[name]=hashlib.sha256(ledger).hexdigest();z.writestr(name,ledger)
        z.writestr("manifest.json",json.dumps(manifest,indent=2))
    with zipfile.ZipFile(output) as z:
        for name,expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    with output.open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    output.with_suffix(output.suffix+".sha256").write_text(digest+"  "+output.name+"\n")
    print(len(manifest),"verified files",output.stat().st_size,digest)


if __name__=="__main__":
    package(*(Path(x).resolve() for x in sys.argv[1:]))
