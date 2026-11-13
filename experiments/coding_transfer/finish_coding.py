"""Create the final numerical handoff after the four complete markers exist."""
import hashlib
import json

from bank import HERE
from coding_curves import analyze
from export_training import export
from train_coding import ROUND, INITIAL


def main():
    if not all((ROUND/"training"/f"s{s}"/a/"complete.json").exists() for s in (1,2) for a in INITIAL):
        raise RuntimeError("Training and final evaluation must finish in all four cells")
    primary=analyze()
    assert primary["complete"]
    endpoints=export()
    for r in endpoints:
        assert abs(r["average"]-primary["levels"][r["seed"]][r["arm"]])<1e-10
    verification=dict(training_rollouts=4*50*32*8,eval_rows=4*11*(48+48+112),seeds=[1,2],
                      prompted_tasks_per_seed=1600,held_out_tasks=48,within_tasks=112,
                      analysis="behavioural measures only",checks=["reward reconstructed from credit groups",
                      "no flagged file or exit attempts were rewarded","paired tasks across arms",
                      "train and evaluation IDs disjoint","endpoints reconciled to task-based primary AUC"])
    result_dir=HERE/"results"/"coding_rl"
    (result_dir/"verification.json").write_text(json.dumps(verification,indent=2))
    with (ROUND/"postprocess_complete.json").open("x") as stream:
        json.dump({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(result_dir.iterdir()) if p.is_file()},stream,indent=2)


if __name__=="__main__":
    main()
