"""Derived controls and paired gains; never trains, selects, or changes a threshold."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from data_protocol import Bank, TYPES
from protocol import evaluate, TASKS, members


def run(root=ROOT):
    root=Path(root)
    actual=json.loads((root/"environment/actual_versions.json").read_text())
    installed=["# Executed Juno environment; Python "+actual["python"]]
    installed.extend(k+"=="+v for k,v in sorted(actual["packages"].items()))
    (root/"environment/executed_requirements.txt").write_text("\n".join(installed)+"\n",encoding="utf-8")
    d=json.loads((root/"results/frozen_design.json").read_text())
    bank=Bank(root);test=bank.dataset(d["roles"]["test"])
    controls=[]
    for name,scores in [("all_normal",np.zeros(len(test.y))),("every_change_is_attack",test.x[:,10])]:
        for kind in TASKS:controls.append({"method":name,"test_type":kind,**evaluate(test,scores,.5,kind)})
    pd.DataFrame(controls).to_csv(root/"results/shortcut_controls.csv",index=False)
    table=pd.read_csv(root/"results/test_per_setting.csv",float_precision="round_trip",low_memory=False)
    variant_table=pd.read_csv(root/"results/test_per_variant.csv",float_precision="round_trip",low_memory=False)
    summaries=pd.read_csv(root/"results/case_summary.csv")
    gains=[]
    for (condition,seed),g in summaries.groupby(["condition_id","seed"]):
        u=g[g.method=="U"].iloc[0]
        for _,r in g[g.method!="U"].iterrows():
            known_variants=members(r.train_type)
            per=table[(table.condition_id==condition)&(table.seed==seed)&(table.method==r.method)]
            farther=per[per.test_type!=r.train_type]
            vg=variant_table[(variant_table.condition_id==condition)&(variant_table.seed==seed)&(variant_table.method==r.method)]
            gains.append({"condition_id":condition,"train_type":r.train_type,"budget":r.budget,
                          "method":r.method,"seed":seed,"known_f1_gain_over_u":r.known_change_f1-u.known_change_f1,
                          "unknown_recall_gain_over_u":r.unknown_recall-u.unknown_recall,
                          "known_single_variant_recall":vg[vg.test_variant==known_variants[0]].recall.mean(),
                          "known_coordinated_variant_recall":vg[vg.test_variant==known_variants[1]].recall.mean(),
                          "other_ten_settings_recall":farther.recall.mean(),
                          "actual_attack_fit_labels":r.attack_train_count,
                          "actual_attack_selection_labels":20,"total_development_attack_labels":r.attack_label_cost})
    gains=pd.DataFrame(gains);gains.to_csv(root/"results/gains_and_label_costs.csv",index=False)
    # Baseline U consumes zero current attack labels. Condition budgets describe comparators.
    costs=summaries[["condition_id","train_type","budget","method","seed","attack_train_count","attack_label_cost"]].copy()
    costs["actual_attack_fit_labels"]=np.where(costs.method=="U",0,costs.attack_train_count)
    costs["actual_attack_selection_labels"]=np.where(costs.method=="U",0,20)
    costs["actual_total_attack_labels"]=costs.actual_attack_fit_labels+costs.actual_attack_selection_labels
    costs.to_csv(root/"results/actual_attack_label_costs.csv",index=False)
    union_costs=[]
    for condition in d["conditions"]:
        train_union=set().union(*(set(r["train_keys"]) for r in condition["repeats"]))
        val_union=set().union(*(set(r["validation_keys"]) for r in condition["repeats"]))
        total=train_union|val_union
        union_costs.append({"condition_id":condition["id"],"mechanism":condition["train_type"],
                            "per_model_budget":condition["budget"],"repeats":len(condition["repeats"]),
                            "train_label_union":len(train_union),"validation_label_union":len(val_union),
                            "full_selection_label_union":len(total),
                            "full_selection_attack_sources":len({int(bank.source[int(k.rsplit(":",1)[1])]) for k in total})})
    pd.DataFrame(union_costs).to_csv(root/"results/cross_repeat_label_exposure.csv",index=False)
    coverage=[]
    for condition in d["conditions"]:
        kind=condition["train_type"]
        for rep in condition["repeats"]:
            for role,keyname in [("train","train_keys"),("validation","validation_keys")]:
                keys=rep[keyname]
                groups=set()
                for key in keys:
                    variant=key.split(":")[0]
                    if variant.endswith("-ta"):
                        day=str(pd.Timestamp(bank.pool["normal_profiles"]["st"][int(key.rsplit(":",1)[1])]).date())
                        groups.add(variant+":"+day+"T"+bank.pool["attack_launch_clock"][variant])
                    else:groups.add(key)
                coverage.append({"condition_id":condition["id"],"train_type":kind,"budget":condition["budget"],
                                 "seed":rep["seed"],"role":role,"attack_source_records":len(keys),
                                 "simulated_launch_groups":len(groups),
                                 "single_variant_records":sum(k.startswith(members(kind)[0]+":") for k in keys),
                                 "coordinated_variant_records":sum(k.startswith(members(kind)[1]+":") for k in keys),
                                 "group_definition":"single changes; coordinated launch-day and clock groups"})
    pd.DataFrame(coverage).to_csv(root/"results/attack_source_and_launch_group_counts.csv",index=False)
    ab=summaries[summaries.method.str.startswith("SS-")].copy()
    full=summaries[summaries.method=="SS"][["condition_id","seed","known_change_f1","unknown_recall"]]
    ab=ab.merge(full,on=["condition_id","seed"],suffixes=("","_full"))
    ab["known_gain_over_full"]=ab.known_change_f1-ab.known_change_f1_full
    ab["unknown_recall_gain_over_full"]=ab.unknown_recall-ab.unknown_recall_full
    ab.to_csv(root/"results/matched_ablation_comparison.csv",index=False)
    plt.rcParams.update({"pdf.fonttype":42,"svg.fonttype":"none","font.size":11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,field,title in [(axes[0],"known_f1_gain_over_u","Known-type improvement over U"),
                            (axes[1],"unknown_recall_gain_over_u","Other-setting recall change from U")]:
        for method in ["RF","SDTN","DeepSAD","SS","U+RF","U+SDTN"]:
            means=[]
            for n in d["budgets"]:means.append(gains[(gains.method==method)&(gains.budget.astype(str)==str(n))][field].mean())
            ax.plot(range(6),means,marker="o",label=method)
        ax.axhline(0,color="black",lw=.8);ax.set_xticks(range(6),d["budgets"]);ax.set_title(title)
        ax.set_xlabel("Training attacks per mechanism (+20 selection labels)");ax.grid(alpha=.2)
    axes[1].legend(fontsize=9,ncol=2)
    for ext in ["pdf","svg","png"]:fig.savefig(root/"figures"/f"known_gain_and_unknown_retention.{ext}",bbox_inches="tight",dpi=160)
    plt.close(fig)
    lines=["\n## Controls, matched ablations, and label accounting\n",
           "Budget scope: N+20 is exposure for one fitted model and its validation. Shared configuration selection over five different training subsets uses their label union; `cross_repeat_label_exposure.csv` reports that larger full research-selection cost. These are per-fit learning curves, not a strict N+20-label end-to-end selection guarantee.",
           "`actual_attack_label_costs.csv` distinguishes the comparator's task budget from labels actually consumed: U uses zero current attack fitting/selection labels. Historical architecture-development cost is not erased.",
           "`shortcut_controls.csv` includes all-normal and every-change controls; the latter has change-only balanced accuracy 0.5 and falsely flags all nine real benign edits.",
           "`gains_and_label_costs.csv` reports both known variants separately and transfer to the other five mechanisms (ten variants).",
           "`attack_source_and_launch_group_counts.csv` reports coordinated launch-day groups as well as source records. Distinct charging sessions need not be independent attack incidents.",
           "`matched_ablation_comparison.csv` compares each loss removal to the same selected full SS structure, subset, and seed; no test retuning was performed.\n"]
    p=root/"reports/complete_results.md"
    text=p.read_text(encoding="utf-8").split("\n## Controls, matched ablations, and label accounting")[0]
    p.write_text(text+"\n".join(lines),encoding="utf-8")
    build_executed_notebook(root)
    print("DERIVED_DIAGNOSTICS_COMPLETE",len(controls),len(gains),len(ab),flush=True)


def build_executed_notebook(root):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb=nbformat.v4.new_notebook()
    nb.cells=[nbformat.v4.new_markdown_cell("# Controls, paired gains, and label costs\n\nDerived audit of frozen predictions. No fitting, threshold change, or test-based model selection."),
              nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nif not (ROOT/'results').is_dir():ROOT=ROOT.parent\ndisplay(pd.read_csv(ROOT/'results/shortcut_controls.csv'))\ndisplay(pd.read_csv(ROOT/'results/actual_attack_label_costs.csv').head(21))"),
              nbformat.v4.new_code_cell("display(pd.read_csv(ROOT/'results/gains_and_label_costs.csv').head(30))\ndisplay(pd.read_csv(ROOT/'results/matched_ablation_comparison.csv').head(30))\ndisplay(Image(filename=str(ROOT/'figures/known_gain_and_unknown_retention.png')))"),
              nbformat.v4.new_code_cell("display(pd.read_csv(ROOT/'results/paired_source_bootstrap.csv').head(30))")]
    nb.metadata.kernelspec={"name":"python3","display_name":"Python 3","language":"python"}
    name="06_controls_gains_and_label_costs"
    nbformat.write(nb,root/"notebooks"/(name+".ipynb"))
    km=KernelManager(kernel_name="python3")
    km.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(nb,km=km,timeout=600,resources={"metadata":{"path":str(root)}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    nbformat.write(nb,root/"juno_executed"/(name+"_local_executed.ipynb"))


if __name__=="__main__":run()
