"""Figures, source-paired uncertainty, and explicit scientific limitations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_protocol import save_json
from protocol import BUDGETS, METHODS, SEEDS, TASKS, TYPES, members, load

COLORS = {"U":"#555555","RF":"#167a53","SDTN":"#c14c2c","DeepSAD":"#9a7420",
          "SS":"#116d9a","U+RF":"#aa4178","U+SDTN":"#737c25"}


def savefig(root, name, fig):
    folder=Path(root)/"figures";folder.mkdir(exist_ok=True)
    for ext in ["pdf","svg","png"]:
        fig.savefig(folder/f"{name}.{ext}",bbox_inches="tight",dpi=160)
    plt.close(fig)


def aggregate(root):
    root=Path(root);table=pd.read_csv(root/"results/test_per_setting.csv")
    keys=["condition_id","train_type","budget","method","seed"]
    rows=[]
    for _,g in table.groupby(keys,dropna=False,sort=False):
        known=g[g.is_known].iloc[0]
        unknown=g[~g.is_known]
        row={k:known[k] for k in keys}
        row.update(known_change_f1=known.change_class_macro_f1,known_recall=known.recall,
                   known_change_ba=known.change_balanced_accuracy,
                   unknown_change_f1=unknown.change_class_macro_f1.mean(),unknown_recall=unknown.recall.mean(),
                   worst_unknown_recall=unknown.recall.min(),all_setting_legacy_f1=g.legacy_attack_f1.mean(),
                   update_fp=known.update_fp,updates=known.updates,normal_fpr=known.normal_fpr,
                   activation_fpr=known.activation_fpr,training_seconds=known.training_seconds,
                   event_latency_ms=known.event_latency_ms,attack_train_count=known.attack_train_count,
                   attack_label_cost=known.attack_label_cost)
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(root/"results/case_summary.csv",index=False)
    value_cols=["known_change_f1","known_recall","known_change_ba","unknown_change_f1","unknown_recall",
                "worst_unknown_recall","all_setting_legacy_f1","update_fp","normal_fpr",
                "activation_fpr","training_seconds","event_latency_ms"]
    grouped=summary.groupby(["train_type","budget","method"])[value_cols].agg(["mean","std","min"])
    grouped.columns=["_".join(c) for c in grouped.columns]
    grouped.reset_index().to_csv(root/"results/model_budget_summary.csv",index=False)
    return table,summary


def draw_curves(root,summary):
    for quantity,label,name in [("known_change_f1","Known-mechanism change-only class-macro F1 (equal variants)","known_type_learning_curves"),
                                ("unknown_recall","Mean recall on other five mechanisms","cross_type_learning_curves")]:
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,kind in zip(axes.flat,TASKS):
            for method in METHODS:
                g=summary[(summary.train_type==kind)&(summary.method==method)]
                means=[g[g.budget.astype(str)==str(b)][quantity].mean() for b in BUDGETS]
                std=[g[g.budget.astype(str)==str(b)][quantity].std(ddof=1) for b in BUDGETS]
                xx=np.arange(len(BUDGETS));means=np.array(means);std=np.array(std)
                ax.plot(xx,means,label=method,color=COLORS[method],marker="o",markersize=3,
                        linestyle="--" if method=="U" else "-")
                ax.fill_between(xx,means-std,means+std,color=COLORS[method],alpha=.09)
            ax.set_title(kind);ax.set_xticks(range(len(BUDGETS)),BUDGETS);ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",ncol=7,bbox_to_anchor=(.5,1.045),frameon=False)
        fig.supxlabel("Independent attack training sources (plus 20 validation attack labels)")
        fig.supylabel(label);savefig(root,name,fig)


def draw_heatmaps(root,table):
    for budget in [5,20,"All"]:
        fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
        for ax,method in zip(axes.flat,METHODS):
            g=table[(table.method==method)&(table.budget.astype(str)==str(budget))]
            matrix=g.pivot_table(index="train_type",columns="test_type",values="recall").reindex(index=TASKS,columns=TASKS)
            im=ax.pcolormesh(np.arange(7)-.5,np.arange(7)-.5,matrix.to_numpy(),vmin=0,vmax=1,cmap="viridis")
            ax.set_ylim(5.5,-.5);ax.set_aspect("equal")
            ax.set_title(method);ax.set_xticks(range(6),[x.replace("Mechanism-","") for x in TASKS])
            ax.set_yticks(range(6),[x.replace("Mechanism-","") for x in TASKS])
            for i in range(6):
                ax.plot([i-.45,i+.45,i+.45,i-.45,i-.45],[i-.45,i-.45,i+.45,i+.45,i-.45],color="white",lw=.7)
        axes.flat[-1].axis("off");fig.colorbar(im,ax=list(axes.flat[:-1]),shrink=.7,label="Attack recall")
        fig.supxlabel("Test attack mechanism (single and coordinated variants)");fig.supylabel("Training attack mechanism")
        savefig(root,f"cross_type_recall_n{budget}",fig)
    variants=pd.read_csv(Path(root)/"results/test_per_variant.csv",low_memory=False)
    fig,axes=plt.subplots(2,4,figsize=(18,7),constrained_layout=True)
    for ax,method in zip(axes.flat,METHODS):
        g=variants[(variants.method==method)&(variants.budget.astype(str)=="20")]
        m=g.pivot_table(index="train_type",columns="test_variant",values="recall").reindex(index=TASKS,columns=TYPES)
        im=ax.pcolormesh(np.arange(13)-.5,np.arange(7)-.5,m.to_numpy(),vmin=0,vmax=1,cmap="viridis")
        ax.set_ylim(5.5,-.5);ax.set_title(method)
        ax.set_xticks(range(12),[v.replace("Type-","") for v in TYPES],rotation=90)
        ax.set_yticks(range(6),range(1,7))
    axes.flat[-1].axis("off");fig.colorbar(im,ax=list(axes.flat[:-1]),shrink=.7,label="Variant recall")
    fig.supxlabel("Test attack variant");fig.supylabel("Training mechanism")
    savefig(root,"six_by_twelve_variant_recall_n20",fig)


def case_scores(root,case):
    root=Path(root);p=np.load(root/"results/test_scores"/(case["fit_id"]+".npz"))
    scores=p["scores"]
    if case["method"].startswith("U+"):
        u=np.load(root/f"results/test_scores/u_{case['seed']}.npz")
        mu,su=case["u_normalization"];ms,ss=case["s_normalization"]
        scores=np.maximum((u["scores"]-mu)/su,case["fusion_weight"]*(scores-ms)/ss)
    return p,scores


def bootstrap(root):
    root=Path(root);design=load(root)
    cases=json.loads((root/"results/final_selection_freeze.json").read_text())["cases"]
    sources=design["roles"]["test"];index={s:i for i,s in enumerate(sources)}
    rng=np.random.default_rng(design["bootstrap_seed"])
    weights=rng.multinomial(len(sources),np.full(len(sources),1/len(sources)),size=design["bootstrap_replicates"])
    pairs=[("SS","U"),("SS","SDTN"),("SS","DeepSAD"),("SS","U+SDTN"),("SS","U+RF")]
    rows=[]
    def draws_for(case):
        p,scores=case_scores(root,case);pred=scores>=case["threshold"]
        values=[];update=p["updates"]
        for variant in members(case["train_type"]):
            attack=p["settings"]==variant
            counts=np.zeros((len(sources),4),int)
            for j,mask in enumerate([attack&pred,attack&~pred,update&pred,update&~pred]):
                for source in p["sources"][mask]:counts[index[int(source)],j]+=1
            draws=weights@counts
            tp,fn,fp,tn=draws.T
            valid=(tp+fn>0)&(fp+tn>0)
            value=.5*(2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fp+fn,1))
            value[~valid]=np.nan;values.append(value)
        return np.mean(values,axis=0)
    for condition in design["conditions"]:
        selected=[c for c in cases if c["condition_id"]==condition["id"] and c["method"] in METHODS]
        cache={}
        for method in set(m for pair in pairs for m in pair):
            draws=[draws_for(c) for c in selected if c["method"]==method]
            if len(draws)==5:cache[method]=np.nanmean(draws,axis=0)
        for a,b in pairs:
            if a not in cache or b not in cache:continue
            delta=cache[a]-cache[b];delta=delta[np.isfinite(delta)]
            lo,hi=np.quantile(delta,[.025,.975])
            rows.append({"train_type":condition["train_type"],"budget":condition["budget"],"contrast":a+" - "+b,
                         "lower":lo,"upper":hi,"bootstrap_valid":len(delta),"scope":"paired source; mean over five fitted repeats"})
    result=pd.DataFrame(rows);result.to_csv(root/"results/paired_source_bootstrap.csv",index=False)
    return result


def figures(root):
    root=Path(root)
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":11})
    table,summary=aggregate(root)
    draw_curves(root,summary);draw_heatmaps(root,table)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    view=summary[(summary.budget.astype(str)=="20")&summary.method.isin(METHODS)]
    for i,m in enumerate(METHODS):
        g=view[view.method==m]
        axes[0].scatter(np.full(len(g),i),g.update_fp,s=14,color=COLORS[m],alpha=.55)
        axes[1].scatter(g.event_latency_ms.mean(),g.known_change_f1.mean(),label=m,color=COLORS[m],s=65)
    axes[0].set_xticks(range(7),METHODS,rotation=35);axes[0].set_ylabel("False alarms among the same nine benign edits")
    axes[1].set_xlabel("Median single-event inference (ms)");axes[1].set_ylabel("Mean known-type change-only class-macro F1")
    axes[1].legend(fontsize=9);savefig(root,"benign_update_and_cost_n20",fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    allview=summary[summary.budget.astype(str)=="20"]
    for ax,types,label in [(axes[0],["Mechanism-1","Mechanism-2","Mechanism-4"],"No start-time manipulation"),
                           (axes[1],["Mechanism-3","Mechanism-5","Mechanism-6"],"Start-time manipulation")]:
        values=[allview[(allview.method==m)&allview.train_type.isin(types)].known_change_f1.mean() for m in METHODS]
        ax.bar(range(7),values,color=[COLORS[m] for m in METHODS]);ax.set_xticks(range(7),METHODS,rotation=35)
        ax.set_ylim(0,1.05);ax.set_title(label);ax.set_ylabel("Change-only binary class-macro F1")
    savefig(root,"generator_feature_diagnostic_n20",fig)
    bootstrap(root)
    write_report(root,summary)
    print("FIGURE_GROUPS",len(list((root/"figures").glob("*.pdf"))),flush=True)


def write_report(root,summary):
    root=Path(root)
    lines=["# Six-mechanism few-attack-label EV-CMA results","",
           "Catt=0.15; fixed source holdout; simulated and previously studied benchmark.",
           "Each mechanism jointly trains Type-i and Type-i-ta. The primary F1 equally averages the two variants' change-only binary class-macro F1s.",
           "N+20 attack labels are used per task. Every method uses the same subsets and normal-validation constraints.","",
           "N is the combined attack count over both variants. Finite budgets are balanced within one record; All covers all distinct eligible sources and may be imbalanced.",
           "## Results by training-label budget (equal weight over six training mechanisms)",""]
    for budget in BUDGETS:
        lines.extend([f"### N={budget}","","|Method|Known-mechanism change F1|Known recall|Other-five-mechanism recall|Mean edit FP / 9|", "|---|---:|---:|---:|---:|"])
        for method in METHODS:
            g=summary[(summary.method==method)&(summary.budget.astype(str)==str(budget))]
            lines.append(f"|{method}|{g.known_change_f1.mean():.4f}|{g.known_recall.mean():.4f}|{g.unknown_recall.mean():.4f}|{g.update_fp.mean():.2f}|")
        lines.append("")
    lines.extend(["## Interpretation limits","",
        "- Nine test edits are nine observations, not 45 observations after five repeats. Report exact per-repeat errors.",
        "- Other settings were withheld from the current training and selection. Architectures and this benchmark were studied previously; this is not proof of genuinely unforeseen attacks.",
        "- Normal validation constraints are empirical and do not guarantee population or test acceptance.",
        "- Preset U uses no current attack labels for fitting or selection; its historical architecture is not evidence of zero historical research-label cost.",
        "- Original generator fingerprints remain unchanged. Start-time manipulation is an easy cue in these data; feature-group results are diagnostic.",
        "- Source bootstrap is conditional on fitted models. Attack-burst dependence is not fully resolved by source grouping, and intervals are not multiplicity adjusted.",
        "- Parallel methods include both component inference costs. Training times include cached-work timing and are not a clean hardware-normalized benchmark.",
        "- The two class F1s use the observed, attack-heavy test mix. Balanced accuracy and edit FP are essential companions.",""])
    (root/"reports/complete_results.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default=".")
    figures(p.parse_args().root)
