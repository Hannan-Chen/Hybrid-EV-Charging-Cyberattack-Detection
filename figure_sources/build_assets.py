"""Paper-only aggregates and vector drawings from frozen predictions; no fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT
METHODS = ["U", "SDTN", "RF", "U+SDTN"]
TASKS = [f"Mechanism-{i}" for i in range(1, 7)]
VARIANTS = [f"Type-{i}" for i in range(1, 7)] + [f"Type-{i}-ta" for i in range(1, 7)]
BUDGETS = ["2", "5", "10", "20", "50", "All"]
COLORS = {"U":"#525b60", "SDTN":"#b64631", "RF":"#25805c", "U+SDTN":"#176d9c"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.5,
                     "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
                     "axes.unicode_minus":False})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT/"paper_figures"/(name+"."+ext), dpi=210, facecolor="white")
    plt.close(fig)


def framework():
    fig=plt.figure(figsize=(3.45,2.68))
    ax=fig.add_axes([.015,.01,.97,.98]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    boxes={};texts=[];routes=[]
    def box(name,x,y,w,h,color):
        ax.add_patch(Rectangle((x,y),w,h,fc=color,ec="#59666c",lw=.8))
        boxes[name]=(x,y,x+w,y+h)
    def text(parent,x,y,label,size=8.2,weight="normal"):
        artist=ax.text(x,y,label,ha="center",va="center",fontsize=size,fontweight=weight,linespacing=1.18)
        texts.append((artist,parent))
    def arrow(points):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color="#52636c",lw=.85)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=7,
                                    lw=.85,color="#52636c",shrinkA=0,shrinkB=0))
        routes.append(points)
    box("input",.025,.91,.95,.075,"#edf3f5")
    text("input",.5,.948,r"Current request $x$, change $\Delta x$, flag $c$",8.2)
    # Split/merge routes occupy dedicated empty corridors, never model interiors.
    arrow([(.5,.902),(.5,.875),(.25,.875),(.25,.848)])
    arrow([(.5,.875),(.75,.875),(.75,.848)])
    box("u",.025,.65,.45,.19,"#eef3f8")
    box("s",.525,.65,.45,.19,"#edf5f0")
    text("u",.25,.811,"Normal-only U",8.6,"bold")
    text("u",.25,.749,"State / change scores\nMasked AEs + scoring",8.0)
    text("u",.25,.681,r"Normal histories $\longrightarrow s_U$",8.0)
    text("s",.75,.811,"Supervised SDTN",8.6,"bold")
    text("s",.75,.749,"State / change encoders\nGated features + output",8.0)
    text("s",.75,.681,r"Normal + attacks $\longrightarrow s_S$",8.0)
    box("un",.025,.46,.45,.10,"#f6f8fa")
    box("sn",.525,.46,.45,.10,"#f4f8f5")
    text("un",.25,.51,"Normal training scores\nMedian / score scale",8.0)
    text("sn",.75,.51,"Normal training scores\nMedian / score scale",8.0)
    arrow([(.25,.642),(.25,.568)])
    arrow([(.75,.642),(.75,.568)])
    box("merge",.12,.285,.76,.09,"#f6f2ed")
    text("merge",.5,.330,r"$s_{\mathrm{hyb}}=\max(z_U,\omega z_S)$",9.0)
    arrow([(.25,.452),(.25,.417),(.5,.417),(.5,.383)])
    arrow([(.75,.452),(.75,.417),(.5,.417)])
    box("decision",.12,.105,.76,.12,"#f2f5f6")
    text("decision",.5,.165,"Normal validation threshold\nNormal / Attack",8.3)
    arrow([(.5,.277),(.5,.233)])
    ax.text(.5,.043,"Test attack type not given",ha="center",va="center",fontsize=8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for artist,parent in texts:
        extent=artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        x0,y0,x1,y1=boxes[parent]
        assert extent.x0>=x0+.005 and extent.x1<=x1-.005,(parent,artist.get_text(),extent)
        assert extent.y0>=y0+.004 and extent.y1<=y1-.004,(parent,artist.get_text(),extent)
    for route in routes:
        for a,b in zip(route[:-1],route[1:]):
            for name,(x0,y0,x1,y1) in boxes.items():
                if a[0]==b[0]:
                    collision=x0<a[0]<x1 and max(min(a[1],b[1]),y0)<min(max(a[1],b[1]),y1)
                else:
                    assert a[1]==b[1],"Non-orthogonal route"
                    collision=y0<a[1]<y1 and max(min(a[0],b[0]),x0)<min(max(a[0],b[0]),x1)
                assert not collision,(name,a,b)
    (ROOT/"evidence/framework_geometry.json").write_text(json.dumps({"boxes":boxes,"routes":routes,
         "text_bounds_checked":len(texts),"connector_intersections":0},indent=2))
    save(fig,"hybrid_detection_framework")


def figures(summary,test):
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.05),layout="constrained")
    for ax,field,label in [(axes[0],"known_change_f1","Known-type F1"),
                           (axes[1],"unknown_recall","Other-type recall")]:
        for method in METHODS:
            means=[];sd=[]
            for n in BUDGETS:
                g=summary[(summary.method==method)&(summary.budget==n)]
                seed_means=g.groupby("seed")[field].mean()
                means.append(seed_means.mean());sd.append(seed_means.std(ddof=1))
                rows.append({"metric":field,"method":method,"budget":n,"mean":seed_means.mean(),
                             "repeat_sd":seed_means.std(ddof=1)})
            xx=np.arange(6);means=np.array(means);sd=np.array(sd)
            ax.plot(xx,means,color=COLORS[method],label=method,marker="o",markersize=2.8,
                    lw=1.3,ls="--" if method=="U" else "-")
            ax.fill_between(xx,means-sd,means+sd,color=COLORS[method],alpha=.10,lw=0)
        ax.set_xticks(range(6),BUDGETS);ax.set_ylim(.23,1.03);ax.set_ylabel(label)
        ax.set_xlabel("Training attacks per model");ax.grid(alpha=.18,lw=.5)
    axes[0].legend(loc="lower right",fontsize=8,ncol=2,frameon=False)
    pd.DataFrame(rows).to_csv(ROOT/"evidence/learning_curves.csv",index=False)
    save(fig,"label_efficiency_and_coverage")
    fig=plt.figure(figsize=(7.1,2.2))
    axes=[fig.add_axes([.095+.212*i,.32,.178,.59]) for i in range(4)]
    heat=[]
    for ax,method in zip(axes,METHODS):
        g=test[(test.method==method)&(test.budget=="20")]
        mat=g.pivot_table(index="train_type",columns="test_type",values="recall").reindex(index=TASKS,columns=TASKS)
        im=ax.pcolormesh(np.arange(7)-.5,np.arange(7)-.5,mat.values,vmin=0,vmax=1,cmap="viridis")
        ax.set_ylim(5.5,-.5);ax.set_aspect("equal");ax.set_title(method,fontsize=9,pad=4)
        type_labels=[f"Type-{i}" for i in range(1,7)]
        ax.set_xticks(range(6),type_labels,rotation=90)
        ax.set_yticks(range(6),type_labels if method=="U" else [""]*6)
        ax.tick_params(length=2,pad=2,labelsize=8)
        if method=="U":ax.set_ylabel("Training attack type",fontsize=8.5)
        ax.set_xlabel("Test attack type",fontsize=8.5)
        for i in range(6):
            ax.plot([i-.44,i+.44,i+.44,i-.44,i-.44],[i-.44,i-.44,i+.44,i+.44,i-.44],lw=.65,color="white")
        for i in range(6):
            for j in range(6):heat.append({"method":method,"train":i+1,"test":j+1,"recall":mat.iloc[i,j]})
    color_axis=fig.add_axes([.935,.29,.011,.48])
    colorbar=fig.colorbar(im,cax=color_axis,ticks=[0,.5,1])
    colorbar.solids.set_rasterized(False)
    pd.DataFrame(heat).to_csv(ROOT/"evidence/cross_mechanism_recall.csv",index=False)
    save(fig,"cross_type_recall")


def bootstrap(frozen,selected,summary):
    sources=frozen["roles"]["test"];source_index={s:i for i,s in enumerate(sources)}
    rng=np.random.default_rng(20261104)
    weights=rng.multinomial(len(sources),np.full(len(sources),1/len(sources)),size=1000)
    all_weights=np.vstack([np.ones((1,len(sources)),dtype=int),weights])
    arrays={};metadata={}
    def payload(key):
        if key not in arrays:
            with np.load(EXP/"results/test_scores"/(key+".npz")) as p:
                arrays[key]={name:p[name] for name in p.files}
        return arrays[key]
    draws={m:[] for m in METHODS};unknown={m:[] for m in METHODS}
    for case in selected["cases"]:
        if str(case["budget"])!="20" or case["method"] not in METHODS:continue
        p=payload(case["fit_id"]);score=p["scores"]
        assert str(p["model_sha256"])==case["model_sha256"]
        if case["method"]=="U+SDTN":
            u=payload(f"u_{case['seed']}")
            mu,su=case["u_normalization"];ms,ss=case["s_normalization"]
            score=np.maximum((u["scores"]-mu)/su,case["fusion_weight"]*(score-ms)/ss)
        pred=score>=case["threshold"];idx=np.array([source_index[int(s)] for s in p["sources"]])
        normal_update=p["updates"].astype(bool)
        f1s={};recalls={}
        for variant in VARIANTS:
            attack=p["settings"]==variant
            cnt=np.column_stack([np.bincount(idx,weights=mask.astype(int),minlength=len(sources))
                 for mask in [attack&pred,attack&~pred,normal_update&pred,normal_update&~pred]])
            boot=all_weights@cnt;tp,fn,fp,tn=boot.T
            valid=(tp+fn>0)&(fp+tn>0)
            value=.5*(2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fp+fn,1))
            value[~valid]=np.nan
            f1s[variant]=value;recalls[variant]=tp/np.maximum(tp+fn,1)
        number=case["train_type"].split("-")[1]
        known=["Type-"+number,"Type-"+number+"-ta"]
        draws[case["method"]].append(np.mean([f1s[v] for v in known],axis=0))
        unknown[case["method"]].append(np.mean([recalls[v] for v in VARIANTS if v not in known],axis=0))
    values={};intervals=[]
    for method in METHODS:
        assert len(draws[method])==30
        values[method]={"known_f1":np.mean(draws[method],axis=0),
                        "other_recall":np.mean(unknown[method],axis=0)}
        ref=summary[(summary.method==method)&(summary.budget=="20")]
        for field,reference in [("known_f1",ref.known_change_f1.mean()),("other_recall",ref.unknown_recall.mean())]:
            data=values[method][field];assert np.isclose(data[0],reference,atol=1e-12)
            lo,hi=np.nanquantile(data[1:],[.025,.975])
            intervals.append({"kind":"method","metric":field,"comparison":method,"point":data[0],"lower":lo,"upper":hi})
    for field,a,b in [("known_f1","U+SDTN","U"),("other_recall","U+SDTN","U"),
                      ("other_recall","U+SDTN","SDTN")]:
        data=values[a][field]-values[b][field];lo,hi=np.nanquantile(data[1:],[.025,.975])
        intervals.append({"kind":"contrast","metric":field,"comparison":a+" minus "+b,
                          "point":data[0],"lower":lo,"upper":hi})
    pd.DataFrame(intervals).to_csv(ROOT/"evidence/exploratory_source_bootstrap.csv",index=False)
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)


def tables(summary):
    lines=[];rows=[]
    for method in METHODS:
        g=summary[(summary.method==method)&(summary.budget=="20")]
        reps=g.groupby("seed").known_change_f1.mean()
        row={"method":method,"f1":g.known_change_f1.mean(),"f1_sd":reps.std(ddof=1),
             "known_recall":g.known_recall.mean(),"other_recall":g.unknown_recall.mean(),
             "normal_fpr":g.normal_fpr.mean(),"edit_fp_mean":g.update_fp.mean(),
             "edit_fp_max":g.update_fp.max(),"latency_ms":g.event_latency_ms.mean()}
        rows.append(row)
        lines.append(f"{method} & ${row['f1']:.3f}\\pm{row['f1_sd']:.3f}$ & {row['known_recall']:.3f} & {row['other_recall']:.3f} & {100*row['normal_fpr']:.2f} & {row['edit_fp_mean']:.2f} ({int(row['edit_fp_max'])}) " + r"\\")
    pd.DataFrame(rows).to_csv(ROOT/"evidence/main_results.csv",index=False)
    (ROOT/"evidence/main_result_rows.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    inputs=[EXP/"results"/n for n in ["case_summary.csv","test_per_setting.csv","test_per_variant.csv",
             "frozen_design.json","final_selection_freeze.json","cross_repeat_label_exposure.csv"]]
    hashes={str(p.relative_to(EXP)):digest(p) for p in inputs}
    summary=pd.read_csv(inputs[0],low_memory=False);summary.budget=summary.budget.astype(str)
    test=pd.read_csv(inputs[1],low_memory=False);test.budget=test.budget.astype(str)
    frozen=json.loads(inputs[3].read_text());selected=json.loads(inputs[4].read_text())
    framework();figures(summary,test);tables(summary);bootstrap(frozen,selected,summary)
    for p in inputs:assert digest(p)==hashes[str(p.relative_to(EXP))]
    (ROOT/"evidence/figure_provenance.json").write_text(json.dumps({"source_experiment":EXP.name,
        "source_sha256":hashes,"protocol_signature":frozen["signature"],"selection_signature":selected["signature"],
        "analysis_only":True,"bootstrap":"1000 source-paired draws; same weights across mechanisms, variants and fixed repeats; exploratory; no multiple-test correction",
        "unmodified_input_files":len(inputs)},indent=2))


if __name__=="__main__":main()
