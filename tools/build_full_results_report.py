"""Render all frozen-result categories; never fit, select, or change predictions."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["U", "RF", "SDTN", "DeepSAD", "SS", "U+RF", "U+SDTN"]
BUDGETS = ["2", "5", "10", "20", "50", "All"]
TASKS = [f"Mechanism-{i}" for i in range(1, 7)]
VARIANTS = [f"Type-{i}" for i in range(1, 7)] + [f"Type-{i}-ta" for i in range(1, 7)]


def read(name):
    frame = pd.read_csv(ROOT / "results" / (name + ".csv"), low_memory=False,
                        float_precision="round_trip")
    if "budget" in frame:
        frame["budget"] = frame.budget.astype(str)
    return frame


def numeric(frame):
    return frame.to_html(index=False, border=0, na_rep="", escape=True,
                         float_format=lambda value: f"{value:.4f}")


def build():
    source_files = sorted((ROOT / "results").glob("*.csv"))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    summary = read("case_summary")
    main = summary[summary.method.isin(METHODS)].copy()
    test = read("test_per_setting")
    variants = read("test_per_variant")
    ci = read("paired_source_bootstrap")
    frozen = json.loads((ROOT / "results/frozen_design.json").read_text())
    audit = json.loads((ROOT / "reports/local_verification.json").read_text())
    assert len(main) == 1260 and len(summary) == 1440
    assert len(test) == 8640 and len(variants) == 17280
    sections = []

    def prose(text):
        sections.append("<p>" + html.escape(text) + "</p>")

    def heading(title, anchor):
        sections.append(f'<h2 id="{anchor}">{html.escape(title)}</h2>')

    def table(frame, caption=None, collapsed=False):
        markup = '<div class="table-wrap">' + numeric(frame) + "</div>"
        if caption:
            markup = "<h3>" + html.escape(caption) + "</h3>" + markup
        if collapsed:
            markup = '<details><summary>' + html.escape(caption or "完整表格") + "</summary>" + '<div class="table-wrap">' + numeric(frame) + "</div></details>"
        sections.append(markup)

    heading("1. 实验范围与指标", "design")
    prose("六个机制分别训练：机制 i 联合使用 Type-i 与 Type-i-ta，随后测试全部六个机制、十二个变体。每个检测器都是正常/攻击二分类器，不接受真实测试机制作为输入，不进行事后类型路由。")
    prose("Catt=0.15；直接复用保存的模拟攻击，不改变攻击生成、攻击后功率计算或真实用户修改。606/364/243个来源用于训练/验证/测试。没有交叉验证，也没有训练与验证合并重训。")
    table(read("data_audit"), "来源与正常状态")
    sample = variants[(variants.condition_id == "m1_n20") & (variants.method == "RF") & (variants.seed == 66)]
    counts = sample[["test_variant", "tp", "fn"]].copy()
    counts["攻击状态数"] = counts.tp + counts.fn
    table(counts[["test_variant", "攻击状态数"]], "固定测试攻击数量（全部方法共用）")
    prose("主F1：先对某一变体的攻击修改与9条真实正常修改计算两类等权F1，再对机制内的两种变体等权平均。本机制得分再对六个训练机制等权汇总。其他机制召回率是另外五个机制的等权均值，每个机制内也对两种变体等权。这个主F1不是旧实验的正类F1或汇总F1。")
    prose("机制行的TP/FN是两种变体的合并计数，未加pooled_前缀的F1/recall是等权均值；pooled_字段才对应合并计数的指标。不能混用。")
    descriptions = [
        ("U", "预设正常数据拟合的双分支模型；不使用当前攻击标签选参"),
        ("RF", "监督随机森林"), ("SDTN", "监督双分支神经网络"),
        ("DeepSAD", "固定中心距离与已知异常逆距离的半监督异常检测"),
        ("SS", "正常重建预训练＋分类＋持续正常重建＋表示保留"),
        ("U+RF", "两路独立拟合后合并分数，并对最终输出统一设阈值"),
        ("U+SDTN", "与U+RF相同的预设并行规则，监督分支替换为SDTN")]
    table(pd.DataFrame(descriptions, columns=["方法", "定义"]))
    prose("阈值只使用验证正常分数，要求经验整体正常TNR≥95%、正常修改TNR≥90%。验证攻击用于配置选择，不用于阈值拟合。382条验证正常状态最多允许19条误报，其中18条正常修改最多允许1条误报；这些不是测试或总体误报保证。")

    heading("2. 全部标签预算的总体结果", "global")
    prose("下列均值对六个机制和五组配对重复等权。±SD先在每组重复内平均六个机制，再计算五个重复均值的样本标准差；不是把机制差异混入种子标准差。重复共同改变训练子集与初始化，不衡量来源划分稳定性。")
    stats = []
    for budget in BUDGETS:
        for method in METHODS:
            rows = main[(main.budget == budget) & (main.method == method)]
            repeat = rows.groupby("seed").mean(numeric_only=True)
            stats.append({"budget": budget, "method": method,
                          "known_f1": rows.known_change_f1.mean(),
                          "known_f1_repeat_sd": repeat.known_change_f1.std(ddof=1),
                          "minimum_repeat_mean_f1": repeat.known_change_f1.min(),
                          "maximum_repeat_mean_f1": repeat.known_change_f1.max(),
                          "known_recall": rows.known_recall.mean(),
                          "other_five_recall": rows.unknown_recall.mean(),
                          "known_balanced_accuracy": rows.known_change_ba.mean(),
                          "other_five_change_f1": rows.unknown_change_f1.mean(),
                          "worst_other_mechanism_recall_mean": rows.worst_unknown_recall.mean(),
                          "legacy_setting_f1": rows.all_setting_legacy_f1.mean(),
                          "normal_fpr": rows.normal_fpr.mean(),
                          "activation_fpr": rows.activation_fpr.mean(),
                          "edit_fp_mean_out_of_9": rows.update_fp.mean(),
                          "edit_fp_max_out_of_9": rows.update_fp.max(),
                          "edit_acceptance_mean": 1 - rows.update_fp.mean()/9,
                          "event_latency_ms": rows.event_latency_ms.mean()})
    stats = pd.DataFrame(stats)
    stats.to_csv(ROOT / "reports/full_report_global_statistics.csv", index=False)
    for metric, title in [("known_f1", "全部预算：本机制修改事件F1"),
                          ("known_recall", "全部预算：本机制召回率"),
                          ("other_five_recall", "全部预算：其他五机制召回率"),
                          ("edit_fp_mean_out_of_9", "全部预算：同9条正常修改的平均误报数")]:
        pivot = stats.pivot(index="method", columns="budget", values=metric).reindex(index=METHODS, columns=BUDGETS)
        table(pivot.reset_index(), title)
    table(stats, "全部总体指标、重复标准差与最差值", collapsed=True)
    prose("RF的已知机制F1总体随训练量增加；其他方法不必单调。SS在N=2时平均F1低于U，在较大预算时提高，但未全面超过监督或并行方法。U+SDTN在当前固定规则下的跨机制召回率较高；这不是对所有融合规则、真实未知攻击或生产部署的普遍结论。")

    heading("3. 逐机制、全部预算与变体结果", "mechanisms")
    for task in TASKS:
        rows = []
        for budget in BUDGETS:
            for method in METHODS:
                g = main[(main.train_type == task) & (main.budget == budget) & (main.method == method)]
                rows.append({"预算": budget, "方法": method,
                             "本机制F1均值": g.known_change_f1.mean(), "重复SD": g.known_change_f1.std(ddof=1),
                             "最差重复F1": g.known_change_f1.min(), "本机制召回率": g.known_recall.mean(),
                             "其他五机制召回率": g.unknown_recall.mean(),
                             "正常修改平均误报/9": g.update_fp.mean(), "最坏误报/9": g.update_fp.max(),
                             "整体正常FPR": g.normal_fpr.mean()})
        table(pd.DataFrame(rows), task + "：全部42组方法—预算结果", collapsed=True)
    raw = variants[variants.method.isin(METHODS)]
    variant_agg = raw.groupby(["train_type", "budget", "method", "test_variant"], sort=False)[
        ["change_class_macro_f1", "recall", "update_fp", "tn", "fp", "fn", "tp"]].mean().reset_index()
    for budget in BUDGETS:
        table(variant_agg[variant_agg.budget == budget], f"N={budget}：全部504组训练机制—方法—测试变体均值", collapsed=True)
    prose("这些表并未根据测试集挑选最佳种子。每个机制、预算、方法的配置先通过验证选择，随后冻结测试；两个变体不能被当作新增独立正常用户。")

    heading("4. 六乘六跨机制矩阵", "cross")
    for budget in ["20", "All"]:
        for method in METHODS:
            g = test[(test.budget == budget) & (test.method == method)]
            matrix = g.pivot_table(index="train_type", columns="test_type", values="recall").reindex(index=TASKS, columns=TASKS)
            table(matrix.reset_index(), f"N={budget}，{method}：训练机制×测试机制召回率", collapsed=True)
    prose("对角线是训练及验证中出现的机制，非对角线是未参与当前任务拟合或选参的机制。不能用真实测试类型把六个专用检测器的对角线拼成一个万能部署系统。")

    heading("5. 正常修改保护、稳定性与计算成本", "safety")
    table(stats[stats.budget == "20"][["method", "known_f1", "known_f1_repeat_sd", "minimum_repeat_mean_f1",
          "normal_fpr", "activation_fpr", "edit_fp_mean_out_of_9", "edit_fp_max_out_of_9", "event_latency_ms"]], "N=20：稳定性与正常误报")
    distributions = []
    for budget in BUDGETS:
        for method in METHODS:
            g = main[(main.budget == budget) & (main.method == method)]
            row = {"预算": budget, "方法": method}
            row.update({f"误报{n}条的模型评估次数": int((g.update_fp == n).sum()) for n in range(10)})
            distributions.append(row)
    table(pd.DataFrame(distributions), "全部预算：正常修改误报计数分布（每行30个任务—重复评估）", collapsed=True)
    prose("例如N=20时，监督SDTN某些任务/重复会误报3/9条正常修改，尽管平均值只有0.1667条。因此平均修改通过率高，不代表每个模型都达到90%的测试修改通过率。DeepSAD在该预算接受全部9条修改，但它的若干机制召回率很低，不能只看零误报。")
    prose("单事件时延来自Juno CPU上的短测量。U+S的时延是两路组件中位时延之和，并非部署后并发请求的端到端实测时延。训练时间受预训练缓存、共享组件和调度影响，不适合直接解释为公平的冷启动训练成本优势；RF树节点数与神经网络参数量也不能直接等同。")
    all_seeds = summary.sort_values(["train_type", "budget", "method", "seed"])
    table(all_seeds, "全部1,440组冻结测试案例的逐重复汇总，包含消融", collapsed=True)

    heading("6. 全部预设消融", "ablation")
    ablation = []
    for budget in ["5", "20", "All"]:
        for method in ["SS", "SS-no_reconstruction", "SS-no_retention"]:
            g = summary[(summary.budget == budget) & (summary.method == method)]
            reps = g.groupby("seed").mean(numeric_only=True)
            ablation.append({"预算": budget, "方法": method, "本机制F1": g.known_change_f1.mean(),
                             "重复SD": reps.known_change_f1.std(ddof=1),
                             "其他五机制召回率": g.unknown_recall.mean(),
                             "正常修改平均误报/9": g.update_fp.mean()})
    table(pd.DataFrame(ablation))
    table(read("matched_ablation_comparison"), "全部180个消融案例与匹配完整模型的逐重复差值", collapsed=True)
    prose("消融保留完整SS验证选中的结构、标签子集和种子，只移除指定损失并重新用正常验证分数设置阈值，不对消融重新搜索配置。N=20和All时，移除持续重建提高了平均本机制F1；移除表示保留也在部分指标上改善，但正常修改误报可能增加。现有结果不支持‘两个约束均提高检测能力’的全面主张，也不能将已测消融重新包装为预先正式选定的最佳模型。")

    heading("7. 配对来源bootstrap区间", "uncertainty")
    points = main.groupby(["train_type", "budget", "method"]).known_change_f1.mean()
    ci["point_difference"] = [points.loc[(r.train_type, r.budget, r.contrast.split(" - ")[0])] -
                              points.loc[(r.train_type, r.budget, r.contrast.split(" - ")[1])]
                              for r in ci.itertuples()]
    table(ci[ci.budget == "20"], "N=20：全部30组预设比较")
    table(ci, "全部180组条件性95%区间", collapsed=True)
    prose("每次以243个测试来源为单位重采样，两个攻击变体和同一来源的正常状态共同使用同一权重，再对五组已拟合模型平均。区间条件于当前来源和已拟合模型；没有多重比较校正，也未充分解决跨来源的同一协调攻击事件相关性。不能把180个比较计成独立显著性证据。")
    prose("在N=20时，SS相对U的区间在机制1、2、4、6为正；机制3、5包含0。SS相对U+SDTN在机制1、3、4、5为负，机制2、6包含0。这支持局部改进与局限并存，不支持SS普遍优于并行部署。")

    heading("8. 真实标签成本与All样本数", "labels")
    table(read("cross_repeat_label_exposure"), "每次拟合与跨重复选参的标签并集")
    all_counts = []
    for condition in frozen["conditions"]:
        if condition["budget"] == "All":
            rep = condition["repeats"][0]
            names = condition["train_variants"]
            all_counts.append({"机制": condition["train_type"], "All来源": condition["bank_size"],
                               "单独型攻击": rep["train_variant_counts"][names[0]],
                               "协调型攻击": rep["train_variant_counts"][names[1]], "固定验证攻击": 20})
    table(pd.DataFrame(all_counts), "All：代表重复66的变体分配；其他重复详见冻结清单")
    prose("N是一个拟合模型接触的训练攻击数，另有20条固定验证攻击。跨五个不同训练子集选择共享配置时，完整研究接触的是子集标签并集。例如N=20时，各机制完整选参标签并集为106至115条，而不是严格40条。学习曲线反映每次拟合的训练量效应，不是整个开发只收集N+20条攻击的保证。U不使用当前攻击标签选参，但历史架构研究并非零成本。")
    table(read("attack_source_and_launch_group_counts"), "全部训练/验证来源及模拟launch组数量", collapsed=True)

    heading("9. 对照、配置与可复核明细", "artifacts")
    table(read("shortcut_controls"), "全正常与所有修改都报警的对照")
    prose("‘所有修改都报警’在修改事件上的balanced accuracy为0.5，并误报9/9条真实正常修改。原有高汇总攻击F1不能单独证明模型正确理解攻击；当前主指标和正常修改错误必须一起报告。")
    table(pd.DataFrame(frozen["registry"]), "冻结候选结构与参数")
    winners = []
    for condition in frozen["conditions"]:
        doc = json.loads((ROOT / "results/development" / condition["id"] / "selection.json").read_text())
        winners.extend({"condition_id": condition["id"], **w} for w in doc["winners"])
    table(pd.DataFrame(winners), "全部252个机制—预算—方法的验证选定配置", collapsed=True)
    prose(f"本地复核通过{audit['assertions']:,}项检查：4,320次开发候选拟合、180次预设消融、5个非监督基线；无记录到的开发拟合失败；1,440个冻结测试案例、8,640行机制结果、17,280行变体明细。复核成功表示来源/标签隔离、哈希、阈值、选参及指标一致，不表示现场有效性已获证明。")
    links = ["test_per_setting.csv", "test_per_variant.csv", "case_summary.csv", "model_budget_summary.csv",
             "matched_ablation_comparison.csv", "paired_source_bootstrap.csv", "cross_repeat_label_exposure.csv",
             "actual_attack_label_costs.csv", "frozen_design.json", "final_selection_freeze.json"]
    sections.append("<ul>" + "".join(f'<li><a href="../results/{name}">{name}</a></li>' for name in links) + "</ul>")
    sections.append('<p><a href="../juno_executed/05_final_test_figures_and_audit_juno_executed.ipynb">Juno结果Notebook</a> · <a href="../juno_executed/06_controls_gains_and_label_costs_local_executed.ipynb">本地诊断Notebook</a></p>')

    heading("10. 全部九组结果图", "figures")
    for image in sorted((ROOT / "figures").glob("*.png")):
        data = base64.b64encode(image.read_bytes()).decode("ascii")
        sections.append('<figure><img loading="lazy" alt="' + html.escape(image.stem) + '" src="data:image/png;base64,' + data + '"><figcaption>' +
                        html.escape(image.stem) + f' · <a href="../figures/{image.stem}.pdf">PDF</a> · <a href="../figures/{image.stem}.svg">SVG</a></figcaption></figure>')
    prose("攻击仍为模拟数据；测试来源此前已被研究。正常用户修改只有9条，重复模型、机制和预算不会增加独立用户数。start-time等生成器线索仍保留，不能将分数解读为任意真实弱攻击或未知机制的保证。")
    navigation = [("design", "设计"), ("global", "全部预算"), ("mechanisms", "逐机制/变体"), ("cross", "跨机制"),
                  ("safety", "误报与稳定性"), ("ablation", "消融"), ("uncertainty", "区间"), ("labels", "标签成本"),
                  ("artifacts", "配置与明细"), ("figures", "全部图片")]
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>六机制EV充电攻击检测：完整实验结果</title><style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;color:#182329;background:#fff;margin:0;line-height:1.65;letter-spacing:0}
main{max-width:1450px;margin:auto;padding:24px}h1{font-size:28px;line-height:1.3}h2{font-size:23px;margin-top:38px;border-bottom:1px solid #cad4d7;padding-bottom:8px}h3{font-size:17px}p{max-width:1100px}a{color:#096588}nav{display:flex;flex-wrap:wrap;gap:8px 20px;padding:12px 0;border-block:1px solid #cad4d7}
.table-wrap{overflow-x:auto;margin:14px 0 24px}table{border-collapse:collapse;min-width:550px;width:max-content;max-width:100%;font-size:13px}th,td{padding:6px 10px;border-bottom:1px solid #dfe4e5;white-space:nowrap;text-align:right}th{background:#edf3f4;color:#172b32;text-align:left}td:first-child{text-align:left}tr:nth-child(even){background:#f8faf9}details{margin:18px 0}summary{cursor:pointer;font-weight:bold;color:#17495a}figure{margin:24px 0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;overflow-wrap:anywhere}code{overflow-wrap:anywhere}li{overflow-wrap:anywhere}@media(max-width:600px){main{padding:14px}h1{font-size:24px}h2{font-size:20px}}
</style></head><body><main><h1>六机制EV充电攻击检测：完整实验结果</h1><p>冻结实验、全部方法与预算、逐机制与变体结果、重复稳定性、消融和来源聚类区间。</p><nav>'''
    document += "".join(f'<a href="#{anchor}">{text}</a>' for anchor, text in navigation) + "</nav>"
    document += "".join(sections) + "</main></body></html>"
    output = ROOT / "reports/full_results_zh.html"
    output.write_text(document, encoding="utf-8")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    assert before == after, "Source results changed during reporting"
    provenance = {"source_csv_sha256": before, "protocol_signature": frozen["signature"],
                  "local_verification_assertions": audit["assertions"],
                  "new_report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "scope": "derived reporting only; no fitting, selection or threshold changes",
                  "global_sd_definition": "sample SD of five repeat means, each mean equally over six mechanisms"}
    (ROOT / "reports/full_results_report_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("HTML_REPORT", output)
    print("SOURCE_CSV_UNCHANGED", len(before))
    print("GLOBAL_STAT_ROWS", len(stats), "HTML_BYTES", output.stat().st_size)


if __name__ == "__main__":
    build()
