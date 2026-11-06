"""Plot original normal/attack power on a cyclic 04:00--04:00 time-of-day axis."""
from pathlib import Path
from datetime import datetime, timedelta
import ast
import hashlib
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT
EXP = ROOT
STEM = "cma_12_attack_power_04to04"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pool_path = RELEASE / "data/processed/fixed_multi_input_attack_pool_seed66.joblib"
    experiment_pool = EXP / "data/processed/fixed_multi_input_attack_pool_seed66.joblib"
    reference_path = RELEASE / "reference/cma_attack_generator_reference.ipynb"
    states_path = RELEASE / "data/processed/multi_input_state_catalog.csv"
    plan_path = RELEASE / "data/processed/multi_input_source_split_plan.json"
    original_audit_path = RELEASE / "reference/original_power_profile_audit.json"
    frozen_path = EXP / "results/frozen_design.json"
    inputs = [pool_path, experiment_pool, reference_path, states_path, plan_path,
              original_audit_path, frozen_path]
    hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in inputs}
    assert sha(pool_path) == sha(experiment_pool), "Attack pools differ"
    pool = joblib.load(pool_path)
    assert pool["control_params"] == {"Catt": .15, "Chav": 30., "Pl": 24., "dt": 5}
    assert sha(reference_path) == pool["attack_reference_sha256"]
    original_audit = json.loads(original_audit_path.read_text())
    notebook = json.loads(reference_path.read_text(encoding="utf-8"))
    source = "\n\n".join("".join(c.get("source", [])) for c in notebook["cells"]
                          if c.get("cell_type") == "code")
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in {"build_power_matrix", "build_power_matrix_ta"}]
    assert sorted(n.name for n in nodes) == ["build_power_matrix", "build_power_matrix_ta"]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    ast_hash = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    assert ast_hash == original_audit["power_function_ast_sha256"]
    namespace = {"np": np, "datetime": datetime, "timedelta": timedelta}
    # Execute only the two original power functions, never attack-generation cells.
    exec(compile(module, str(reference_path), "exec"), namespace)
    power = namespace["build_power_matrix"]
    power_ta = namespace["build_power_matrix_ta"]
    states = pd.read_csv(states_path)
    plan = json.loads(plan_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    test_sources = set(map(int, plan["test_source_ids"]))
    assert test_sources == set(map(int, frozen["roles"]["test"]))
    ids = states.loc[states.source_id.isin(test_sources)
                     & states.event_type.eq("activation"), "state_id"].astype(int).to_numpy()
    assert len(ids) == len(test_sources) == 243
    assert states.loc[states.state_id.isin(ids), "source_id"].nunique() == 243

    def subset(profiles):
        return {key: np.asarray(profiles[key])[ids].copy() for key in ["st", "et", "d", "disc"]}

    dt = int(pool["control_params"]["dt"])
    limit = float(pool["control_params"]["Pl"])
    normal_profiles = subset(pool["normal_profiles"])
    baseline = power(normal_profiles, dt).sum(axis=0)
    hour = np.arange(len(baseline)) * dt / 60.
    arrays = {"local_hour": hour, "normal_power_kW": baseline}
    audit_rows = []
    for i in range(1, 7):
        for coordinated in [False, True]:
            variant = f"Type-{i}" + ("-ta" if coordinated else "")
            attacked = subset(pool["attack_profiles"][variant])
            eligible = set(map(int, pool["attack_state_indices"][variant]))
            local_ids = [j for j, state in enumerate(ids) if int(state) in eligible]
            clock = pool["attack_launch_clock"][variant] if coordinated else None
            if coordinated:
                h, m = map(int, clock.split(":"))
                matrix = power_ta(attacked, dt, datetime(2025, 1, 1, h, m), normal_profiles, local_ids)
            else:
                matrix = power(attacked, dt)
            assert matrix.shape == (243, len(hour)) and np.isfinite(matrix).all()
            assert matrix.min() >= -1e-8 and matrix.max() <= limit + 1e-8
            curve = matrix.sum(axis=0)
            reference = original_audit["per_attack"][variant]
            assert len(local_ids) == reference["eligible_test_activation_sources"]
            assert clock == reference["launch_clock"]
            assert np.isclose(curve.max(), reference["max_aggregate_power_kW"], rtol=0, atol=1e-10)
            assert np.isclose(matrix.max(), reference["max_individual_profile_power_kW"], rtol=0, atol=1e-10)
            delta = curve - baseline
            assert np.allclose(baseline + delta, curve, rtol=0, atol=1e-10)
            arrays[variant + "_power_kW"] = curve
            arrays[variant + "_delta_kW"] = delta
            audit_rows.append({"variant": variant, "coordinated_launch": clock,
                               "eligible_sources": len(local_ids), "delta_min_kW": float(delta.min()),
                               "delta_max_kW": float(delta.max()), "negative_delta_slots": int((delta < -1e-10).sum()),
                               "original_aggregate_max_matches": True})
    data = pd.DataFrame(arrays)
    curves_path = ROOT / "evidence/attack_power_change_curves.csv"
    data.to_csv(curves_path, index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    # Reorder existing slots only. The final point repeats 04:00 to close the cycle.
    first_slot = int(4 * 60 / dt)
    order = np.r_[np.arange(first_slot, len(hour)), np.arange(first_slot)]
    closed_order = np.r_[order, first_slot]
    display_hour = 4 + np.arange(len(closed_order)) * dt / 60.
    assert display_hour[0] == 4 and display_hour[-1] == 28
    assert sorted(order.tolist()) == list(range(len(hour)))
    display_data = pd.DataFrame({"display_hour": display_hour,
                                 "source_local_hour": hour[closed_order]})
    plotted_columns = ["normal_power_kW"] + [f"Type-{i}{suffix}_power_kW"
                                            for i in range(1, 7) for suffix in ["", "-ta"]]
    for column in plotted_columns:
        display_data[column] = data[column].to_numpy()[closed_order]
    display_path = ROOT / "evidence/attack_power_04to04_display.csv"
    display_data.to_csv(display_path, index=False)
    fig, axes = plt.subplots(4, 3, figsize=(7.15, 4.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.09, top=.89, hspace=.55, wspace=.16)
    limits = (0., float(data[plotted_columns].to_numpy().max()) * 1.02)
    plot_order = [f"Type-{i}" for i in range(1, 7)] + [f"Type-{i}-ta" for i in range(1, 7)]
    for index, (ax, variant) in enumerate(zip(axes.flat, plot_order)):
        coordinated = variant.endswith("-ta")
        ax.axvspan(16, 21, color="#D9EAD3", alpha=.45, linewidth=0, zorder=0)
        ax.plot(display_hour, display_data["normal_power_kW"], color="#4472C4", linewidth=1.05)
        ax.plot(display_hour, display_data[variant + "_power_kW"], color="#C00000",
                linestyle="--", linewidth=1.05)
        title = variant
        if coordinated:
            clock = pool["attack_launch_clock"][variant]
            h, m = map(int, clock.split(":"))
            ax.axvline(h+m/60, color=".35", linestyle=":", linewidth=.8, zorder=1)
            title += f" ($t_a$ = {clock})"
        ax.set_title(title, fontsize=9, pad=4)
        ax.set(xlim=(4, 28), ylim=limits, xticks=range(4, 29, 4), yticks=[0, 200, 400])
        ax.set_xticklabels([f"{t % 24:02d}" for t in range(4, 29, 4)])
        ax.tick_params(labelsize=8, length=2.5, pad=2)
        ax.grid(axis="y", alpha=.18, linewidth=.5)
        if index % 3 == 0:
            ax.set_ylabel("Power (kW)", fontsize=9)
        if index >= 9:
            ax.set_xlabel("Local hour", fontsize=9)
    fig.legend(handles=[Line2D([], [], color="#4472C4", lw=1.2, label="Normal inputs"),
                        Line2D([], [], color="#C00000", ls="--", lw=1.2, label="Manipulated inputs")],
               loc="upper center", bbox_to_anchor=(.5, .995), ncol=2, frameon=False, fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        box = ax.title.get_window_extent(renderer)
        parent = ax.get_window_extent(renderer)
        assert box.x0 >= parent.x0-1 and box.x1 <= parent.x1+1, "Panel title exceeds its width"
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT / "paper_figures" / f"{STEM}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    for path in inputs:
        assert sha(path) == hashes[path.relative_to(ROOT.parent).as_posix()]
    report = {"status": "PASS", "source_sha256": hashes, "power_function_ast_sha256": ast_hash,
              "control_params": pool["control_params"], "test_activation_sources": len(ids),
              "time_grid_minutes": dt, "time_grid_slots": len(hour), "raw_delta_definition": "delta = attacked - normal",
              "scope": "time-of-day aggregation of initial test-source profiles, not a chronological event stream",
              "original_power_functions_unchanged": True, "attack_generation_executed": False,
              "display_quantity": "absolute power (kW)", "shared_y_limits_kW": limits,
              "display_start_hour": 4, "display_end_hour": 28,
              "unique_original_slots": len(order), "display_points": len(closed_order),
              "display_source_slot_indices": closed_order.tolist(),
              "endpoint_is_repeated_04_hour_sample": True, "display_curves_sha256": sha(display_path),
              "subplot_shape": [4, 3], "subplot_variants": plot_order, "curves_per_subplot": 2,
              "curves_sha256": sha(curves_path), "per_variant": audit_rows}
    (ROOT / "evidence/attack_power_profile_audit.json").write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(audit_rows).to_string(index=False))


if __name__ == "__main__":
    main()
