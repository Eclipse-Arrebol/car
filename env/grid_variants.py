from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
from pandapower.networks import case33bw

ALL_CLIENTS = ["old_city", "new_city", "suburb"]


@dataclass(frozen=True)
class LMPStats:
    mean: float
    std: float
    min: float
    max: float
    p05: float
    p95: float


IEEE33_PV_BUSES = {
    "new_city": [(6, 0.600, "pv_b6"), (18, 0.500, "pv_b18"), (25, 0.400, "pv_b25")],
    "suburb": [(12, 0.200, "pv_b12")],
}


def _scale_loads(net: pp.pandapowerNet, factor: float) -> None:
    if len(net.load):
        net.load.loc[:, "p_mw"] *= factor
        net.load.loc[:, "q_mvar"] *= factor


def _scale_lines(net: pp.pandapowerNet, factor: float) -> None:
    if len(net.line):
        net.line.loc[:, "r_ohm_per_km"] *= factor
        net.line.loc[:, "x_ohm_per_km"] *= factor


def _add_pv(net: pp.pandapowerNet, bus: int, p_mw: float, name: str) -> None:
    pp.create_sgen(
        net,
        bus=bus,
        p_mw=p_mw,
        q_mvar=0.0,
        sn_mva=p_mw * 1.05,
        name=name,
        controllable=False,
        type="PV",
    )


def _add_storage(net: pp.pandapowerNet, bus: int, name: str) -> None:
    pp.create_storage(
        net,
        bus=bus,
        p_mw=0.0,
        max_e_mwh=0.5,
        soc_percent=50.0,
        min_e_mwh=0.0,
        max_p_mw=0.25,
        min_p_mw=-0.25,
        name=name,
        controllable=False,
    )


def _remove_existing_ext_grid_cost(net: pp.pandapowerNet) -> None:
    if len(net.poly_cost) == 0:
        return
    if "et" in net.poly_cost.columns:
        mask = net.poly_cost["et"] == "ext_grid"
    elif "element_type" in net.poly_cost.columns:
        mask = net.poly_cost["element_type"] == "ext_grid"
    else:
        return
    if mask.any():
        net.poly_cost.drop(net.poly_cost.index[mask], inplace=True)
        net.poly_cost.reset_index(drop=True, inplace=True)


def _apply_client_ext_grid_cost(net: pp.pandapowerNet, client_name: str) -> None:
    cost_map = {"old_city": 25.0, "new_city": 18.0, "suburb": 15.0}
    _remove_existing_ext_grid_cost(net)
    ext_grid_idx = int(net.ext_grid.index[0])
    pp.create_poly_cost(net, ext_grid_idx, "ext_grid", cp1_eur_per_mw=cost_map[client_name])


def _apply_line_capacity_scaling(net: pp.pandapowerNet, client_name: str) -> None:
    if len(net.line) == 0:
        return
    fallback = 0.4
    if "max_i_ka" not in net.line.columns:
        net.line.loc[:, "max_i_ka"] = fallback
    net.line.loc[:, "max_i_ka"] = net.line["max_i_ka"].fillna(fallback)
    factor_map = {"old_city": 0.6, "new_city": 1.0, "suburb": 1.2}
    net.line.loc[:, "max_i_ka"] *= factor_map[client_name]


def build_grid_variant(client_name: str, seed: int = 42) -> pp.pandapowerNet:
    """Build a client-specific IEEE 33-bus pandapower network."""

    if client_name not in ALL_CLIENTS:
        raise ValueError(f"Unknown client_name={client_name!r}; expected one of {ALL_CLIENTS}")

    _ = seed
    net = copy.deepcopy(case33bw())

    if client_name == "old_city":
        _scale_loads(net, 1.30)
        _scale_lines(net, 1.15)
        _apply_line_capacity_scaling(net, client_name)
        for bus, name in ((17, "svc_node_17"), (32, "svc_node_32")):
            svc_idx = pp.create_sgen(
                net,
                bus=bus,
                p_mw=0.0,
                q_mvar=0.0,
                sn_mva=1.5,
                name=name,
                controllable=True,
                type="SVC",
            )
            net.sgen.at[svc_idx, "min_q_mvar"] = -1.5
            net.sgen.at[svc_idx, "max_q_mvar"] = 1.5
            net.sgen.at[svc_idx, "min_p_mw"] = 0.0
            net.sgen.at[svc_idx, "max_p_mw"] = 0.0
            try:
                pp.create_poly_cost(
                    net,
                    svc_idx,
                    "sgen",
                    cp1_eur_per_mw=0.001,
                    cq2_eur_per_mvar2=0.0001,
                )
            except TypeError:
                # Older pandapower versions may not accept q-cost kwargs; keep the SVC zero-cost.
                pass
    elif client_name == "new_city":
        _apply_line_capacity_scaling(net, client_name)
        for bus, p_mw, name in IEEE33_PV_BUSES["new_city"]:
            _add_pv(net, bus=bus, p_mw=p_mw, name=name)
    elif client_name == "suburb":
        _scale_loads(net, 0.80)
        _apply_line_capacity_scaling(net, client_name)
        for bus, p_mw, name in IEEE33_PV_BUSES["suburb"]:
            _add_pv(net, bus=bus, p_mw=p_mw, name=name)
        _add_storage(net, bus=22, name="battery_b22")

    _apply_client_ext_grid_cost(net, client_name)
    return net


def _daily_load_curve(hour: int) -> float:
    # Simple two-peak load shape: morning and evening peaks with low overnight demand.
    base = 0.55 + 0.12 * np.cos(2 * np.pi * (hour - 6) / 24.0)
    morning = 0.20 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
    evening = 0.30 * np.exp(-0.5 * ((hour - 19) / 2.5) ** 2)
    night_dip = -0.18 * np.exp(-0.5 * ((hour - 3) / 2.5) ** 2)
    return float(np.clip(base + morning + evening + night_dip, 0.4, 1.0))


def _daily_pv_curve(hour: int) -> float:
    if hour < 6 or hour > 18:
        return 0.0
    return float(np.clip(np.sin(np.pi * (hour - 6) / 12.0) ** 1.8, 0.0, 1.0))


def _apply_hourly_profiles(net: pp.pandapowerNet, hour: int) -> None:
    load_scale = _daily_load_curve(hour)
    pv_scale = _daily_pv_curve(hour)
    if len(net.load):
        net.load.loc[:, "scaling"] = load_scale
    if len(net.sgen):
        net.sgen.loc[:, "scaling"] = pv_scale


def _run_opf_and_collect_lmp(net: pp.pandapowerNet) -> np.ndarray:
    pp.runopp(net)
    if "lam_p" not in net.res_bus:
        raise RuntimeError("OPF did not populate net.res_bus['lam_p']")
    return net.res_bus["lam_p"].to_numpy(dtype=float)


def _summarize(values: np.ndarray) -> LMPStats:
    return LMPStats(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
        min=float(np.min(values)),
        max=float(np.max(values)),
        p05=float(np.percentile(values, 5)),
        p95=float(np.percentile(values, 95)),
    )


def _fmt_stats(stats: LMPStats) -> str:
    return (
        f"mean={stats.mean:.4f}, std={stats.std:.4f}, min={stats.min:.4f}, "
        f"max={stats.max:.4f}, p05={stats.p05:.4f}, p95={stats.p95:.4f}"
    )


def _make_histogram_figure(client_values: Dict[str, np.ndarray], stats_map: Dict[str, LMPStats]) -> Path:
    all_values = np.concatenate([client_values[name] for name in ALL_CLIENTS])
    x_min = float(np.min(all_values))
    x_max = float(np.max(all_values))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for ax, client_name in zip(axes, ALL_CLIENTS):
        vals = client_values[client_name]
        ax.hist(vals, bins=40, range=(x_min, x_max), color="#4C78A8", alpha=0.85, edgecolor="white")
        ax.set_title(f"{client_name} | mean={stats_map[client_name].mean:.3f}")
        ax.set_xlabel("LMP ($/MWh)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Count")
    fig.suptitle("LMP distribution across client grid variants", y=0.98, fontsize=14)
    footer = " | ".join(
        f"{name}: mean={stats_map[name].mean:.3f}, std={stats_map[name].std:.3f}" for name in ALL_CLIENTS
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out_path = Path("lmp_distribution.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _relative_diff(a: float, b: float) -> float:
    denom = max(min(abs(a), abs(b)), 1e-9)
    return abs(a - b) / denom


if __name__ == "__main__":
    client_values: Dict[str, np.ndarray] = {}
    stats_map: Dict[str, LMPStats] = {}

    for client_name in ALL_CLIENTS:
        base_net = build_grid_variant(client_name)
        all_lmps = []
        skipped_hours = 0
        for hour in range(24):
            net = copy.deepcopy(base_net)
            _apply_hourly_profiles(net, hour)
            try:
                lmps = _run_opf_and_collect_lmp(net)
            except pp.OPFNotConverged:
                skipped_hours += 1
                print(f"[skip] {client_name} hour={hour} not converged")
                continue
            except Exception as exc:
                raise RuntimeError(f"OPF failed for {client_name} at hour {hour}: {exc}") from exc
            all_lmps.append(lmps)
        if not all_lmps:
            raise RuntimeError(f"No converged OPF solutions collected for {client_name}")
        values = np.concatenate(all_lmps)
        client_values[client_name] = values
        stats_map[client_name] = _summarize(values)
        suffix = f" | skipped_hours={skipped_hours}" if skipped_hours else ""
        print(f"{client_name}: {_fmt_stats(stats_map[client_name])}{suffix}")

    pdf_path = _make_histogram_figure(client_values, stats_map)
    print(f"Saved histogram figure to {pdf_path.resolve()}")

    pairwise = [("old_city", "new_city"), ("old_city", "suburb"), ("new_city", "suburb")]
    hetero_ok = all(_relative_diff(stats_map[a].mean, stats_map[b].mean) >= 0.20 for a, b in pairwise)
    print("[PASS] LMP heterogeneity OK" if hetero_ok else "[FAIL] LMP heterogeneity insufficient")
