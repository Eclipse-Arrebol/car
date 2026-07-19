"""Client-specific IEEE 33-bus grid variants.

These variants are used to create heterogeneous grid-side conditions for
federated or cross-scenario experiments while keeping the same bus numbering as
the base IEEE33 feeder.
"""

from __future__ import annotations

import copy

import pandapower as pp


ALL_GRID_VARIANTS = ["ieee33", "old_city", "new_city", "suburb"]


IEEE33_PV_BUSES = {
    "new_city": [(6, 0.600, "pv_b6"), (18, 0.500, "pv_b18"), (25, 0.400, "pv_b25")],
    "suburb": [(12, 0.200, "pv_b12")],
}


def _bus_lookup(net):
    lookup = {}
    for idx, row in net.bus.iterrows():
        name = str(row.get("name", ""))
        if name.startswith("Bus_"):
            lookup[int(name.split("_", 1)[1])] = idx
        else:
            try:
                lookup[int(name)] = idx
            except ValueError:
                pass
    return lookup


def _scale_loads(net, factor):
    if len(net.load):
        net.load.loc[:, "p_mw"] *= factor
        net.load.loc[:, "q_mvar"] *= factor


def _scale_lines(net, factor):
    if len(net.line):
        net.line.loc[:, "r_ohm_per_km"] *= factor
        net.line.loc[:, "x_ohm_per_km"] *= factor


def _add_pv(net, bus_num, p_mw, name):
    bus_idx = _bus_lookup(net)[bus_num]
    pp.create_sgen(
        net,
        bus=bus_idx,
        p_mw=p_mw,
        q_mvar=0.0,
        sn_mva=p_mw * 1.05,
        name=name,
        controllable=False,
        type="PV",
    )


def _add_storage(net, bus_num, name):
    bus_idx = _bus_lookup(net)[bus_num]
    try:
        pp.create_storage(
            net,
            bus=bus_idx,
            p_mw=0.0,
            max_e_mwh=0.5,
            soc_percent=50.0,
            min_e_mwh=0.0,
            max_p_mw=0.25,
            min_p_mw=-0.25,
            name=name,
            controllable=False,
        )
    except TypeError:
        # Older pandapower versions have a narrower create_storage signature.
        pp.create_storage(
            net,
            bus=bus_idx,
            p_mw=0.0,
            max_e_mwh=0.5,
            soc_percent=50.0,
            name=name,
        )


def _remove_existing_ext_grid_cost(net):
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


def _apply_ext_grid_cost(net, variant):
    # Keep LMPs on the same scale as the base IEEE33 setup. Values are in
    # EUR/MWh before PPPowerGrid33.get_lmp converts them to per-kWh prices.
    cost_map = {"ieee33": 1000.0, "old_city": 1200.0, "new_city": 900.0, "suburb": 800.0}
    _remove_existing_ext_grid_cost(net)
    ext_grid_idx = int(net.ext_grid.index[0])
    pp.create_poly_cost(
        net,
        element=ext_grid_idx,
        et="ext_grid",
        cp1_eur_per_mw=cost_map[variant],
        cp2_eur_per_mw2=0.0,
    )


def _apply_line_capacity_scaling(net, variant):
    if len(net.line) == 0:
        return
    if "max_i_ka" not in net.line.columns:
        net.line.loc[:, "max_i_ka"] = 0.4
    net.line.loc[:, "max_i_ka"] = net.line["max_i_ka"].fillna(0.4)
    factor_map = {"ieee33": 1.0, "old_city": 0.6, "new_city": 1.0, "suburb": 1.2}
    net.line.loc[:, "max_i_ka"] *= factor_map[variant]


def _add_old_city_svcs(net):
    lookup = _bus_lookup(net)
    for bus_num, name in ((17, "svc_node_17"), (32, "svc_node_32")):
        svc_idx = pp.create_sgen(
            net,
            bus=lookup[bus_num],
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
            pass


def build_grid_variant(variant, base_net):
    """Return a variant of the provided IEEE33 pandapower network."""

    if variant not in ALL_GRID_VARIANTS:
        raise ValueError(f"Unknown grid_variant={variant!r}; expected one of {ALL_GRID_VARIANTS}")

    net = copy.deepcopy(base_net)
    if variant == "old_city":
        _scale_loads(net, 1.30)
        _scale_lines(net, 1.15)
        _apply_line_capacity_scaling(net, variant)
        _add_old_city_svcs(net)
    elif variant == "new_city":
        _apply_line_capacity_scaling(net, variant)
        for bus_num, p_mw, name in IEEE33_PV_BUSES["new_city"]:
            _add_pv(net, bus_num, p_mw, name)
    elif variant == "suburb":
        _scale_loads(net, 0.80)
        _apply_line_capacity_scaling(net, variant)
        for bus_num, p_mw, name in IEEE33_PV_BUSES["suburb"]:
            _add_pv(net, bus_num, p_mw, name)
        _add_storage(net, 22, "battery_b22")
    else:
        _apply_line_capacity_scaling(net, variant)

    _apply_ext_grid_cost(net, variant)
    return net


__all__ = ["ALL_GRID_VARIANTS", "build_grid_variant"]
