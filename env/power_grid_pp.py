import copy
import time

import pandapower as pp


IEEE33_STATION_BUSES = {
    0: 6,
    1: 9,
    2: 12,
    3: 18,
    4: 22,
    5: 25,
    6: 17,
    7: 33,
}


# Baran-Wu IEEE 33-bus radial feeder line data: from_bus, to_bus, R(ohm), X(ohm).
# Bus numbers follow the conventional 1-based IEEE 33-bus notation.
IEEE33_LINE_DATA = [
    (1, 2, 0.0922, 0.0470),
    (2, 3, 0.4930, 0.2511),
    (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070),
    (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351),
    (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650),
    (11, 12, 0.3744, 0.1238),
    (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129),
    (14, 15, 0.5910, 0.5260),
    (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740),
    (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554),
    (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083),
    (23, 24, 0.8980, 0.7091),
    (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034),
    (26, 27, 0.2842, 0.1447),
    (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585),
    (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619),
    (32, 33, 0.3410, 0.5302),
]


# Standard IEEE 33-bus load data in kW/kVAr. Slack bus 1 has no load.
IEEE33_LOAD_DATA = {
    2: (100, 60),
    3: (90, 40),
    4: (120, 80),
    5: (60, 30),
    6: (60, 20),
    7: (200, 100),
    8: (200, 100),
    9: (60, 20),
    10: (60, 20),
    11: (45, 30),
    12: (60, 35),
    13: (60, 35),
    14: (120, 80),
    15: (60, 10),
    16: (60, 20),
    17: (60, 20),
    18: (90, 40),
    19: (90, 40),
    20: (90, 40),
    21: (90, 40),
    22: (90, 40),
    23: (90, 50),
    24: (420, 200),
    25: (420, 200),
    26: (60, 25),
    27: (60, 25),
    28: (60, 20),
    29: (120, 70),
    30: (200, 600),
    31: (150, 70),
    32: (210, 100),
    33: (60, 40),
}


class PPPowerGrid33:
    """Pandapower IEEE 33-bus distribution grid with the legacy PowerGrid interface."""

    _THEVENIN_CACHE = None

    def __init__(
        self,
        station_bus_map=None,
        compute_thevenin=True,
        client_name: str = "base",
        net=None,
    ):
        self.v_nominal_kv = 12.66
        self.v_min = 0.95
        self.v_max = 1.05
        self.client_name = client_name
        self.station_bus_map = dict(station_bus_map or IEEE33_STATION_BUSES)
        self.station_power_nodes = {
            station_id: f"Bus_{bus}"
            for station_id, bus in self.station_bus_map.items()
        }
        self.power_node_to_bus = {
            power_node: bus
            for station_id, power_node in self.station_power_nodes.items()
            for bus in [self.station_bus_map[station_id]]
        }
        self.base_net = self._materialize_writable_net(net) if net is not None else self._build_ieee33_net()
        self.net = self._materialize_writable_net(self.base_net)
        self.bus_lookup = self._build_bus_lookup(self.net)
        self.line_lookup = self._build_line_lookup(self.net)

        self.bus_voltages = {f"Bus_{i}": 1.0 for i in range(1, 34)}
        self.line_losses = {}
        self.total_loss = 0.0
        self.voltage_violations = []
        self.runpp_call_count = 0
        self.runpp_total_time_s = 0.0
        self.last_runpp_time_s = 0.0
        self.thevenin_r_ohm = {}
        if compute_thevenin:
            if PPPowerGrid33._THEVENIN_CACHE is None:
                PPPowerGrid33._THEVENIN_CACHE = self._compute_thevenin_resistances()
            self.thevenin_r_ohm = dict(PPPowerGrid33._THEVENIN_CACHE)

    @staticmethod
    def _materialize_writable_net(net):
        writable = copy.deepcopy(net)
        for attr in ("bus", "line", "load", "sgen", "storage", "ext_grid", "poly_cost"):
            if hasattr(writable, attr):
                table = getattr(writable, attr)
                if table is not None and hasattr(table, "copy"):
                    setattr(writable, attr, table.copy(deep=True))
        return writable

    @staticmethod
    def _copy_table_columns(src_table, dst_table, columns):
        for col in columns:
            if col in src_table.columns and col in dst_table.columns:
                dst_table.loc[:, col] = src_table.loc[:, col].values

    @classmethod
    def _rebuild_clean_net_from_variant(cls, src_net):
        """Recreate a client variant as a fresh pandapower net.

        This avoids read-only views or cached table state inherited from
        `case33bw()` or previous transformations while preserving the
        variant topology and device parameters.
        """

        clean = pp.create_empty_network(sn_mva=float(getattr(src_net, "sn_mva", 10.0)))
        bus_map = {}
        for _, row in src_net.bus.iterrows():
            bus_idx = pp.create_bus(
                clean,
                vn_kv=float(row.get("vn_kv", 12.66)),
                name=str(row.get("name", "")),
                type=row.get("type", "b"),
                zone=row.get("zone", None),
                in_service=bool(row.get("in_service", True)),
            )
            bus_map[_] = bus_idx

        for _, row in src_net.ext_grid.iterrows():
            ext_idx = pp.create_ext_grid(
                clean,
                bus=bus_map[row["bus"]],
                vm_pu=float(row.get("vm_pu", 1.0)),
                va_degree=float(row.get("va_degree", 0.0)),
                name=row.get("name", None),
                in_service=bool(row.get("in_service", True)),
            )
            for col in ("slack_weight", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"):
                if col in src_net.ext_grid.columns and col in clean.ext_grid.columns:
                    clean.ext_grid.at[ext_idx, col] = row[col]

        for _, row in src_net.line.iterrows():
            line_idx = pp.create_line_from_parameters(
                clean,
                from_bus=bus_map[row["from_bus"]],
                to_bus=bus_map[row["to_bus"]],
                length_km=float(row.get("length_km", 1.0)),
                r_ohm_per_km=float(row.get("r_ohm_per_km", 0.0)),
                x_ohm_per_km=float(row.get("x_ohm_per_km", 0.0)),
                c_nf_per_km=float(row.get("c_nf_per_km", 0.0)),
                max_i_ka=float(row.get("max_i_ka", 0.4)),
                name=row.get("name", None),
                type=row.get("type", "ol"),
                in_service=bool(row.get("in_service", True)),
            )
            for col in ("df", "parallel", "g_us_per_km", "endtemp_degree", "alpha", "temperature_degree_celsius"):
                if col in src_net.line.columns and col in clean.line.columns:
                    clean.line.at[line_idx, col] = row[col]

        if len(getattr(src_net, "load", [])):
            for _, row in src_net.load.iterrows():
                load_idx = pp.create_load(
                    clean,
                    bus=bus_map[row["bus"]],
                    p_mw=float(row.get("p_mw", 0.0)),
                    q_mvar=float(row.get("q_mvar", 0.0)),
                    name=row.get("name", None),
                    in_service=bool(row.get("in_service", True)),
                    scaling=float(row.get("scaling", 1.0)),
                    type=row.get("type", None),
                )
                for col in ("controllable", "const_z_percent", "const_i_percent", "sn_mva", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"):
                    if col in src_net.load.columns and col in clean.load.columns:
                        clean.load.at[load_idx, col] = row[col]

        if len(getattr(src_net, "sgen", [])):
            for _, row in src_net.sgen.iterrows():
                sgen_idx = pp.create_sgen(
                    clean,
                    bus=bus_map[row["bus"]],
                    p_mw=float(row.get("p_mw", 0.0)),
                    q_mvar=float(row.get("q_mvar", 0.0)),
                    sn_mva=float(row.get("sn_mva", 0.0)) if not pp.isna(row.get("sn_mva", None)) else None,
                    name=row.get("name", None),
                    controllable=bool(row.get("controllable", False)),
                    type=row.get("type", None),
                    in_service=bool(row.get("in_service", True)),
                    scaling=float(row.get("scaling", 1.0)),
                )
                for col in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"):
                    if col in src_net.sgen.columns and col in clean.sgen.columns:
                        clean.sgen.at[sgen_idx, col] = row[col]

        if len(getattr(src_net, "storage", [])):
            for _, row in src_net.storage.iterrows():
                storage_idx = pp.create_storage(
                    clean,
                    bus=bus_map[row["bus"]],
                    p_mw=float(row.get("p_mw", 0.0)),
                    max_e_mwh=float(row.get("max_e_mwh", 0.0)),
                    soc_percent=float(row.get("soc_percent", 0.0)),
                    min_e_mwh=float(row.get("min_e_mwh", 0.0)),
                    max_p_mw=float(row.get("max_p_mw", 0.0)),
                    min_p_mw=float(row.get("min_p_mw", 0.0)),
                    name=row.get("name", None),
                    controllable=bool(row.get("controllable", False)),
                    in_service=bool(row.get("in_service", True)),
                )
                for col in ("max_q_mvar", "min_q_mvar"):
                    if col in src_net.storage.columns and col in clean.storage.columns:
                        clean.storage.at[storage_idx, col] = row[col]

        if len(getattr(src_net, "poly_cost", [])):
            for _, row in src_net.poly_cost.iterrows():
                element_type = row.get("et", row.get("element_type", None))
                element = int(row.get("element", 0))
                pp.create_poly_cost(
                    clean,
                    element=element,
                    et=element_type,
                    cp1_eur_per_mw=float(row.get("cp1_eur_per_mw", 0.0)),
                    cp2_eur_per_mw2=float(row.get("cp2_eur_per_mw2", 0.0)),
                    cq1_eur_per_mvar=float(row.get("cq1_eur_per_mvar", 0.0)) if "cq1_eur_per_mvar" in row else 0.0,
                    cq2_eur_per_mvar2=float(row.get("cq2_eur_per_mvar2", 0.0)) if "cq2_eur_per_mvar2" in row else 0.0,
                    standing_cost=float(row.get("standing_cost", 0.0)) if "standing_cost" in row else 0.0,
                )

        return cls._materialize_writable_net(clean)

    @staticmethod
    def _build_ieee33_net():
        net = pp.create_empty_network(sn_mva=10.0)
        bus_indices = {}
        for bus in range(1, 34):
            bus_indices[bus] = pp.create_bus(
                net,
                vn_kv=12.66,
                name=f"Bus_{bus}",
                type="b",
            )

        ext_idx = pp.create_ext_grid(
            net,
            bus=bus_indices[1],
            vm_pu=1.0,
            va_degree=0.0,
            name="Slack_Bus_1",
        )

        pp.create_poly_cost(
            net,
            element=ext_idx,
            et="ext_grid",
            cp1_eur_per_mw=1000.0,
            cp2_eur_per_mw2=0.0,
        )

        for bus, (p_kw, q_kvar) in IEEE33_LOAD_DATA.items():
            pp.create_load(
                net,
                bus=bus_indices[bus],
                p_mw=p_kw / 1000.0,
                q_mvar=q_kvar / 1000.0,
                name=f"Load_{bus}",
            )

        for from_bus, to_bus, r_ohm, x_ohm in IEEE33_LINE_DATA:
            pp.create_line_from_parameters(
                net,
                from_bus=bus_indices[from_bus],
                to_bus=bus_indices[to_bus],
                length_km=1.0,
                r_ohm_per_km=r_ohm,
                x_ohm_per_km=x_ohm,
                c_nf_per_km=0.0,
                max_i_ka=0.4,
                name=f"Line_{from_bus}_{to_bus}",
            )
        return net

    @staticmethod
    def _parse_bus_number(name, fallback_idx):
        text = str(name)
        if text.startswith("Bus_"):
            try:
                return int(text.split("_", 1)[1])
            except (IndexError, ValueError):
                pass
        if text.isdigit():
            return int(text)
        return int(fallback_idx) + 1

    def _build_bus_lookup(self, net):
        return {
            self._parse_bus_number(row["name"], idx): idx
            for idx, row in net.bus.iterrows()
        }

    def _build_line_lookup(self, net):
        lookup = {}
        for idx, row in net.line.iterrows():
            from_bus_idx = row["from_bus"]
            to_bus_idx = row["to_bus"]
            from_bus = self._parse_bus_number(net.bus.at[from_bus_idx, "name"], from_bus_idx)
            to_bus = self._parse_bus_number(net.bus.at[to_bus_idx, "name"], to_bus_idx)
            lookup[f"{from_bus}-{to_bus}"] = idx
        return lookup

    @classmethod
    def from_client_name(cls, client_name: str, station_bus_map=None, compute_thevenin=True):
        from env.grid_variants import build_grid_variant

        variant_net = build_grid_variant(client_name)
        clean_net = cls._rebuild_clean_net_from_variant(variant_net)
        return cls(
            station_bus_map=station_bus_map,
            compute_thevenin=compute_thevenin,
            client_name=client_name,
            net=clean_net,
        )

    def _net_with_loads(self, loads):
        net = copy.deepcopy(self.base_net)
        for power_node, load_kw in loads.items():
            bus_num = self._resolve_bus_number(power_node)
            if bus_num is None:
                continue
            bus_idx = self.bus_lookup[bus_num]
            pp.create_load(
                net,
                bus=bus_idx,
                p_mw=float(load_kw) / 1000.0,
                q_mvar=0.0,
                name=f"EV_Load_{power_node}",
            )
        return net

    def _resolve_bus_number(self, power_node):
        if isinstance(power_node, int):
            return power_node
        if power_node in self.power_node_to_bus:
            return self.power_node_to_bus[power_node]
        if isinstance(power_node, str) and power_node.startswith("Bus_"):
            return int(power_node.split("_", 1)[1])
        return None

    def get_station_power_node(self, station_id):
        return self.station_power_nodes[station_id]

    def _runpp(self, net):
        t0 = time.perf_counter()
        safe_net = self._materialize_writable_net(net)
        pp.runpp(safe_net, algorithm="bfsw", calculate_voltage_angles=False)
        elapsed = time.perf_counter() - t0
        self.runpp_call_count += 1
        self.runpp_total_time_s += elapsed
        self.last_runpp_time_s = elapsed
        return safe_net

    def run_power_flow(self, loads):
        """Run pandapower AC load flow. Loads are kW keyed by Bus_N or station power node."""
        self.net = self._net_with_loads(loads)
        self.net = self._runpp(self.net)

        self.bus_voltages = {}
        self.voltage_violations = []
        for bus_idx, row in self.net.res_bus.iterrows():
            bus_num = self._parse_bus_number(self.net.bus.at[bus_idx, "name"], bus_idx)
            key = f"Bus_{bus_num}"
            vm_pu = float(row.vm_pu)
            self.bus_voltages[key] = round(vm_pu, 6)
            if vm_pu < self.v_min or vm_pu > self.v_max:
                self.voltage_violations.append((key, vm_pu))

        self.line_losses = {}
        for line_idx, row in self.net.res_line.iterrows():
            line_obj = self.net.line.at[line_idx, "name"] if "name" in self.net.line.columns else None
            if isinstance(line_obj, str) and line_obj:
                line_name = line_obj.replace("Line_", "").replace("_", "-")
            else:
                line_row = self.net.line.loc[line_idx]
                from_bus_idx = line_row["from_bus"]
                to_bus_idx = line_row["to_bus"]
                from_bus = self._parse_bus_number(self.net.bus.at[from_bus_idx, "name"], from_bus_idx)
                to_bus = self._parse_bus_number(self.net.bus.at[to_bus_idx, "name"], to_bus_idx)
                line_name = f"Bus_{from_bus}-Bus_{to_bus}"
            self.line_losses[line_name] = round(float(row.pl_mw) * 1000.0, 6)
        self.total_loss = float(self.net.res_line.pl_mw.sum() * 1000.0)
        return self.bus_voltages

    def optimize_power(self, requested_loads):
        """Compatibility placeholder: station-level allocation still lives in ChargingStation."""
        return dict(requested_loads)

    def _compute_thevenin_resistances(self):
        result = {}
        base_net = self._runpp(self.base_net)
        slack_v = self.v_nominal_kv * 1000.0
        perturb_mw = 1.0

        for bus_num, bus_idx in self.bus_lookup.items():
            if bus_num == 1:
                result[bus_num] = 0.0
                continue
            test_net = copy.deepcopy(self.base_net)
            pp.create_load(
                test_net,
                bus=bus_idx,
                p_mw=perturb_mw,
                q_mvar=0.0,
                name=f"Thevenin_Test_{bus_num}",
            )
            test_net = self._runpp(test_net)
            dv_pu = (
                float(base_net.res_bus.at[bus_idx, "vm_pu"])
                - float(test_net.res_bus.at[bus_idx, "vm_pu"])
            )
            # For a small active-power perturbation, R_th ~= dV_phase^2 / dP_3phase.
            r_ohm = max(0.0, dv_pu * (slack_v ** 2) / (perturb_mw * 1e6))
            result[bus_num] = r_ohm
        return result

    def get_bus_thevenin_resistance(self, bus_idx):
        bus_num = self._resolve_bus_number(bus_idx)
        if bus_num is None:
            raise KeyError(f"Unknown bus identifier: {bus_idx}")
        return self.thevenin_r_ohm[bus_num]

    def get_last_bus_voltage(self, bus_idx):
        bus_num = self._resolve_bus_number(bus_idx)
        if bus_num is None:
            raise KeyError(f"Unknown bus identifier: {bus_idx}")
        return self.bus_voltages.get(f"Bus_{bus_num}", 1.0)

    def get_lmp(self):
        try:
            net = copy.deepcopy(self.net)
            pp.runopp(net)
        except Exception:
            return None
        lmp = {}
        for bus_idx, row in net.res_bus.iterrows():
            bus_num = self._parse_bus_number(self.net.bus.at[bus_idx, "name"], bus_idx)
            lmp[bus_num] = float(row.lam_p) / 1000.0
        return lmp


__all__ = [
    "IEEE33_STATION_BUSES",
    "PPPowerGrid33",
]
