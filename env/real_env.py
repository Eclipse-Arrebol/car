import random, sys, os
from typing import Optional

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from env.base_env import TrafficPowerEnv, setup_background_traffic_and_respawn_nodes
from env.entities import EV
from env.charging_station import ChargingStation
from env.power_grid import get_tou_multiplier
from env.power_grid_pp import IEEE33_STATION_BUSES, PPPowerGrid33
from env.osm_loader import load_road_network, load_road_network_by_point, load_road_network_from_file
from env.traffic_profiles import get_traffic_profile


_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_cur)
if _root not in sys.path:
    sys.path.insert(0, _root)


def _safe_path_display(path: str) -> str:
    try:
        return os.path.relpath(path, _root)
    except ValueError:
        return path


class RealTrafficEnv(TrafficPowerEnv):
    def __init__(
        self,
        place: str = "Wuchang District, Wuhan, China",
        num_stations: int = 2,
        num_evs: int = 10,
        max_nodes: int = 30,
        cache_dir: str = None,
        seed: int = 42,
        lat: float = None,
        lon: float = None,
        dist_m: int = 1500,
        graphml_file: str = None,
        offline: bool = False,
        station_node_ids: list = None,
        respawn_after_full_charge: bool = True,
        num_chargers_per_station: int = 4,
        client_name: str = "base",
        background_ue_net_tntp: Optional[str] = None,
        background_ue_trips_tntp: Optional[str] = None,
        background_ue_scale: float = 1.0,
        background_ue_max_iter: int = 800,
        background_ue_tol: float = 1e-4,
        background_ue_verbose: bool = False,
        traffic_profile: str = "base",
    ):
        self.traffic_profile = get_traffic_profile(traffic_profile)
        self.respawn_after_full_charge = respawn_after_full_charge
        self.num_chargers_per_station = max(1, int(num_chargers_per_station))
        self.client_name = client_name
        self.background_ue_net_tntp = background_ue_net_tntp
        self.background_ue_trips_tntp = background_ue_trips_tntp
        self.background_ue_scale = float(background_ue_scale) * self.traffic_profile.ue_scale_multiplier
        self.background_ue_max_iter = int(background_ue_max_iter)
        self.background_ue_tol = float(background_ue_tol)
        self.background_ue_verbose = bool(background_ue_verbose)
        self.background_daily_profile_name = self.traffic_profile.daily_profile
        if graphml_file is not None:
            graph, station_nodes, self.node_positions = load_road_network_from_file(
                filepath=graphml_file,
                num_stations=num_stations,
                max_nodes=max_nodes,
                cache_dir=cache_dir,
                seed=seed,
                station_node_ids=station_node_ids,
            )
        elif lat is not None and lon is not None:
            graph, station_nodes, self.node_positions = load_road_network_by_point(
                lat=lat, lon=lon, dist_m=dist_m,
                num_stations=num_stations,
                max_nodes=max_nodes,
                cache_dir=cache_dir,
                seed=seed,
                offline=offline,
            )
        else:
            graph, station_nodes, self.node_positions = load_road_network(
                place=place,
                num_stations=num_stations,
                max_nodes=max_nodes,
                cache_dir=cache_dir,
                seed=seed,
                offline=offline,
            )

        self.traffic_graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.charge_trigger_soc = 30.0
        self.station_node_ids = station_nodes
        self.num_stations = num_stations

        self.power_grid = (
            PPPowerGrid33.from_client_name(
                self.client_name,
                station_bus_map={
                    i: IEEE33_STATION_BUSES[i]
                    for i in range(num_stations)
                },
            )
            if self.client_name != "base"
            else PPPowerGrid33(
                station_bus_map={
                    i: IEEE33_STATION_BUSES[i]
                    for i in range(num_stations)
                }
            )
        )
        self.stations = self._build_charging_stations(station_nodes)

        non_station = [n for n in graph.nodes() if n not in station_nodes]
        self.evs = []
        for i in range(num_evs):
            start = self._sample_spawn_node(non_station or list(graph.nodes()))
            self.evs.append(EV(i, start))

        self.power_limit = 15.0
        self.time_step = 0
        self.steps_per_day = 144
        self.step_duration_h = 1 / 6
        self.bpr_alpha = 0.15
        self.bpr_beta = 4.0
        self.edge_active_counts = {}
        self.tou_multiplier = 1.0
        self.price_noise = 0.0
        self.prev_total_load = 0.0
        setup_background_traffic_and_respawn_nodes(self)
        self.edge_index = self._build_edge_index()

        self._path_cache_step: dict = {}

        self.lmp_update_interval = 6
        self._cached_lmp = None
        self._lmp_step_counter = 0
        self._completed_evs_this_step = []
        self._abandoned_evs_this_step = []
        self._arrivals_this_step = []
        self._dispatched_t0_this_step = []

        print(f"[RealTrafficEnv] nodes={self.num_nodes}, "
              f"station_nodes={station_nodes}, EVs={num_evs}, "
              f"traffic_profile={self.traffic_profile.name}, "
              f"background_ue_scale={self.background_ue_scale})")

    def reset(self):
        self._reset_mask_stats_and_print()
        self.power_grid = (
            PPPowerGrid33.from_client_name(
                self.client_name,
                station_bus_map={
                    i: IEEE33_STATION_BUSES[i]
                    for i in range(self.num_stations)
                },
            )
            if self.client_name != "base"
            else PPPowerGrid33(
                station_bus_map={
                    i: IEEE33_STATION_BUSES[i]
                    for i in range(self.num_stations)
                }
            )
        )
        self.stations = self._build_charging_stations(self.station_node_ids)
        setup_background_traffic_and_respawn_nodes(self)

        non_station = [n for n in self.traffic_graph.nodes()
                       if n not in self.station_node_ids]
        num_evs = len(self.evs)
        self.evs = []
        for i in range(num_evs):
            start = self._sample_spawn_node(non_station or list(self.traffic_graph.nodes()))
            self.evs.append(EV(i, start))

        self.time_step = 0
        self.steps_per_day = 144
        self.step_duration_h = 1 / 6
        self.edge_active_counts = {}
        self.tou_multiplier = 1.0
        self.price_noise = 0.0
        self.prev_total_load = 0.0
        self._cached_lmp = None
        self._lmp_step_counter = 0
        self._completed_evs_this_step = []
        self._abandoned_evs_this_step = []
        self._arrivals_this_step = []
        self._dispatched_t0_this_step = []
        return self.get_graph_state()

    def _build_charging_stations(self, station_nodes):
        stations = []
        for i in range(self.num_stations):
            station = ChargingStation(
                station_id=i,
                traffic_node_id=station_nodes[i],
                power_node_id=self.power_grid.get_station_power_node(i),
                num_chargers=self.num_chargers_per_station,
                respawn_after_full_charge=self.respawn_after_full_charge,
            )
            station.power_bus_idx = IEEE33_STATION_BUSES[i]
            self._apply_respawn_profile(station)
            stations.append(station)
        return stations

    def _node_xy(self, node):
        positions = getattr(self, "node_positions", {}) or {}
        pos = positions.get(node)
        if pos is None:
            pos = self.traffic_graph.nodes[node]
        if isinstance(pos, dict):
            if "x" in pos and "y" in pos:
                return float(pos["x"]), float(pos["y"])
            if "lon" in pos and "lat" in pos:
                return float(pos["lon"]), float(pos["lat"])
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            return float(pos[0]), float(pos[1])
        return None

    def _spawn_weights(self, nodes):
        nodes = list(nodes)
        if not nodes:
            return []
        style = self.traffic_profile.spawn_style
        if style == "uniform":
            return [1.0 for _node in nodes]

        xy = {node: self._node_xy(node) for node in nodes}
        xy = {node: pos for node, pos in xy.items() if pos is not None}
        if len(xy) < 2:
            return [1.0 for _node in nodes]

        xs = [pos[0] for pos in xy.values()]
        ys = [pos[1] for pos in xy.values()]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        max_r = max(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in xy.values())
        max_r = max(max_r, 1e-9)

        if style == "hotspot":
            sorted_nodes = sorted(xy, key=lambda n: xy[n][0])
            hot_a = xy[sorted_nodes[max(0, int(len(sorted_nodes) * 0.35) - 1)]]
            hot_b = xy[sorted_nodes[max(0, int(len(sorted_nodes) * 0.70) - 1)]]

        weights = []
        for node in nodes:
            pos = xy.get(node)
            if pos is None:
                weights.append(1.0)
                continue
            x, y = pos
            r = (((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) / max_r
            if style == "central":
                weight = 0.15 + 3.0 * (1.0 - r) ** 2
            elif style == "peripheral":
                weight = 0.15 + 2.5 * r ** 2
            elif style == "hotspot":
                da = (((x - hot_a[0]) ** 2 + (y - hot_a[1]) ** 2) ** 0.5) / max_r
                db = (((x - hot_b[0]) ** 2 + (y - hot_b[1]) ** 2) ** 0.5) / max_r
                weight = 0.20 + 2.2 * max((1.0 - da) ** 2, (1.0 - db) ** 2)
            else:
                weight = 1.0
            weights.append(max(0.01, float(weight)))
        return weights

    def _sample_spawn_node(self, candidates):
        candidates = list(candidates)
        if not candidates:
            return random.choice(list(self.traffic_graph.nodes()))
        weights = self._spawn_weights(candidates)
        if weights and sum(weights) > 0.0:
            return random.choices(candidates, weights=weights, k=1)[0]
        return random.choice(candidates)

    def _apply_respawn_profile(self, station):
        candidates = [n for n in self.traffic_graph.nodes() if n not in self.station_node_ids]
        if not candidates:
            candidates = list(self.traffic_graph.nodes())
        station.legal_respawn_nodes = candidates
        station.respawn_node_weights = self._spawn_weights(candidates)
