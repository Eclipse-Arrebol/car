import random


class EV:
    def __init__(self, ev_id, start_node):
        self.id = ev_id  # 车辆编号
        self.curr_node = start_node  # 当前所在路网节点
        self.target_station_idx = None  # 目标充电站索引，未指派时为 None
        self.soc = random.uniform(20.0, 50.0)  # 荷电状态 SOC（百分比）
        self.target_soc = 95.0  # 目标充电 SOC（百分比）
        self.battery_capacity_kwh = 60.0  # 电池容量 kWh
        self.charge_efficiency = 0.92  # 充电效率（电网能量→电池）
        self.time_value = 35.0  # 时间价值，用于成本/奖励折算
        self.drive_kwh_per_km = 0.18  # 每公里耗电 kWh/km，用于行驶扣 SOC
        self.status = "IDLE"  # 粗粒度状态；move 仅在 MOVING_TO_CHARGE 下推进路网
        self.path = []  # 待走节点序列，队头为下一条边的终点
        self.last_traversed_nodes = []  # 本步内新到达的节点，move 开头清空
        self.current_edge_from = None  # 当前边起点；不在边上为 None
        self.current_edge_target = None  # 当前边终点；不在边上为 None
        self.remaining_edge_time_h = 0.0  # 当前边剩余行驶时间（小时）
        self.current_edge_speed_kph = 0.0  # 当前边限速/速度 km/h
        self.current_edge_length_m = 0.0  # 当前边长度 m
        self.low_soc_triggered = False  # 是否已触发过低 SOC 相关逻辑
        self.charge_decision_pending = False  # 是否待做一次充电/路径决策
        self.remaining_replans = 1  # 剩余可重规划次数
        self.assigned_station = None  # 被指派的充电站（对象或索引，视上层而定）
        self._decision_state = None  # 决策时状态缓存（内部）
        self._decision_snap = None  # 决策快照（内部）

        self.travel_steps = 0  # 处于行驶的仿真步计数
        self.wait_steps = 0  # 排队/等待步计数
        self.charge_steps = 0  # 充电中步计数
        self.travel_time_h = 0.0  # 累计行驶时间（小时）
        self.wait_time_h = 0.0  # 累计等待时间（小时）
        self.charge_time_h = 0.0  # 累计充电时间（小时）
        self.total_fee_paid = 0.0  # 累计支付费用
        self.total_energy_charged = 0.0  # 累计充入电池能量 kWh
        self.charge_sessions = 0  # 完成的充电会话次数
        self.charge_started_count = 0  # 开始充电次数（含未完成）
        self.abandoned_charge_count = 0  # 放弃充电次数
        self.just_abandoned_this_step = False  # 本步是否刚放弃充电

        self.t0_state = None  # 多步遗留：决策锚点 t0 的状态
        self.t0_action = None  # 多步遗留：t0 时刻动作
        self.t0_step = -1  # 多步遗留：t0 所在步号
        self.travel_time_at_dispatch = 0.0  # 多步遗留：派发决策时累计行驶时间
        self.abandon_reason = None  # 放弃充电原因码/说明

    def reset_for_respawn(self, new_node, new_soc):
        self.curr_node = new_node  # 重生后的新起点节点
        self.soc = float(new_soc)  # 重生后的低电量 SOC
        self.target_station_idx = None  # 清空旧站点目标
        self.assigned_station = None  # 清空旧站点引用
        self.path = []  # 清空旧路径
        self.last_traversed_nodes = []  # 清空旧轨迹
        self.current_edge_from = None  # 清空边起点
        self.current_edge_target = None  # 清空边终点
        self.remaining_edge_time_h = 0.0  # 清空剩余边时间
        self.current_edge_speed_kph = 0.0  # 清空边速度
        self.current_edge_length_m = 0.0  # 清空边长度
        self.low_soc_triggered = False  # 重新允许低 SOC 触发
        self.charge_decision_pending = False  # 清空决策挂起
        self.remaining_replans = 1  # 恢复默认重规划次数
        self.status = "IDLE"  # 重生后回到空闲态

    def move(self, env, step_hours=1.0):
        self.last_traversed_nodes = []
        if self.status != "MOVING_TO_CHARGE":
            return

        remaining = max(0.0, float(step_hours))
        moved_hours = 0.0

        while remaining > 1e-9:
            if self.remaining_edge_time_h <= 1e-9:
                if not self.path:
                    break
                next_node = self.path[0]
                length_m, speed_kph, travel_time_h = env.enter_edge(self.curr_node, next_node)
                self.current_edge_from = self.curr_node
                self.current_edge_target = next_node
                self.current_edge_speed_kph = speed_kph
                self.current_edge_length_m = length_m
                self.remaining_edge_time_h = max(1e-6, travel_time_h)

            consume = min(remaining, self.remaining_edge_time_h)
            remaining -= consume
            moved_hours += consume
            self.remaining_edge_time_h -= consume

            if self.remaining_edge_time_h <= 1e-9:
                env.leave_edge(self.current_edge_from, self.current_edge_target)
                self.curr_node = self.current_edge_target
                self.last_traversed_nodes.append(self.curr_node)
                if self.path and self.path[0] == self.current_edge_target:
                    self.path.pop(0)
                self.current_edge_from = None
                self.current_edge_target = None
                self.current_edge_speed_kph = 0.0

        # 按实际走过的时间比例折算行驶距离，再乘单位能耗
        edge_full_time_h = self.remaining_edge_time_h + moved_hours  # 近似整条边时长
        dist_km = moved_hours / max(1e-6, edge_full_time_h) * self.current_edge_length_m / 1000.0
        soc_consumed = dist_km * self.drive_kwh_per_km / self.battery_capacity_kwh * 100.0
        self.soc -= soc_consumed
        self.travel_time_h += moved_hours
        if self.soc < 0:
            self.soc = 0
