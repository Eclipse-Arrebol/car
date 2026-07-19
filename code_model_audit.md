# 当前项目“论文方法建模审计”

审计日期：2026-07-16  
审计对象：当前工作区 `G:\car_charge\car_charge` 的实际文件状态（含未提交修改）  
审计原则：以可执行调用链为准；注释、类名、参数名和历史分支只作线索。

## 0. 审计边界与结论先行

1. 用户指定的 `paper_draft_cn_v3.md` 不在当前工作区、附件目录或当前 Git 可见历史中。本报告无法对 v3 逐句核对；第 15 节仅把仓库中的 `docs/paper_draft.md` 作为“可见旧稿”审计，所有针对 v3 的结论均标记为待人工定位。
2. 当前论文相关主路径不是根目录的 `main.py`。`main.py:13,44` 导入不存在的 `training.config` / `training.trainer`，当前不可作为可执行统一入口。实际论文实验路径是：
   - 单地：`train_hindsight.py:125-264`；
   - 集中式：`train_centralized_hindsight.py:151-230`；
   - 联邦：`train_federated_hindsight.py:353-456`；
   - FedAvg/FedProx/FedRep 对比编排：`run_fl_methods.py:31-65,85-131`；
   - 多范式编排：`run_paradigm_training.py:46-142,167-249`。
3. 当前可复现执行中不存在 UE 求解。`env/background_traffic.py:88-102` 尝试导入 `env.background.ue_assignment`，但当前工作区没有可导入源模块，实测为 `ModuleNotFoundError`，随后静默回退 `env/background_traffic.py:43-61` 的启发式边流。历史提交 `f399915` 曾包含该模块，但不属于当前检出版本，不能算当前实现。
4. BPR 确实用于预计路径和实际穿边时间，但代码不是标准的 `x/C`，而是 `x/(C t0)`，把容量从 veh/h 转成“自由流时长内可容纳车辆数”；当前背景流是无物理单位的启发式数值，单位解释仍不完整。
5. `pandapower.runpp` 确实每个环境步在充电功率形成后执行一次 AC 潮流；网络为代码自建的 IEEE 33 节点、32 线路、0 变压器径向网。`runopp` 也每 6 步尝试一次并读取 `res_bus.lam_p`，失败则静默退回 ToU 电价。
6. 默认论文网络 `station_only` 不读取 `vm_pu`（特征 6），也不读取线路损耗/过载。因此 **AC `runpp` 结果不直接进入当前策略或 hindsight 奖励**。电网可能通过另一次 `runopp` 得到的节点有功边际价格进入价格特征和费用。
7. 充电功率是离散时步下的“两段分段近似 CC-CV”：80% 以下 60 kW，80%–95% 线性降到 1 kW 下限，再受站级 600 kW 比例缩放；不是连续电化学 CC-CV 模型。
8. 训练奖励在“开始充电”而非“充满/完成充电”时回填；行驶与等待时间是已实现值，费用是开始时 LMP × 预计购电量的快照，不是充电过程逐时累计的最终费用。放弃样本被丢弃，无放弃惩罚。
9. 所有 hindsight 样本均 `done=True, next_state=None`，故 Double DQN 目标严格退化为 `y=r`。代码维护 online/target 网络，但当前论文学习问题实质是带回放的终端上下文 bandit/Q 回归，而不是多步 bootstrapped DQN。
10. 当前 `fedrep` 是“共享 `encoder.*`、本地保留 `head.*`”的部分参数聚合，没有原始 FedRep 的 head/representation 分阶段局部优化。应称“FedRep 式部分参数共享”或“共享编码器与本地决策头的个性化联邦训练”。

## 1. 真实程序总体流程

### 1.1 初始化

以 `train_federated_hindsight.py` 为当前最完整论文路径：

1. `main()` 解析城市/网格、交通比例和联邦参数（`train_federated_hindsight.py:173-242,353-365`）。
2. `build_env()` 构造 `RealTrafficEnv`（`train_federated_hindsight.py:245-270`）。
3. `RealTrafficEnv.__init__()` 加载 `map_outputs/ema/ema.graphml` 或其 pickle 缓存（`env/real_env.py:68-96`；`env/osm_loader.py:183-295`）。当前缓存图为 74 节点、129 条无向边，4 个站点交通节点为 `[59, 2, 49, 15]`；图来自 TNTP 转 GraphML，坐标由 spring layout 生成，不应称真实测绘 OSM 路网（`env/tntp_loader.py:124-181`）。
4. `_build_power_grid()` 创建 IEEE33 变体（`env/real_env.py:103,171-178`）；`_build_charging_stations()` 将 4 站映射至配电母线 6、9、12、18（`env/real_env.py:180-191`；`env/power_grid_pp.py:9-18`）。
5. 构造 EV：初始 SOC 均匀采样 `[20,50]%`，60 kWh 电池（`env/real_env.py:106-110`；`env/entities.py:5-15`）。
6. `setup_background_traffic_and_respawn_nodes()` 构造日变化曲线和背景边流（`env/base_env.py:20-36`）。当前因 UE 模块不可导入，实际得到启发式基线。
7. 构造图张量：`x:[74,19]`、`edge_index:[2,258]`、`edge_attr:[258,2]`（`env/base_env.py:469-481,595-625`）。
8. `build_agent()` 创建 online/target `StationOnlyGraphQNetwork`；`HindsightTrainer` 持有尚未兑现奖励的派发条目（`train_federated_hindsight.py:273-287,367-379`；`trainer/trainer.py:30-35`）。

### 1.2 每个 10 分钟时隙的真实顺序

`HindsightTrainer.step_episode()` 和 `TrafficPowerEnv.step()` 联合定义顺序：

1. 在环境推进前，筛选 `IDLE` 且 SOC<30% 的 EV，按 SOC 升序排序（`env/base_env.py:205-226`）。
2. 对第 `j` 辆待决策 EV，用本时隙前序车辆的 `pending_counts` 构造个体状态、可选动作掩码并 ε-greedy 选站；动作尚未写入真实站点（`trainer/trainer.py:114-125`）。
3. 保存 `(state, action, mask)` 为 pending transition（`trainer/trainer.py:43-57`）。
4. `env.step(actions)` 先令 `time_step += 1`，更新背景流和清空本步缓存（`env/base_env.py:849-867`）。注意：决策状态使用推进前背景流，实际派发路径使用推进后的背景流。
5. 再按 SOC 升序重建本步 `decision_metrics` 和提交计数（`env/base_env.py:869-877`）。
6. EV 更新（`env/base_env.py:879-978`）：
   - `IDLE` 每步固定 SOC−0.5；有动作则按 BPR 时间最短路派发，无动作则随机跳到邻居；
   - `MOVING_TO_CHARGE` 调 `EV.move()` 在固定 1/6 h 内穿越若干边；
   - 到站后 append 至站点队列；
   - `WAITING` 的等待时间每步 +1/6 h，4 h 超时放弃；
   - `CHARGING` 仅增加计时，实际 SOC/费用在站点模块更新。
7. 用本步到站数更新站点到达 EMA（`env/base_env.py:980-981`；`env/charging_station.py:125-131`）。
8. 更新 ToU 和全局价格噪声；每 6 步尝试一次 `runopp` 计算节点边际价格，输入为各站上一步 `last_total_load`（`env/base_env.py:983-995`）。
9. 每站 `ChargingStation.step()`（`env/base_env.py:996-1019`；`env/charging_station.py:189-256`）：
   - 更新价格；
   - 队首 FCFS 填充空桩；
   - 计算 CC-CV 请求功率并按站级 600 kW 比例限幅；
   - 对新开充电车辆触发 `on_charge_started` 回调；
   - 离散更新所有在充车辆 SOC、累计实际费用；
   - SOC≥95% 时完成并释放连接列表。
10. 汇总站点 kW 至配电母线，`run_power_flow(grid_loads)` 调 `pp.runpp(..., algorithm="bfsw")`（`env/base_env.py:1010-1021`；`env/power_grid_pp.py:224-252`）。
11. 计算一个包含用户、队列、电网成本、功率波动和电压违约的旧环境标量 reward（`env/base_env.py:1031-1049`），但当前 hindsight 训练入口丢弃该值（`trainer/trainer.py:127`）。
12. `HindsightTrainer` 处理新开充电事件，在充电开始时计算单车 hindsight reward 并把 terminal transition 写入回放缓冲（`trainer/trainer.py:59-95,129-133`）。
13. 每个环境步后，缓冲达到 batch size 即本地回放训练（`train_federated_hindsight.py:304-330`）。
14. 每轮所有客户端完成后，服务端按当轮新写入 transition 数加权聚合（`train_federated_hindsight.py:400-435`）。

### 1.3 真实数据流（按代码顺序）

当前可复现数据流不是题设给出的 UE 链，而是：

`启发式背景边流基线`  
→ `日曲线×正弦因子后的背景并发量`  
→ `背景量 + 当前受控 EV 边占用 + 候选车`  
→ `非标准容量换算 BPR 边时间`  
→ `最短时间路径/锁定的穿边时间`  
→ `FCFS 离散队列`  
→ `分段近似 CC-CV 站级功率`  
→ `站点 kW 映射 IEEE33 母线`  
→ `pandapower AC runpp：电压与有功线损`  
→ `下一状态完整图中的 vm_pu（但默认 Q 网络不读）`。

另一条价格支路为：

`上一步站点负荷`  
→ `每 6 步 pandapower runopp`  
→ `res_bus.lam_p/1000`（失败则 ToU×随机扰动）  
→ `站点 current_price / current_lmp`  
→ `默认 Q 网络价格特征 + 充电开始费用快照`。

学习支路为：

`决策前个体状态 + 同时隙提交计数`  
→ `站点集合 Q 网络`  
→ `选站`  
→ `到达并开始充电时的行驶/等待实现值 + 费用快照`  
→ `terminal transition`  
→ `本地回放 Q 回归`  
→ `FedAvg/FedProx/编码器部分聚合`。

## 2. UE 背景交通流审计

| 问题 | 当前代码事实 | 证据 |
|---|---|---|
| UE 的含义 | 设计意图为 TNTP OD 需求在 BPR 路阻下的用户均衡背景交通分配；当前未执行 | `train_hindsight.py:61-87` 仅为参数/注释；实际导入在 `env/background_traffic.py:88-102` 失败回退 |
| 当前来源 | 初始化和每次 reset 重新生成一次启发式空间基线；每时隙只乘时间曲线，不重新求均衡 | `env/base_env.py:20-36,127-168,403-409` |
| 算法 | 当前执行为 `_heuristic_base_background_flows`，不是 Frank–Wolfe/MSA/Beckmann | `env/background_traffic.py:43-61` |
| OD | 当前执行不读取 OD。文件 `EMA_trips.tntp` 声明 74 zones、总 OD flow 65576.375431，但当前模块不解析它 | `map_outputs/ema/EMA_trips.tntp:1-6`；调用在异常前后见 `env/background_traffic.py:83-102` |
| 受控 EV | 受控 EV 用 `edge_active_counts` 单独叠加 | `env/base_env.py:389-390,416-424,448-467` |
| 时间变化 | 基线乘归一化早晚双峰 `profile[t]` 和另一个 5% 正弦因子 | `env/background_traffic.py:12-25`；`env/base_env.py:403-409` |
| 城市 `ue_scale` | 只有成功得到 `ue_flows` 后才乘；当前回退路径在 `:83-91` 直接返回 heuristic，故 1.3/1.0/0.7 **完全不起作用** | `env/background_traffic.py:81-91,107-114` |

当前不能写

`x[e,t] = x_UE[e,t] + x_EV[e,t]`。

严格对应当前执行的是：

`x_query[e,t] = h[e] * d[t] * (1 + 0.05 sin(2πt/144)) + n_active[e,t] + 1`，

其中 `h[e]` 是 0.5–约 3.0 的启发式无量纲边值，`n_active` 是当前受控 EV 并发占用数，末尾 `+1` 是候选/即将进边车辆。代码：`env/background_traffic.py:43-61`、`env/base_env.py:403-424,427-439`。

重复计数风险：当前背景不是由受控 EV 生成，因此代码层面没有同一对象重复计数。若未来恢复 TNTP UE，OD 是否已包含论文中的受控 EV 群体属于语义问题，代码没有排除机制，需在数据定义中明确。

## 3. BPR 道路时间

### 3.1 实际公式与单位

代码先由边长和速度计算：

`t0[e] = length_m[e] / 1000 / speed_kph[e]`（小时），见 `env/base_env.py:366-385`。

实际 BPR 是：

`C_concurrent[e] = max(1, capacity_vehph[e] * t0[e])`

`t[e,t] = max(1e-6, t0[e] * (1 + 0.15 * (x[e,t]/C_concurrent[e])^4))`

见 `env/base_env.py:411-414`。参数 `alpha=0.15, beta=4.0` 在 `env/real_env.py:114-117`。

这不是论文常见的直接 `x/C`，因为代码把 `capacity`（veh/h）乘 `t0`（h）转成边上的并发车辆容量。`x` 实际由启发式背景值、受控 EV 当前占边数量和候选车组成；只有受控 EV 部分明确是“辆”。

当前 EMA 图边属性由 TNTP 双向记录平均得到，例：74 节点、129 无向边；TNTP 原文件声明 258 有向 link，容量约 veh/h、自由流时间 h（`map_outputs/ema/EMA_net.tntp:1-10`；`env/tntp_loader.py:101-121`）。

### 3.2 路径与预计/实现值

- 选站状态：每站重新求以动态 BPR 时间为权重的最短路，并缓存同一步同 OD 结果（`env/base_env.py:483-508`）。
- 派发时：`env.step` 再求一次最短时间路径（`env/base_env.py:888-904`）。
- 实际穿边：进入边时锁定当时 BPR 时间，在一个 1/6 h 时步内推进（`env/entities.py:66-100`）。在途重规划被硬编码禁用（`env/base_env.py:270-272`）。
- 状态使用预计 `trip_time_h`（特征 10）；hindsight 奖励使用到充电开始时累计的 `travel_time_h` 实现值（`env/base_env.py:629-639,769-789`；`trainer/trainer.py:77-82`）。

时隙长度为 `1/6 h = 10 min`（`env/real_env.py:112-117`）。

潜在问题：决策状态在 `env.step` 前构造，而 `env.step:850-851` 先推进时刻并更新背景交通后才真正计算路径，预计状态与实际派发存在一个时隙的背景流错位。

## 4. 车辆能耗与 SOC

常量：SOC 为百分数；电池 60 kWh；行驶能耗 0.18 kWh/km；触发阈值 30%；目标 95%；充电效率 0.92（`env/entities.py:9-14`；`env/base_env.py:51,205-221`）。

意图公式为：

`E_drive = distance_km * 0.18` kWh；

`SOC_new = max(0, SOC_old - 100 E_drive/60)`。

代码见 `env/entities.py:101-108`。但 `distance_km` 的实现只使用 `current_edge_length_m` 和本次 `moved_hours` 的比例：

`distance_km = moved_hours/(remaining_edge_time_h + moved_hours) * current_edge_length_m/1000`。

若一个 10 分钟步跨越多条边，它只保留最后一条边长度，不能准确累计所有已走边，属于潜在能耗 bug。另有两个模型外行为：

- `IDLE` EV 每步固定 SOC−0.5，与距离/速度无关（`env/base_env.py:879-881`）；
- 无充电动作的 `IDLE` EV 会瞬间随机跳到邻居，不记录旅行时间，只承受固定 SOC−0.5（`env/base_env.py:920-923`）。

因此论文若写车辆全程物理行驶能耗模型，表述过强；只能写“充电导航行驶阶段采用固定单位里程耗电的离散近似”，并披露上述实现限制。

## 5. 充电站排队

- 默认论文规模每站 8 桩；类默认 4 桩；运行脚本以 CLI 覆盖（`env/charging_station.py:6-16`；`run_fl_methods.py:51-58`）。
- 队列为 Python list；到站 append，开桩 pop(0)，是 FCFS（`env/base_env.py:239-249`；`env/charging_station.py:195-206`）。
- 不维护每个桩的绝对空闲时刻。系统以 10 分钟离散时步推进 `connected_evs` 和 `queue`。
- 实际等待时间：处于 `WAITING` 的每步 +1/6 h；当站点下一次 `step()` 有空桩即开始充电（`env/base_env.py:954-974`）。
- 结束后从 `connected_evs` 删除，释放桩（`env/charging_station.py:224-255`）。

预计等待并非题设的 `min_k b_sk` 模型，而是：

`L = len(queue) + floor(max(0,incoming_count))`；

若 `L<=0`，`W_hat=0`；否则

`W_hat = min_{connected i} E_i/(eta p_i) + (L/S) * mean_charge_time`。

见 `env/charging_station.py:52-72`。问题：当所有桩均占用但当前 queue 为空且 `incoming_count=0` 时直接返回 0，忽略在充车辆剩余时间；候选车本身也未显式计入 `L`。

在途/前瞻变量严格区分：

- `evs_heading_to[s]`：已派发、尚未到站的真实在途数（`env/base_env.py:170-184`）；
- `station.predicted_arrivals`：历史到站数的 EMA，alpha=0.3（`env/charging_station.py:32-34,125-131`）；
- `pending_counts[s]`：当前时隙前序决策已提交但尚未执行的临时数（`trainer/trainer.py:117-125`）。

预计等待函数只接收 `pending_counts`，不直接接收历史在途数或 arrival EMA；默认网络通过特征 18 间接看到在途+pending 的比例。

## 6. CC-CV 充电

实现是分段近似，不是连续电池电压/电流状态方程。

单车请求功率（kW）：

`p_req(SOC) = 60`, 当 `SOC<80`；

`p_req(SOC) = max(1, 60*(95-SOC)/(95-80))`, 当 `80<=SOC<95`。

见 `env/charging_station.py:133-149`。若总请求超过站级上限 600 kW：

`rho = min(1, 600/sum_i p_req_i)`，`p_i = rho p_req_i`（`env/charging_station.py:151-171`）。

离散 SOC 更新：

`SOC_i(t+1) = min(100, SOC_i(t) + 100*p_i(t)*(1/6)*0.92/60)`，

结束条件实际硬编码 `SOC>=95`（`env/charging_station.py:213-224`）。

限制：

- 60 kW 同时充当桩额定功率和车辆请求上限，没有独立车辆最大接收功率；
- 站级 600 kW 是固定容量，不由 pandapower 电压/线路约束动态收紧；
- EV 负荷无功固定 0（`env/power_grid_pp.py:196-209`），等价于功率因数 1；
- `estimate_charge_time_hours()` 对 CV 段用 30 kW 平均功率近似，而实际功率线性 taper，预计时长与离散实现不完全一致（`env/charging_station.py:88-115`）。

## 7. 电价和费用

### 7.1 价格

每站 `current_price`：

1. 若 `runopp` 成功，`energy_price = lam_p/1000 * (1+noise)`；
2. 否则 `energy_price = base_price * ToU * (1+noise)`；
3. `current_lmp = max(0.1, energy_price)`；
4. `current_price = max(0.1, current_lmp + 0.15 + 0.08*(queue+connected))`。

证据：`env/power_grid_pp.py:299-309`；`env/charging_station.py:37-50`。噪声为每步全站共享的 `Uniform(-0.1,0.1)`，不是站点独立噪声（`env/base_env.py:983-1008`）。

`runopp` 每 6 步尝试一次；当前步前使用上一步 `last_total_load`。异常被 `get_lmp()` 捕获并返回 `None`，没有日志或显式成功标志。已有评估文件中 old/new/suburb 平均 fee 明显约为 90/58/53（如 `evaluation/paradigm/matrix_sparse15_hetero.json`），与 1200/900/800 的外部电网成本差异相符，是 OPF 价格在历史运行中生效的旁证，但不是每次调用成功的审计日志。

ToU 有单位 bug：`get_tou_multiplier()` 把 `time_step % 144` 直接当“小时”与 7、10、15、18、23 比较（`env/power_grid.py:4-11`）。结果只有一天最前 23 个 10 分钟步按时段变化，其余 121 步都落入 `hour>=23` 的低谷 0.5；不能按标准 24 h ToU 公式写入论文。

### 7.2 三种费用概念

1. 状态预计费用：`current_price * remaining_battery_energy/efficiency`（`env/charging_station.py:117-123`）。
2. hindsight reward 费用快照：在充电开始时
   `F_start = current_lmp * remaining_battery_energy/efficiency`，不含 0.15 服务费和 0.08 拥堵加价（`env/charging_station.py:181-187,195-204`）。
3. 车辆实际累计费用：
   `F_actual += p_i(t)*current_price(t)*Delta_t`，会受 CC-CV 功率和逐步价格影响（`env/charging_station.py:215-222`），但当前 hindsight reward 不使用它。

因此论文奖励费用不能写成逐时求和；严格应写充电开始价格快照公式。

## 8. pandapower 配电网

| 项目 | 实现 |
|---|---|
| 网络 | 代码自建 Baran–Wu IEEE 33-bus 径向网，不是 pandapower 内置 case |
| 规模 | 33 bus，32 line，0 trafo，1 ext_grid，32 个基础 load；每次潮流再按站创建 EV load |
| 电压等级 | 12.66 kV；slack 1.0 pu；审计阈值 0.95–1.05 pu |
| 站点母线 | 最多 8 站依次映射 6、9、12、18、22、25、17、33；当前 4 站用前四个 |
| 基础负荷 | `IEEE33_LOAD_DATA`，kW/kVAr 除 1000 转 MW/Mvar |
| EV 负荷 | 站点 kW 除 1000 转 MW，`q_mvar=0` |
| AC 潮流 | 每环境步 `pp.runpp(..., algorithm="bfsw")` |
| OPF | 每 6 步尝试 `pp.runopp`，只读 `res_bus.lam_p` |
| 读取结果 | `vm_pu`、`res_line.pl_mw`；不读取 line loading、trafo loading、无功损耗 |
| 不收敛 | `runpp` 无 try/except，会中断仿真；`runopp` 捕获所有异常并无声退回 ToU |
| 安全约束 | 只统计 0.95–1.05 电压违约；不据此掩码或限功率；未实现线路/变压器过载动作约束 |

代码：`env/power_grid_pp.py:21-93,101-143,145-252,299-309`。

第一次构造每个 grid variant 时还会通过 33 次左右的 AC 潮流扰动计算 Thevenin 电阻（`env/power_grid_pp.py:258-285`）；这是状态估计辅助量，不是每步主潮流。其预计电压结果在 `_estimate_ev_station_metrics` 中计算，却没有写进 19 维状态或奖励（`env/base_env.py:517-588`），当前属于死端诊断量。

## 9. 交通—配电耦合闭环逐箭头审计

| 箭头 | 输入→输出 | 单位 | 是否执行/进入默认策略 | 代码 |
|---|---|---|---|---|
| 选站→移动 | action→目标站与最短时间路径 | station id→节点序列 | 是 | `env/base_env.py:879-904` |
| 移动→到站 | path/BPR time→curr_node/queue append | h、km、%SOC | 是 | `env/entities.py:66-108`; `env/base_env.py:925-952` |
| 到站→排队 | EV→station.queue | EV 对象 | 是，FCFS 离散 | `env/base_env.py:239-249` |
| 排队→充电 | queue head→connected_evs | EV 对象 | 是 | `env/charging_station.py:195-206` |
| 充电→站负荷 | SOC→CC-CV power allocation | kW | 是 | `env/charging_station.py:137-171,208-222` |
| 站负荷→节点 | power_node_id→EV load | kW→MW | 是 | `env/base_env.py:996-1021`; `env/power_grid_pp.py:196-209` |
| 节点→AC 潮流 | P/Q load→vm_pu/active loss | MW/Mvar→pu/kW | 是 | `env/power_grid_pp.py:224-252` |
| AC 结果→下一状态 | bus_voltages→feature 6 | pu | 完整图中是；默认 `station_only` 不读 | `env/base_env.py:613-614`; `agents/network_station_only.py:20-30` |
| AC 结果→奖励 | voltage/loss→旧 env reward | pu/kW | 旧标量有，hindsight 训练无 | `env/base_env.py:1039-1049`; `trainer/trainer.py:77-93` |
| 负荷→OPF 价 | 上步站负荷→lam_p | kW→货币/kWh | 条件执行，成功时默认网络读取价格 | `env/base_env.py:986-1008`; `env/power_grid_pp.py:299-309` |
| 价格→后续选站 | current_price→feature 3 | 货币/kWh | 是，默认网络读 idx3 | `env/base_env.py:610,631-636`; `agents/network_station_only.py:20-28` |

结论：存在“动作—充电负荷—AC 潮流”的物理仿真链，也存在“负荷—OPF 边际价—后续选站”的条件价格反馈链；但 **AC `runpp` 的电压/线损结果没有进入默认策略和训练奖励**。不能写成“电压安全约束驱动的 RL 闭环”。

## 10. 奖励

### 10.1 当前训练奖励

`trainer/trainer.py:6-17,77-93`：

`r_i = -(wT*T_actual/0.58 + wW*W_actual/0.17 + wF*F_start/65)`。

默认权重 `(0.4,0.4,0.2)`；`HETERO_REWARD=1` 时：old_city `(0.15,0.70,0.15)`，new_city `(0.15,0.15,0.70)`，suburb `(0.70,0.15,0.15)`，均和为 1（`reward_profiles.py:18-35`）。

- `T_actual`：派发至充电开始的累计行驶时间；
- `W_actual`：到站至充电开始的累计离散等待；
- `F_start`：开始时 `current_lmp × 预计购电量`，不是最终实际累计费；
- 生成时点：充电开始，不是充满；
- 写入回放：`done=True,next_state=None`；
- 放弃：pending 条目直接删除，不写负奖励（`trainer/trainer.py:100-112`）。

0.58、0.17、65 只以字面常量出现，当前代码/文档没有计算来源、数据集统计脚本或版本化元数据。只能写“人工设定/来源未确认的归一化常数”，不能声称来自训练集最大值、测试集或经验上界。

### 10.2 未用于当前训练的环境 reward

`env/base_env.py:1031-1049` 另算：

`R_env = -(0.08 user_cost + 10 queue_cost + 0.03 grid_cost + 0.01 fluct_cost + 10*N_voltage_violation)`。

其中 `grid_cost` 含逐步站点购电费和 `20*line_loss`。当前 hindsight 入口在 `trainer/trainer.py:127` 用 `_` 丢弃它，不能写进论文训练目标。

## 11. 状态空间、动作空间和掩码

当前完整节点状态 `x in R^(N×19)`；EMA 图 N=74。

| idx | 变量/物理意义 | 单位/归一化 | 默认 station_only 是否读取 |
|---:|---|---|---|
| 0 | 当前节点 EV 数 | 辆，未归一化 | 否 |
| 1 | 是否站点 | 0/1 | 否 |
| 2 | queue 长度；个体状态再加 pending | 辆，未归一化 | 否 |
| 3 | `current_price` | 货币/kWh，未归一化 | 是 |
| 4 | connected EV 数 | 辆 | 否 |
| 5 | `last_total_load/max_grid_power` | 比例 | 是 |
| 6 | 上次 runpp 母线电压 | pu | 否 |
| 7 | ToU multiplier | 0.5/1/1.5，全节点 | 否 |
| 8 | 查询 EV SOC | SOC/100，只写当前 EV 节点 | 是（作为 EV 特征） |
| 9 | `1/(1+trip_time_h)` | 变换值 | 否 |
| 10 | 预计行驶时间 | h，未归一化 | 是 |
| 11 | 预计排队+充电服务时间 | h，未归一化 | 是 |
| 12 | generalized cost/100 | 混合货币成本 | 否 |
| 13 | 全局价格噪声 | [-0.1,0.1] | 否 |
| 14 | 到站数 EMA | 辆/步的平滑值 | 否 |
| 15 | 预计等待/4h | clip 到 2 | 是 |
| 16 | `(queue+connected+EMA+pending)/(20+chargers)` | 比例，可>1 | 否 |
| 17 | 空闲桩比例 | `[0,1]` | 是 |
| 18 | `(historical heading+pending)/total heading` | 跨站占比 | 是 |

构造代码：`env/base_env.py:595-661`。边特征是 `[length_km, speed_kph/100]`，但默认网络完全不用 `edge_index/edge_attr`（`env/base_env.py:469-481`；`agents/network_station_only.py:113-167`）。

动作空间为站点索引 `{0,...,S-1}`，当前 S=4。

动作掩码检查图可达性、SOC 是否足够到站并留 2 个百分点余量、预计时间是否<24h；可选队列超时 mask 默认关闭；全被屏蔽时重新全部放开（`env/base_env.py:791-847`）。但论文编排脚本显式传 `--no-use-action-mask`（`run_fl_methods.py:51-58`；`run_paradigm_training.py:101-105`），故主要论文实验无非法动作屏蔽。

## 12. 站点集合 Q 网络与 Double DQN

### 12.1 默认 StationOnly 网络

- 站点输入 7 维 `[3,5,10,11,15,17,18]`；EV 输入 1 维 `[8]`（`agents/network_station_only.py:20-30`）。
- 站点 MLP：7→32→32，ReLU；EV：1→8，ReLU；融合：40→32，LayerNorm+ReLU（`:33-57`）。
- 4 个站点嵌入做 mean pooling 得 32 维全局上下文（`:134-149`）。
- 每站 head：[自身32||全局32]→32→1，ReLU（`:93-98,151-159`）。
- 无 GNN 消息传递、无道路边使用。称“站点集合 Q 网络/DeepSets 式均值池化网络”比“图 Q 网络”准确。

参数量（当前 4 站不影响共享权重）：4817 个可训练参数，其中 `encoder.*` 2704（56.13%），`head.*` 2113（43.87%）。

### 12.2 Double DQN 与训练超参数

一般代码目标：

`a* = argmax_a Q_online(s',a)`；

`y = r + 0.99*(1-done)*Q_target(s',a*)`；

`L = MSE(Q_online(s,a),y)`。

证据：`agents/dqn_base.py:168-188`。online/target 初始化同步，target 每 100 次梯度更新硬同步（`:73-92,199-203`）。Adam，lr=3e-4；梯度范数裁剪 10（`:42,82,190-197`）。Hindsight replay 容量 50000（`agents/hindsight_dqn_agent.py:18-32`）；论文脚本 batch=64。

但 `HindsightTrainer.on_charge_started()` 对全部样本写 `done=True,next_state=None`（`trainer/trainer.py:86-93`），因此实际 `y=r`，gamma、next-state、target Q 均被乘零。Double DQN 名称对应代码框架，不对应当前有效训练机制。

ε：agent 初始 1.0；单地每 episode 乘 0.994 至最小 0.05（`agents/dqn_base.py:87-92,130-132`）。`run_fl_methods.py` 从 warm-start 后覆盖为 0.3，乘 0.985；联邦默认每个 local episode 衰减，2 local episodes/round 即每轮约乘两次（`run_fl_methods.py:51-58`；`train_federated_hindsight.py:304-333,418-422`）。

## 13. FedAvg、FedProx、FedRep

当前论文联邦实现位于 `train_federated_hindsight.py`，不是旧的 `agents/FederatedDQN.py`。

### 13.1 FedAvg

- 仅聚合 `policy_net.state_dict()` 的浮点项；`station_node_ids` 保留全局值（`train_federated_hindsight.py:57-78,128-155`）。
- 权重 `n_k=max(1,num_samples_this_round)`，其中样本数是当轮开始充电并写入 replay 的 transition 数，不是 replay 总量或梯度步数（`:43-55,128-146`）。
- 客户端 target 在每轮下发时由本地 policy 全量同步；服务器 target 在聚合后同步（`:60-79,154-155`）。target 不单独聚合。
- optimizer state 不上传、不聚合、不重置；各客户端在全局权重覆盖后保留旧 Adam moments。
- replay buffer 始终本地保留并跨轮复用。
- 默认 50 轮、每轮每客户端 2×144 环境步；`run_fl_methods.py` 正式对比为 120 轮（`train_federated_hindsight.py:173-178`；`run_fl_methods.py:31-40,51-58`）。

### 13.2 FedProx

每个 batch 反向后给 policy 全部命名参数增加：

`grad += mu*(w-w_global)`，等价于目标加 `(mu/2)||w-w_global||²`（`train_federated_hindsight.py:80-95`）。

锚点在每轮下发后快照（`:400-405`）。target 不受近端项直接优化。CLI 默认 mu=0.01；`run_fl_methods.py` 正式对比实际传 mu=0.1（`train_federated_hindsight.py:212-218`；`run_fl_methods.py:35-40,61-62`）。

### 13.3 当前 FedRep 的真实含义

- 服务端只把 key 前缀为 `encoder.` 的参数视为共享参数；`head.*` 本地保留（`train_federated_hindsight.py:114-126`）。
- 本地每个 batch 同时更新 encoder 和 head，没有先 head 后 representation 的两阶段训练，也没有冻结阶段。
- head 跨轮保留；replay/optimizer 本地保留。
- 对 `station_only`，共享 2704/4817=56.13% 参数，单向传输浮点参数量相对全模型减少约 43.87%。
- 对 `station_attn`，没有任何参数名以 `encoder.` 开头，当前 `_is_shared()` 会共享 0 个参数；这是架构组合 bug。
- 最后一轮聚合后没有再次向客户端分发最终共享 encoder，就直接保存每个客户端模型（`train_federated_hindsight.py:417,437-450`）。因此三个 `*_final.pth` 的 encoder 是最后一轮各自本地训练后的版本，而不是同一个最终聚合 encoder。

结论：不是严格 FedRep，应称“FedRep 式部分参数共享”或“共享编码器与本地决策头的个性化联邦训练”。

## 14. 同一时隙提交计数前瞻

待决策 EV 按 SOC 升序，不随机打乱；SOC 相同保持 EV 列表/ID 顺序（`env/base_env.py:223-226`）。这使低 SOC EV 永远优先，存在决策顺序公平性问题。

对时隙内第 j 辆车：

`z_s^(j) = sum_{k<j} 1(a_k=s)`。

`pending_counts[action] += 1` 与该式一致（`trainer/trainer.py:117-125`；评估路径 `evaluate_decision_ablation.py:127-140`）。但 z 不只作为一个原始特征：

- 加到 idx2 queue count；
- 进入预计等待函数，从而改变 idx11/15；
- 进入 idx16 pressure；
- 与历史 `evs_heading_to` 相加后除以全站总 heading，写 idx18。

证据：`env/base_env.py:627-660`。z 没按充电桩数归一化；idx18 按全站 heading 总量归一，idx16 按 `max_queue_len+num_chargers` 归一。

前序动作不会在决策循环内真实更新 queue、connected、load 或 price；只通过上述合成状态量影响后续车辆。`evs_heading_to` 要到 `env.step` 实际派发才更新（`env/base_env.py:879-918`）。训练和常规模型评估均启用；决策消融 B/A 传 `None` 删除前瞻。

可见旧稿所称“只把 submitted count 注入 price（idx2）”不准确：idx2 是 queue 数，不是 price；真正被默认网络读取的前瞻通道是 idx11、15、18。

## 15. 代码与论文一致性

### A. 与可见 `docs/paper_draft.md` 一致

- 40 EV、4 站、每站 8 桩是主要编排规模（旧稿 `:103-106,268-275`）。
- 19 维完整状态以及默认网络实际读取 `[3,5,10,11,15,17,18]`、EV `[8]`（旧稿 `:116-138`）。
- 默认训练奖励三项和默认/异构权重数值（旧稿 `:147-173,229-244`）。
- 同时隙按顺序累积 `pending_counts` 的总体思想（旧稿 `:175-211`）。
- 主要论文评估显式无 action mask（旧稿 `:271-275`）。

### B. 论文表述过强或错误

1. “UE/Frank–Wolfe 背景流、城市 UE 强度 1.3/1.0/0.7”：当前模块不可导入，回退启发式；回退时 scale 被忽略。旧稿 `:215-218` 不成立。
2. “station-restricted GNN”：默认网络无 message passing、无 edge 使用，应改“站点集合 Q 网络”。旧稿 `:29-33,116-138` 表述过强。
3. “每车 MDP + Double DQN”：所有样本 terminal，TD 目标退化为 reward，应改“延迟反馈的上下文 bandit/终端 Q 回归”。旧稿 `:108-114,140-145`。
4. “充电完成后回填、realized energy fee”：实际充电开始即回填，费用是开始价快照；`on_completed()` 空实现。旧稿 `:149-154` 错误。
5. “动作掩码移除不可行站点”：机制存在，但主要实验脚本禁用。旧稿 `:111-114` 需注明实验未使用。
6. “前瞻注入 price(index 2)”：idx2 是 queue；默认网络实际通过 service/wait/heading 通道接收。旧稿 `:190-196` 错误。
7. “FedAvg 在 `agents/FederatedDQN.py:FederatedServer`”：当前论文路径使用 `train_federated_hindsight.py:FedAvgServer`。旧稿 `:221-227` 路径错误。
8. “traffic–grid coupling simultaneously affects grid loading and policy”：物理负荷与 AC 潮流存在，但默认策略不读 AC voltage/loss，hindsight reward无电网项。旧稿 `:97-105` 应收窄。
9. “实时 LMP”：代码是每 6 步尝试 OPF、失败静默回退；应写条件式 OPF 节点有功边际价格，不能无条件写 DLMP。
10. 归一化常数来源未给出，论文不得声称有数据统计依据。

### C. 代码已实现但旧稿未充分写入

- 每步 AC `runpp`、IEEE33 构网参数、EV kW→MW 和 q=0 映射；
- OPF 条件价格与 ToU fallback 的精确分支；
- 分段近似 CC-CV、站级 600 kW 比例限幅；
- FCFS 离散队列及 4 h 超时放弃；
- BPR 的非标准 `capacity*t0` 并发容量换算；
- FedProx 和 FedRep-style 的当前真实实现及通信参数量；
- 训练放弃样本丢弃、最终 FedRep encoder 未再分发等限制。

## 16. 潜在 bug、未确认项与复现实验风险

### 必须优先处理

1. **UE 不可复现**：当前 `env.background.ue_assignment` 不可导入，所有声称 UE 的当前运行会静默回退；同时 fallback 忽略 `ue_scale`。
2. **ToU 时标错误**：144 steps/day 却把 step index 当 hour。
3. **跨多边行驶能耗错误**：`EV.move()` 未累计本步全部边距离。
4. **奖励时点/费用语义错误**：代码和注释/旧稿均说 completion，实际 charge start；费用不是 actual accumulated fee。
5. **FedRep 最终模型不含最终聚合 encoder**：最后 aggregate 后未 redistribute。
6. **随机种子不足**：训练脚本的 `--seed` 主要传给路网加载，未统一 `random.seed/np.random.seed/torch.manual_seed`；模型初始化、EV/价格随机性不能靠 CLI seed 完整复现。

### 重要模型限制

- 排队估计在“桩全忙但 queue=0”时返回 0；
- 放弃 transition 直接丢弃，训练奖励存在幸存者偏差；
- AC 电压和线损不进默认 policy/reward；无 grid action mask/安全控制；
- `runopp` 失败无日志，无法从单次结果审计 fallback 比例；
- `original`/`lightweight` 网络构造函数不接受 `DQNBase` 传入的 `use_action_mask`，选择这些 variant 会 `TypeError`；
- `fedrep + station_attn` 共享参数为 0；
- heterogeneous 训练日志在 `train_federated_hindsight.py:313-321` 用默认权重重算 reward，日志值与实际存入 buffer 的异构 reward 不一致；
- `evaluate_decision_ablation.py:144-151` 同样固定默认权重，而 `eval_paradigms` 使用的 `evaluate_oracle_wait.run_mode` 才调用 `weights_for`；需明确每张表来自哪个评估入口；
- `runpp` 不收敛无降级处理；
- 线路 loading_percent 虽由 pandapower 产生但未读取，变压器不存在；不得写过载约束。

### 未确认

- 现有历史 checkpoint/JSON 是否在 `ue_assignment.py` 尚存在时生成：当前代码和日志不足以确认，必须用当时 commit/运行日志校验。
- `paper_draft_cn_v3.md` 的逐句状态：文件缺失。
- 0.58、0.17、65 的统计来源。
- 每次 `runopp` 的成功率和各站 `lam_p` 时序；代码未记录。

## 17. 最终摘要

1. **pandapower 是否真正执行 AC 潮流**：是。每环境步在站点充电后调用 `runpp(bfsw)`；另有初始化 Thevenin 探测潮流。
2. **潮流结果是否影响策略**：AC `runpp` 的 vm_pu/line loss 不影响默认 `station_only` 策略或 hindsight reward；另一次 `runopp` 成功时，其 `lam_p` 通过价格影响策略和费用。
3. **UE 是否真实存在**：当前可复现执行中否；实际是启发式背景边流。历史分支曾有 FW 模块，现有历史实验是否用了它未确认。
4. **BPR 是否真实使用**：是，路径预计和实际穿边均用，但分母为 `capacity*t0`。
5. **CC-CV 是否真实实现**：实现了分段离散近似（CC 常功率 + CV 线性 taper），不是完整连续电化学模型。
6. **电价来源**：每 6 步 OPF `lam_p`（成功时）或 ToU×随机噪声（失败时），再加服务费和队列/占桩加价；奖励费用只用 `current_lmp` 开始快照。
7. **奖励**：实际行驶时间、实际离散等待、开始充电时预计购电费用三项；无 grid reward，无 abandon penalty。
8. **FedRep 是否严格**：否，是 encoder 聚合/head 本地保留的 FedRep-inspired 版本。
9. **最严重三项不一致**：UE/城市交通异构不可复现；“完成后实现费用奖励”实际为充电开始估计费；“Double DQN/MDP”实际 terminal bandit 回归。紧随其后是 AC 潮流不进默认策略/奖励。
10. **可直接写入公式**：当前 BPR 变体、固定里程能耗意图式（注明实现近似）、离散 FCFS、分段 CC-CV、EV 负荷映射和 AC 潮流调用、条件式 OPF价格、三项 hindsight reward、站点集合网络、FedAvg/FedProx、encoder/head 部分共享、同时隙 submitted count。
11. **不能写入公式**：当前 UE/FW 均衡方程、无条件 DLMP、逐时实际费用作为训练奖励、配电安全约束/电网奖励、严格 FedRep、有效多步 DDQN、按真实道路地理的交通模型。
12. **是否重跑实验**：若论文保留 UE/交通异构、ToU、车辆能耗、FedRep 或可复现随机种子结论，则修复并加日志后必须重跑；仅把论文收窄为当前启发式交通 + charge-start reward + partial-sharing 实现时，也至少应复核历史 checkpoint 的代码版本。
