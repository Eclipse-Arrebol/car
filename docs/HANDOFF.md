# 交接文档 · EV 充电导航项目测试驱动阶段

> **写给下一个 Claude session**  
> **最后更新：2026-05-24**（累计：EMA L1/L2/**L3**、`network` 冒烟、**EMA graphml 限速 m/h 修复**、**`osm_loader.load_road_network_from_file` 读 graphml 后强制将 `x,y,length,speed_kph,capacity,weight` 转为 float**（兼容 TNTP 导出时属性为字符串，避免 SciPy 邻接矩阵 dtype 报错）、**`map_tools/tntp_net_to_graphml.py` + `env/tntp_loader.py`（`parse_tntp_net` / `tntp_net_to_graphml_graph`）从 `EMA_net.tntp` 重生成与 TNTP 节点编号对齐的 `ema.graphml`**、**`tools/draw_ema_network.py`**：无参画 TNTP 拓扑；`--congestion` 用 `RealTrafficEnv` 快照，**默认边上色 `x_flow`**；**拥堵模式默认 UE 背景基线**（`EMA_net.tntp` + `EMA_trips.tntp` → `env/background/ue_assignment.py` Frank–Wolfe；`--no-ue-background` / `--trips-tntp` / `--ue-max-iter`）；出图 **Nature 风** 双出 **PDF + 600dpi PNG**、拥堵支路 **线宽/透明度** 与 **magma/coolwarm** 色条等）、**`env/background_traffic.py`**：`build_base_background_flows` 可选 TNTP 走 UE；`setup_background_traffic_and_respawn_nodes` 读 `env.background_ue_*`；`RealTrafficEnv(background_ue_net_tntp=...)`；启发式基线 + `BACKGROUND_EDGE_BASE_SCALE` 仍保留）、**环境侧移除 t2 到站决策**、`charging_station` **充满电 snapshot + 会话累计重置 + 路上状态清理**、**BPR `ratio` 单位修复**（`c·t0_h` 分母）、`**trainer/` `HindsightTrainer` + `test_trainer_core`**、`pytest` 专项测、Level2 诊断打印、**`train_hindsight.py` / `tools/debug_station_bias.py` 默认 UE 背景**（`--no-ue-background` / `--ue-*`；`eval_hindsight.py` 仍为启发式）、根目录 `train_hindsight.py` 最小可跑骨架 + 3 episode 冒烟通过、`**agents/network_station_only.py` + `network_variant`/`use_action_mask` CLI 切换**、**200 epoch 评估明显优于 50 epoch，偏站问题已靠真实 `station_node_ids` 对齐与站点决策诊断脚本定位**、**车辆重生：`EV.reset_for_respawn` + 合法节点 `_choose_respawn_node` + 充满后 SOC 10%–20%**、`**tests/test_respawn_logic.py` + `tests/test_trainer_integration.py` 已通过**、**背景日周期 + 基线进 BPR**：`build_daily_profile` + `update_background_traffic` + `_dynamic_profiles`（`TrafficPowerEnv` / `RealTrafficEnv`）**、`**test_background_traffic.py` / `tools/debug_background_traffic.py` 接路网站点时默认 `RealTrafficEnv(offline=True)`**）  
> **本轮补充（文档同步）**：训练默认 **80 EV / 4 站 / 每站 8 桩**（`train_hindsight` 模块常量 + CLI）；`HindsightDQNAgent` 使用 **`num_actions=num_stations`**、**`num_nodes_per_graph=env.num_nodes`**；`ChargingStation` 默认 **单桩请求上限 60 kW**、**站总功率上限 600 kW**；**UE Frank–Wolfe 迭代日志默认关闭**（`compute_ue_background_flows(..., verbose=False)`；`train_hindsight` / `debug_station_bias` 用 **`--ue-verbose`** 打开；`RealTrafficEnv(background_ue_verbose=...)`；`draw_ema_network --congestion` 在 **`--verbose-background`** 时打开 UE 详细日志）；`TrafficPowerEnv` 支持 **`num_chargers_per_station`**；**`tests/test_hindsight_train_scale.py`**；**`tools/compare_travel_time_background.py`**（`--top-edges`）；**`debug_station_bias`** 与训练默认 **站数/车数/桩数/agent 形状** 对齐（旧 2 站 checkpoint 需显式 `--num-stations 2` 等）。  
> 接手时请读 **第二节进度** 与 **第十节测试资产**；纪律与协议仍以 **第四节、第三节** 为准。

---

## 一、用户当前的决策（已锁定，不要再讨论）

用户名 **Eclipse**，华南理工 EI 一年级硕士，做 FedGRL 充电导航研究。

经历了 5 轮 Codex 失控后，用户已经做出以下决定，**请直接接受，不要再提"要不要重构"**：

1. **暂不重构项目代码**
2. **改走测试驱动路线**：给每个文件写单元测试，在写测试过程中发现并修复 bug
3. **单测起点已从 `env/entities.py` 扩展到整条 env 链与部分 agents**（见第十节）
4. **暂不考虑新创新点**（reroute / 多步 MDP 等想法已经否决，作为未来 v2）

---

## 二、当前进度（精确到状态）


| 项                                                                                                                                                         | 状态                                                                                                                                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 重构计划                                                                                                                                                      | ❌ 已废弃（不要复活）                                                                                                                                                                        |
| t2 多步 MDP 方案                                                                                                                                              | ❌ 已废弃（不要复活）                                                                                                                                                                        |
| 单步 hindsight contextual bandit                                                                                                                            | ⏸️ 暂缓（等测试完成后再说）                                                                                                                                                                    |
| `tests/test_entities.py`（EV + mock 路网）                                                                                                                    | ✅ 已落地（含 `remaining_edge_time_h`）                                                                                                                                                   |
| `tests/test_charging_station.py`                                                                                                                          | ✅ 已落地                                                                                                                                                                              |
| `tests/test_power_grid_pp.py`（`compute_thevenin=False`）                                                                                                   | ✅ 已落地                                                                                                                                                                              |
| `tests/test_base_env.py`（决策/BPR/解析等可隔离逻辑）                                                                                                                 | ✅ 已落地                                                                                                                                                                              |
| `tests/test_real_env.py`（offline 合成路网）                                                                                                                    | ✅ 已落地                                                                                                                                                                              |
| `tests/test_real_env_ema_graph_integration.py`（层级 1：T1.1–T1.4）                                                                                            | ✅ 已落地，路网路径 `map_outputs/ema/ema.graphml`                                                                                                                                           |
| `tests/test_real_env_ema_level2_physics.py`（层级 2：T2.1–T2.7）                                                                                               | ✅ 已落地；用例末 `**_diag_print`** 打关键统计（T2.7 abandon 上界已随「无 T2_PENDING」放宽）                                                                                                               |
| `tests/test_real_env_ema_level3_performance.py`（层级 3：T3.1–T3.3 耗时 / cProfile / 内存）                                                                        | ✅ 已落地；**T3.3** 依赖 **psutil**，未装则 skip                                                                                                                                              |
| `tests/test_snapshot_bug.py`（充满后 **会话** 累计字段 + snapshot 语义）                                                                                               | ✅ 已落地（`pytest`）                                                                                                                                                                    |
| `tests/test_respawn_state_cleanup.py`（充满后 **path/边/目标站** 等路上状态清空）                                                                                         | ✅ 已落地（`pytest`）                                                                                                                                                                    |
| `tests/test_respawn_logic.py`（充满 → snapshot → 清零 → **合法节点重生** + 低 SOC 10%–20%、多轮重生、`env.step` 可继续）                                                        | ✅ 已落地（`pytest`）；单测里对 `ChargingStation` 显式设 `legal_respawn_nodes` 以钉死契约                                                                                                             |
| `tests/test_trainer_integration.py`（真 `DQNAgent` + `TrafficPowerEnv` + `HindsightTrainer` 短跑 smoke，防接口漂移）                                                 | ✅ 已落地（`pytest`）；**`TrafficPowerEnv` 支持 `num_chargers_per_station`**；集成用 **每站 8 桩**、**200 step** 内 **`replay(4)`** smoke（高功率下 buffer 增长快于旧 8 样本门槛）                                                                                                                                                                    |
| `tests/test_network.py`（FeatureEncoder / GraphQNetwork 冒烟）                                                                                                | ✅ 已落地                                                                                                                                                                              |
| `tests/test_bpr_congestion.py`（BPR 对 active flow / 单调 / 单位手算）                                                                                             | ✅ 已落地                                                                                                                                                                              |
| **`map_tools/tntp_net_to_graphml.py`** / **`env/tntp_loader.py`**（TNTP → GraphML，节点 `0..N-1` 与 TNTP 交通节点 `+ first_thru` 对齐）                                                                 | ✅ 已落地；重生成后须删 **`map_outputs/ema_cache/local_ema*.pkl`** 并核对 **`config/stations.json`** 站点索引                                                                                                                                   |
| **`tools/draw_ema_network.py`**（TNTP 拓扑；`--congestion` 下 `RealTrafficEnv` 快照；**默认 UE 背景** `EMA_net`+`EMA_trips`；Nature 风 **PDF+600dpi PNG**；拥堵 **`x_flow`/`bpr`** 与线宽–透明度编码） | ✅ 已落地                                                                                                                                                                              |
| **`env/background/ue_assignment.py`**（Frank–Wolfe UE；`compute_ue_background_flows`；`__main__` 可自检 EMA）                                                                                         | ✅ 已落地                                                                                                                                                                              |
| `**trainer/trainer.py`**（`HindsightTrainer`：`pending`、`on_dispatch` / `on_completed` / `on_abandoned`、`step_episode`；与 `**training/trainer.py`** 并存，包名不同） | ✅ 已落地（`tests/test_trainer_core.py`）                                                                                                                                                |
| `**train_federated_hindsight.py` + `trainer/federated_hindsight_trainer.py`**                                                                                                                         | ✅ 已落地；联邦 hindsight 训练入口与 FedAvg trainer。支持 `old_city` / `new_city` / `suburb` 三客户端，`--serial` / 并行模式，`--use-action-mask` / `--no-use-action-mask`，`--rounds`、`--steps-per-episode`、`--batch-size`、`--save-every`；默认 checkpoint 目录 `checkpoints_federated_hindsight/`。当前重点已从“串行客户端训练”推进到“并行/常驻 worker 训练 + replay buffer 跨轮缓存”，但评估侧仍在处理 `pandapower` / Thevenin 的兼容性问题。 |
| **根目录 `train_hindsight.py`**                                                                                                                              | ✅ 已落地；`HindsightTrainer` 训练入口；**默认 80 EV / 4 站 / 每站 8 桩**（`--num-evs` / `--num-stations` / `--num-chargers-per-station`）；`--network` / `--use-action-mask`；**默认 UE**（`--no-ue-background` / `--ue-*` / **`--ue-verbose`** 打印 FW）；**`num_nodes_per_graph=env.num_nodes`**；**每个 `reset` 再跑 FW**（慢） |
| **`tools/debug_station_bias.py`**（偏站诊断：random / shortest_path / model_greedy）                                                                                         | ✅ 已落地；**与 `train_hindsight` 默认 UE + 默认规模对齐**（`TRAIN_DEFAULT_*` + `--num-stations` / `--num-chargers-per-station`；agent **`num_actions`/`num_nodes_per_graph`** 与当前 `RealTrafficEnv` 一致；**`--ue-verbose`**） |
| **`tests/test_hindsight_train_scale.py`**                                                                                                                   | ✅ 已落地；训练默认规模 + `RealTrafficEnv` 离线路网 smoke |
| **`tools/compare_travel_time_background.py`**                                                                                                               | ✅ 已落地；有/无背景下边级 BPR 与 OD 对比；**`--top-edges K`**；可选 **`--ue-net`/`--ue-trips`**（FW 默认静默，与全局 UE verbose 策略一致） |
| `**agents/dqn_base.py**`                                                                                                                                  | ⬜ **下一个优先单测文件**；已支持按 `network_variant` 选择网络与全局 mask 开关                                                                                                                             |
| `agents/FederatedDQN.py`                                                                                                                                  | ⬜ 待定                                                                                                                                                                               |
| `training/trainer.py`                                                                                                                                     | ⬜ 待定                                                                                                                                                                               |
| git 备份                                                                                                                                                    | ⚠️ 仍建议用户每步 `git commit`（历史状态未强制核验）                                                                                                                                                 |


---

## 三、接手协议（看完这一节就能开工）

### 若继续「按文件写单测」

1. *EV 上 `t0_` / `abandon_`* 等多步遗留字段：写测时不必当核心契约；`**t2_*` 已从 `entities.EV` 删除**，环境已无 `**T2_PENDING` / `apply_t2_action`**（见第五节）。
2. **单文件 4–7 个用例**为宜；集成/EMA 层级可单独文件、单独命名（`test_real_env_ema_`*）。
3. **测试 FAILED**：先贴 traceback，区分「断言/场景写错」与「实现 bug」；**禁止**为绿而盲改断言或盲改实现（纪律见第四节）。

### 若用户说「继续测试」

下一文件顺序（与早前规划一致，已从 `entities` 推进到 `network` 之后）：

```
… 已完成：entities → charging_station → power_grid_pp → base_env（含 **BPR 拥堵比修复**）→ real_env(offline) → EMA L1/L2/**L3** → agents/network.py → **charging_station 充满语义 pytest** → **`trainer/` HindsightTrainer + `test_trainer_core`**
→ 当前：agents/dqn_base.py
→ 其后：agents/FederatedDQN.py → **接线**：`training/trainer.py` 是否复用 `trainer.HindsightTrainer`（Step 3，待定）
```

---

## 四、必须遵守的工作纪律（这是 5 轮失控的教训）

### 纪律 1：Cursor 一次只动一个文件（单测文件可新增，但实现 bug 先记 TODO 再扩 scope）

若写测时发现**非当前目标文件**的明显 bug，**记下来**，可让用户记在 `TESTING_TODO.md`，**不主动大改**。

### 纪律 2：测试 FAILED 时，不许为绿而糊弄

让 Cursor **停下来**报告 traceback。严禁：改实现糊弄通过、改断言糊弄通过、「顺手」大重构。

### 纪律 3：每天 commit 一次

用户完成一步后提醒：`git commit`。

### 纪律 4：不要替用户做决定

测试深度、bug 是否当场修，**问用户**。

### 纪律 5：不要碰废弃方案

用户提到 t2 / 多步 MDP / reroute：**简短回应即可，不展开、不复盘**。

---

## 五、已知的代码现状（事实摘要）

### `env/entities.py` 的 EV 类

**已从类中删除**：原 `t2_step`, `t2_decision_pending`, `t2_action`, `t2_state`, `t2_pending_steps`（环境不再做 t2 到站决策）。

**仍存在的多步 / 遗留字段**（写测不必当核心契约）：`t0_state`, `t0_action`, `t0_step`, `travel_time_at_dispatch`, `abandon_reason`, `just_abandoned_this_step` 等。

**核心活字段**（统计）：`travel_time_h`, `wait_time_h`, `total_fee_paid`, `total_energy_charged`, `charge_sessions` 等。注意 `**travel_steps` / `wait_steps` / `charge_steps`** 在 `base_env.step` 里**按状态每步 +1**，**充满电时未重置**（全寿命计数语义）；是否会话化待单独任务。

### `env/charging_station.py`（功率默认与充满分支）

- **功率默认（类 `__init__`）**：**`max_charger_power=60` kW**（CC 段单车请求上限）、**`max_grid_power=600` kW**（站级总功率上限，`optimize_power` 等比例降额）；单测可显式传入其它值。
- **充满电**（`ev.soc >= 95.0`，字面常量）：
  1. **先**把当前会话累计写入 EV 上的 snapshot 字段（如 `charge_fee_snapshot`；队列/行程时间 snapshot 名以代码为准），再将会话内累计 `travel_time_h` / `wait_time_h` / `total_fee_paid` / `total_energy_charged` 置 0。
  2. **再**清空路上状态：`path`、`last_traversed_nodes`、`current_edge_*`、`remaining_edge_time_h`、`target_station_idx`、`assigned_station` 等。
  3. 然后 `status = "IDLE"`、`charge_sessions += 1`，并重置 `low_soc_triggered` / `charge_decision_pending` / `remaining_replans` 等。
  4. 若 `respawn_after_full_charge`：**不再**用「当前节点 + 固定偏移」采样（会越界）；改为 `ev.reset_for_respawn(self._choose_respawn_node(ev), random.uniform(10.0, 20.0))`。`_choose_respawn_node` 从 `station.legal_respawn_nodes` 中随机选 **≠ 当前节点** 的节点；列表为空则退回 `ev.curr_node`（单测须显式注入合法集合）。

### `env/entities.py` · `EV.reset_for_respawn(new_node, new_soc)`

重生专用入口：更新 `curr_node` / `soc`，清空路径与边状态、目标站、`low_soc_triggered` / `charge_decision_pending`，`status="IDLE"`。复杂环境决策仍放在 env / 桩侧，不在此方法内。

### `env/base_env.py`（图状态、合法重生节点、背景流、EMA graphml 限速）

- **合法重生节点**：`self.legal_respawn_nodes = list(self.traffic_graph.nodes())`，`__init__` 里同步到各 `station.legal_respawn_nodes`（避免 `curr_node+1` 类采样越界导致 `get_graph_state` 索引错误）。
- **`TrafficPowerEnv(..., num_chargers_per_station=...)`**：玩具网格上每站并行桩数；与 `train_hindsight` 默认 **8** 及 `test_trainer_integration` 对齐，用于缩短高功率下「充满 → buffer」路径的仿真步数。
- **注意**：`TrafficPowerEnv.reset()` 在重建 `stations` 之后会调用 `**setup_background_traffic_and_respawn_nodes(self)`**，同步 `legal_respawn_nodes` 与背景流基线。`RealTrafficEnv` 不调用基类 `__init__`，但在自身 `**__init__` / `reset**` 中同样调用该函数（定义于 `base_env.py`），与 BPR + 背景流路径一致。
- **背景交通流**：`env/background_traffic.py` 中 `build_daily_profile(144)`（早晚 **高斯双峰** 归一化曲线，**未改**）。**边级空间基线**二选一：（1）**默认/小网格**：`edge_base_background_flow` + `BACKGROUND_EDGE_BASE_SCALE` 的启发式；（2）**TNTP UE**：`build_base_background_flows(..., net_tntp_path=..., trips_tntp_path=..., ue_verbose=...)` 内部调 **`env/background/ue_assignment.compute_ue_background_flows`**（Frank–Wolfe；**`verbose=False` 默认不打印迭代**；`__main__` 自检传 `verbose=True`），再按 `traffic_graph.edges()` 对齐 `(u,v)`/`(v,u)`；或直接用 **`build_base_background_flows_ue`**。**`setup_background_traffic_and_respawn_nodes`**（`base_env.py`）会读取 `env` 上可选属性 **`background_ue_net_tntp` / `background_ue_trips_tntp`**（及 `background_ue_scale`、`background_ue_max_iter`、`background_ue_tol`、**`background_ue_verbose`**）；**`RealTrafficEnv`** 构造参数同名传入即可。**UE 只替换 `background_edge_base_flows` 数值**，不改 graphml 拓扑/坐标/容量属性。`update_background_traffic()` 仍每步把基线 × 日 profile × 小幅正弦写入 `background_edge_flows`；`_dynamic_profiles` 中 `x_flow = _edge_flow + _background_flow + add_vehicle` 后走 `_bpr_time_h`。**注意**：`RealTrafficEnv` 在 **`__init__` 与 `reset`** 各调一次 `setup_background_traffic`，传 UE 时会 **跑两遍 FW**（耗时）。**`train_hindsight.py` / `tools/debug_station_bias.py`**：`--ue-verbose` 打开 FW 日志；**`draw_ema_network --congestion`**：`--verbose-background` 时 **`background_ue_verbose=True`**；**`tools/eval_hindsight.py`** 若未改则仍为启发式，与训练/偏站诊断对比时注意。
- `**TrafficPowerEnv._bpr_time_h`**：拥挤度 `**ratio = x_flow / max(1, c_capacity * t0_h)`**（`x_flow` 为边上车辆数量级，`c_capacity` 为辆/小时，与 `t0_h` 相乘得「边自由流行程内可服务车辆数」量级）；专项测 `**tests/test_bpr_congestion.py**`。旧式 `c * step_duration_h` 分母已废弃。
- `**TrafficPowerEnv._parse_speed_kph**`：若数值 `**>= 1000**`，按 **m/h→km/h** 除以 1000（修复 graphml 把米/小时误标为 `speed_kph` 导致 **亚秒级穿边**）。单测见 `tests/test_base_env.py` 中 `test_parse_speed_graphml_m_per_h_mislabeled_as_kph`。
- **`env/osm_loader.py`**：`load_road_network_from_file` 在 **`.graphml` 分支**取得 `G_raw` 之后调用 **`_coerce_graphml_numeric_attrs`**，将节点 **`x,y`** 与边 **`length,speed_kph,capacity,weight`** 转为 **`float`**（TNTP 管线写 graphml 时常保留字符串以满足 OSMnx 读盘；转 float 后 **`nx.to_scipy_sparse_array`** 与 BPR 数值一致）。
- **环境侧已无 t2 决策**：到站即 `**_commit_arrival_to_waiting`**（入队 `WAITING`）；已删 `**apply_t2_action`**、`**T2_PENDING**` 及超时 auto-accept 日志；`**get_pending_decisions**` 的 `**t2_arrival` 恒为 `[]**`；`**info["pending_t2"]**` 仍为「本步到站入队 EV」列表（telemetry），**非**待决策队列。
- `**training/trainer.py`**：`pending_t2` 环已删；`T0` 的 `backfill_snext` 在 `**env.step` 之后**用 `get_graph_state_for_ev` 立即填上。
- `**agents/dqn_base.py`**：混合 `step_type` 时 **T0 的 `next_type`** 从 `**'t2'**` 改为 `**'t0'**`（新轨迹无 T2）；回放缓冲里旧 **T2** 样本仍可被旧分支吃到。
- `real_env`：不调用 `TrafficPowerEnv.__init__()`，自成初始化（已知隐患，HANDOFF 不要求在此修）。
- `**agents/network.py`**：仍含 `**t2_advantage_fc`** 与 `**action_type='t2'**`（单测 `test_network` 仍覆盖）；与「环境无 t2」可并存，属 **RL 网络遗留**。
- `**agents/network_light.py`**：轻量版网络，已按 `triptime / queuetime / fee` 收缩输入，`price` 仅保留 `lmp` 对应的 `price(3)`。
- `**agents/network_station_only.py`**：站点节点专用网络，默认仅看站点节点 + EV SOC；`use_action_mask` 可在网络层开关；默认不再使用错误的 `[0,8]` 站点索引，而由训练/评估入口注入真实 `station_node_ids`。
- `**train_hindsight.py`**：EMA 训练入口；`--network` / `--use-action-mask`；**默认 80/4/8**；**默认 UE 背景**（与 `draw_ema_network --congestion`、`debug_station_bias` 对齐：`--no-ue-background`、`--ue-net-tntp`、`--ue-trips-tntp`、`--ue-max-iter`、`--ue-tol`、`--ue-scale`、**`--ue-verbose`**）；每个 episode **`reset` 再跑 FW**（耗时）。
- 旧脚本 `tests/step*.py` 等：将来可迁到 `tests/experiments/`，**非当前任务**。

### 用户环境（补充）

- 工作区常见为 `**G:\car_charge\car_charge`**（与文档旧载 `d:\car_charge` 可能不一致，以实际打开路径为准）。
- Windows 下 `**osm_loader` 会 print 中文**；EMA 相关测试文件内已对 `stdout/stderr` 做 **UTF-8 reconfigure**，避免 `cp1252` 控制台编码错误。

---

## 六、用户性格 / 协作风格（保留）

1. 偏好**直接结论 + 可执行步骤**
2. 选项化提问比开放式提问更高效
3. 会调整方向（如测试驱动）——**接受，不劝回**
4. 控制 Cursor 自由度，避免「一把梭」改代码

---

## 七、暂时不要主动展开的话题（保留）

1. **全管线 reward** 是否仍有一处读累计、一处读 snapshot（**充电完成路径**已在桩侧对齐「先 snapshot 再清零累计」）
2. `should_request_charge_decision` 与 `get_pending_decisions`（`**t2_arrival` 已空**）的命名/文档是否仍易误导
3. `real_env` 与基类 `__init__` 不一致的长期隐患
4. `t2_advantage_fc` 是否删除/收敛
5. 历史训练脚本归档

---

## 八、训练与评估（当前 EMA + hindsight 主线）

### 训练入口

```bash
python train_hindsight.py --episodes 800 --steps-per-episode 100 --save-every 50 --save-dir checkpoints_hindsight
```

训练默认使用：

- `env.real_env.RealTrafficEnv`
- `map_outputs/ema/ema.graphml`
- `agents.hindsight_dqn_agent.HindsightDQNAgent`
- `env.power_grid_pp.PPPowerGrid33`
- **规模默认**：**80 EV / 4 站 / 每站 8 桩**（`--num-evs`、`--num-stations`、`--num-chargers-per-station`；与 `TrafficPowerEnv` 玩具网格上的 trainer smoke 对齐思路一致）

`train_hindsight.py` / `tools/eval_hindsight.py` 支持 `--respawn` / `--no-respawn`（默认开启充满后重生，与桩侧 `respawn_after_full_charge` 一致）。站点 `update_price`：传入 `lmp` 时走 **LMP 分支**，否则走 ToU；重生相关单测已用固定 `lmp` 与 `monkeypatch` 隔离定价细节。**UE**：训练侧加 **`--ue-verbose`** 可打印 Frank–Wolfe 迭代（默认静默，避免刷屏）。

**背景基线**：**`train_hindsight.py` / `tools/debug_station_bias.py` / `draw_ema_network --congestion`** 默认启用 UE（`EMA_net.tntp` + `EMA_trips.tntp`，`RealTrafficEnv` 传 `background_ue_*`）；**`--no-ue-background`**（或出图脚本等价开关）回退启发式；**`--ue-net-tntp` / `--ue-trips-tntp` / `--ue-max-iter` / `--ue-tol` / `--ue-scale`**（训练与偏站脚本）可调。每个 **`reset`**（及新建环境 **`__init__`**）会再跑 UE / FW，**耗时**；**`tools/eval_hindsight.py`** 仍为启发式，与训练/偏站诊断对比时注意。**联邦评估**：`tools/eval_federated_hindsight.py` 已添加，但当前仍在处理客户端变体网 + Thevenin 初始化的 `pandapower` 只读写回兼容性（建议先用单机旧评估脚本做 baseline）。

训练日志会输出：

- `avg_trip`
- `avg_queue`
- `avg_fee`
- `avg_reward`
- `epsilon`

其中 `epsilon` 现在每个 episode 衰减一次，默认 `epsilon_decay=0.994`，大约 500 个 episode 左右能降到接近 `epsilon_min=0.05`。

### 评估入口

```bash
python tools/eval_hindsight.py --model-path checkpoints_hindsight/model_ep50.pth
```

**注意**：`eval_hindsight.py` 当前仍用 **启发式** `background_edge_base_flows`（未传 `background_ue_*`），与 **`train_hindsight` / `debug_station_bias` / `draw_ema_network --congestion`** 的默认 UE 不一致；横比指标前需知悉或后续改脚本对齐。

评估脚本默认比较三组策略：

- `random`
- `shortest_path`
- `model_greedy`

评估指标为：

- `avg_trip`
- `avg_queue`
- `avg_fee`
- `avg_reward`
- `completed`

### 评估策略含义

- `random`：在合法动作里随机选站
- `shortest_path`：选预计行驶时间最短的站
- `model_greedy`：按当前模型 Q 值选最大动作

### 常用模型文件

- `checkpoints_hindsight/model_final.pth`
- `checkpoints_hindsight/model_ep50.pth`

### 常用输出目录

- 训练模型：`checkpoints_hindsight/`
- EMA 路网缓存：`map_outputs/ema_cache/`

### 工具脚本与环境类约定（重要）

- **新建或修改「依赖真实/合成路网站点与拓扑」的工具、诊断脚本、Notebook**：在配置好 `sys.path`（项目根入 path）之后，**应使用 `env.real_env.RealTrafficEnv`**（参数与 `train_hindsight.py` / `tools/debug_station_bias.py` / `tools/eval_hindsight.py` 对齐：`graphml_file`、EMA 缓存目录等；部分离线路网脚本可设 `offline=True`），**不要默认实例化 `TrafficPowerEnv`**，以免与 **EMA 图、背景流、`station_node_ids`、BPR 边权** 等训练侧行为不一致。**背景**：`train_hindsight` / `debug_station_bias` / `draw_ema_network --congestion` 默认 **UE**；`eval_hindsight` 仍为启发式（见第八节）。
- **例外**：只验证基类纯数学/约束、刻意要 3×3 网格玩具、或控制依赖的极简 smoke（如 `tests/test_trainer_integration.py` 使用 `TrafficPowerEnv`）时，可继续使用 `**TrafficPowerEnv`**。

---

## 九、下一个 session 的开场建议（已更新）

若用户只说「继续」或「继续测试」：

1. 下一优先文件：`**agents/dqn_base.py**`（写 4–7 条可隔离单测或冒烟）。
2. **推荐全量**：`**python -m pytest tests/ -v`**（仓库已用 `pytest`：`test_snapshot_bug`、`test_respawn_state_cleanup` 等；EMA 仍依赖 `**map_outputs/ema/ema.graphml`** + **osmnx**；**~1–2 分钟**级）。
3. **unittest 全量**（可选）：`**python -m unittest discover -s tests -p "test_*.py" -v`**（会收集 `pytest` 风格文件时行为因版本而异，**优先 pytest**）。
4. **EMA 分层单独跑**：
  - L1：`python -m pytest tests/test_real_env_ema_graph_integration.py -v -s`  
  - L2：`python -m pytest tests/test_real_env_ema_level2_physics.py -v -s`（可加 `**python -u`** 看 `_diag_print`）  
  - L3：`python -m pytest tests/test_real_env_ema_level3_performance.py -v -s`（**T3.3** 要 **psutil**）
5. 专项：`python -m pytest tests/test_snapshot_bug.py tests/test_respawn_state_cleanup.py tests/test_respawn_logic.py -v`
6. **Trainer**：`python -m pytest tests/test_trainer_core.py tests/test_trainer_integration.py -v`（核心逻辑 + 真 agent smoke；集成侧 **8 桩 / 200 step / `replay(4)`** 等见第十节表）
7. **训练规模 smoke**：`python -m pytest tests/test_hindsight_train_scale.py -v`
8. **背景流**：`python -m pytest tests/test_background_traffic.py -v`（启发式 + 日曲线；**UE 单测未覆盖**，UE 自检可 `python env/background/ue_assignment.py` 或 `draw_ema_network --congestion`）

**不要**在未拿到 traceback 时替用户「猜着改」失败用例。

---

## 十、本会话落地的测试资产（便于索引）


| 文件                                              | 内容提要                                                                                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_entities.py`                        | `EV` 初始化、`move`、SOC 钳制、`remaining_edge_time_h`                                                                                         |
| `tests/test_charging_station.py`                | 电价、`optimize_power` 限幅、`step` 出队等                                                                                                      |
| `tests/test_power_grid_pp.py`                   | `PPPowerGrid33` 解析母线、潮流、`optimize_power` 透传                                                                                            |
| `tests/test_base_env.py`                        | `should_request_charge_decision`、`_find_ev_by_id`、静态解析、`_bpr_time_h`（静态点值）                                                             |
| `tests/test_bpr_congestion.py`                  | BPR 对 `**edge_active_counts`**、单调、`ratio` 与手算一致                                                                                        |
| `tests/test_background_traffic.py`              | 144-step 背景流日周期、高斯双峰、边级权重；**需图时**用 `**RealTrafficEnv(offline=True)`**；**不含 UE FW**（UE 见 `ue_assignment` / 手动跑 `draw_ema_network --congestion`） |
| `tests/test_trainer_core.py`                    | `**trainer.HindsightTrainer`**：pending key（`ev_id, charge_sessions`）、snapshot reward、入 buffer 时机；mock 场景需 `**session_idx_override=0**`；随机 SOC 场景用 **`_ensure_low_soc_idle_all_evs`** 保证 T0 决策与 buffer 增长 |
| `tests/test_real_env.py`                        | `offline=True` 合成路网 + `reset`/`step({})`                                                                                               |
| `tests/test_real_env_ema_graph_integration.py`  | **层级 1**：T1.1 路网、T1.2 `reset`（50 EV）、T1.3 单步、`T1.4` 50 步；合法状态**无** `T2_PENDING`                                                        |
| `tests/test_real_env_ema_level2_physics.py`     | **层级 2**：T2.1–T2.7 + 诊断输出                                                                                                              |
| `tests/test_real_env_ema_level3_performance.py` | **层级 3**：T3.1 单步耗时、T3.2 cProfile top15、T3.3 RSS（psutil）                                                                                |
| `tests/test_snapshot_bug.py`                    | 充满后 **snapshot = 当次会话累计**；`**total_fee_paid` 等与 snapshot 写入后清零**                                                                       |
| `tests/test_respawn_state_cleanup.py`           | 充满后 **path / 边状态 / target_station_idx / assigned_station** 等清零                                                                         |
| `tests/test_respawn_logic.py`                   | 充满 → snapshot → 会话清零 → **合法节点重生**、SOC **10%–20%**、多轮重生、`TrafficPowerEnv.step` 不崩；单测对桩注入 `legal_respawn_nodes`                          |
| `tests/test_trainer_integration.py`             | 真 `DQNAgent` + `TrafficPowerEnv` + `HindsightTrainer` 短跑；**每站 8 桩**、**200 step**、**`replay(4)`** smoke；防 `store_transition` / replay 接口漂移 |
| `tests/test_hindsight_train_scale.py`           | 与 `train_hindsight` **默认规模**（80/4/8）+ `RealTrafficEnv` 离线路网 **smoke**（接口与构造不崩）                                                      |
| `tests/test_network.py`                         | `FeatureEncoder`、`GraphQNetwork`（`t0`/`t2`、mask、非法 `action_type`）                                                                      |


**统一运行（推荐）**：

```bash
python -m pytest tests/ -v
```

**仅 unittest + 旧列表**（不含 `pytest` 独占文件时勿单独依赖）：

```bash
python -m unittest tests.test_entities tests.test_charging_station tests.test_power_grid_pp tests.test_base_env tests.test_real_env tests.test_real_env_ema_graph_integration tests.test_real_env_ema_level2_physics tests.test_real_env_ema_level3_performance tests.test_network -v
```

---

## 十一、`env/entities.py` 行尾注释

`EV.__init__` 各字段有**中文行尾注释**（便于读源码写测）。`**t2_`* 已从类中删除**；第五节以当前代码为准。

---

## 十二、待做事件（下一阶段优先级）

1. ~~**车辆重生逻辑**~~：**已落地**（`EV.reset_for_respawn`、`ChargingStation._choose_respawn_node`、合法节点集、`tests/test_respawn_logic.py` + trainer smoke）。**待补小项**：若 `RealTrafficEnv` 对「重生节点是否排除站点」有特殊需求，再在 `legal_respawn_nodes` 构造里区分。
2. ~~**背景交通流第一版**~~：**已在 `TrafficPowerEnv` 与 `RealTrafficEnv` 落地**（`env/background_traffic.py` + `setup_background_traffic_and_respawn_nodes` + `tests/test_background_traffic.py`（`RealTrafficEnv` 离线）+ `tools/debug_background_traffic.py`（同上））。**已加 UE 空间基线**（`env/background/ue_assignment.py` + `build_base_background_flows` 可选 TNTP + `RealTrafficEnv` kwargs；`draw_ema_network --congestion`、`train_hindsight.py`、`debug_station_bias.py` 默认走 UE）。**可选后续**：**`eval_hindsight.py` 对齐 UE**、缓存 UE 结果避免每次 `reset` 重复 FW、强度写入 `info`。**联邦训练/评估**：已新增 `train_federated_hindsight.py`、`trainer/federated_hindsight_trainer.py`、`tools/eval_federated_hindsight.py`；当前重点是把客户端变体网的 `pandapower` / Thevenin 兼容性修稳，再把联邦模型评估流程串起来。
3. **拓展实验规模**（论文向）：当前训练默认 **80 车 / 4 站 / 每站 8 桩**；可再扩至 **100+ 车**、更多站或桩，做多站点选站分布、排队、拥堵消融（与 `TRAIN_DEFAULT_*` / CLI 对齐后跑 `test_hindsight_train_scale` 与长训 smoke）。

### 可视化 / 诊断

```bash
python tools/debug_background_traffic.py --output background_traffic_edge.png
# 指定边（须为当前离线路网中的有向边键之一）：
python tools/debug_background_traffic.py --edge 6,7 --output background_traffic_edge.png
```

**EMA 路网图（`tools/draw_ema_network.py`）**

- **默认**（无 `--congestion`）：只读 `map_outputs/ema/EMA_net.tntp`（可用 **`--input`** 改路径），拓扑布局（Kamada–Kawai / Spring）；输出 **`{stem}.pdf` + `{stem}.png`（600 dpi）**，`bbox_inches=tight`，全局 **Arial 优先** 等小字号样式。
- **`--congestion`**：用 **`RealTrafficEnv`** + 默认 **`map_outputs/ema/ema.graphml`** + **`map_outputs/ema_cache`**（与训练一致）；在 **`reset` 后跑若干步 `step`** 再取**当前步**边上量上色。
- **背景基线（拥堵）**：**默认 UE**——`--input` 指向的 **`EMA_net.tntp`** + **`--trips-tntp`**（默认 `map_outputs/ema/EMA_trips.tntp`）经 **`compute_ue_background_flows`** 写入 `background_edge_base_flows`；**`--no-ue-background`** 回退节点启发式；**`--ue-max-iter`** 默认 **800**（纯 FW 在 EMA 上贴近 `1e-4` gap 常需数百轮）。**`__init__` 与 `reset` 各跑一次 UE**；**Frank–Wolfe 迭代默认静默**，仅在 **`--verbose-background`** 时经 **`background_ue_verbose=True`** 打印详细日志（否则会刷屏两遍）。
- **边上色默认 `x_flow`**：与 `base_env._dynamic_profiles(..., add_vehicle=0)` 一致，**`x_flow = _edge_flow(u,v) + _background_flow(u,v)`**；拥堵支路 **线宽 + alpha** 随归一化流量变化，**`magma`**（掐暗端）色条；**`--ratio-vmax`** 为色条上限。
- **`--congestion-edge-metric bpr`**：**`t_BPR / t0`**；色条 **`coolwarm`**。
- **常用参数**：`--snapshot-step N`、`--sim-steps`、`--num-evs`、`--seed`；**`--background-flow-mult`** 在 UE/启发式基线算出后再整体放大（仅本脚本）；**`--verbose-background`** 打印基线与 profile 后背景均值等。

```bash
python tools/draw_ema_network.py
python tools/draw_ema_network.py --congestion --output map_outputs/ema/EMA_xflow.png
python tools/draw_ema_network.py --congestion --snapshot-step 120 --num-evs 45 --verbose-background
python tools/draw_ema_network.py --congestion --congestion-edge-metric bpr --snapshot-step 60
python tools/draw_ema_network.py --congestion --no-ue-background
python tools/draw_ema_network.py --congestion --trips-tntp map_outputs/ema/EMA_trips.tntp --ue-max-iter 800
```

**从 TNTP 重生成 `ema.graphml`（`map_tools/tntp_net_to_graphml.py`）**

- 节点 **`0..N-1`** 与 TNTP 交通节点对应：**`tntp_id = node + <FIRST THRU NODE>`**（EMA 通常为 `first_thru=1`）。
- **`x,y`** 为拓扑 spring 布局映射到示意经纬度框，**非测绘坐标**；写盘前属性为字符串以满足 **`osmnx.load_graphml`**；加载时由 **`osm_loader._coerce_graphml_numeric_attrs`** 转回数值。
- 覆盖输出后务必删除 **`map_outputs/ema_cache/local_ema*.pkl`**，否则仍命中旧缓存；并核对 **`config/stations.json`** 中站点节点号是否仍合法。

```bash
python map_tools/tntp_net_to_graphml.py
python map_tools/tntp_net_to_graphml.py --backup --layout-seed 42
```

训练/评估常用（`train_hindsight` / `debug_station_bias` 与训练对齐默认 **UE**；`eval_hindsight` 仍为启发式）：

```bash
python train_hindsight.py --network station_only --no-use-action-mask
python train_hindsight.py --no-ue-background
python train_hindsight.py --ue-max-iter 400
python tools/debug_station_bias.py --model-path checkpoints_hindsight/model_ep400.pth --no-use-action-mask
python tools/debug_station_bias.py --ue-max-iter 200
python tools/eval_hindsight.py --model-path checkpoints_hindsight/model_ep50.pth
python tools/compare_travel_time_background.py --help
```

