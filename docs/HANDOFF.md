# 交接文档 · EV 充电导航项目测试驱动阶段

> **写给下一个 Claude session**  
> **最后更新：2026-05-12**（累计：EMA L1/L2/**L3**、`network` 冒烟、**EMA graphml 限速 m/h 修复**、**环境侧移除 t2 到站决策**、`charging_station` **充满电 snapshot + 会话累计重置 + 路上状态清理**、`pytest` 专项测、Level2 诊断打印）  
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

| 项 | 状态 |
|---|---|
| 重构计划 | ❌ 已废弃（不要复活） |
| t2 多步 MDP 方案 | ❌ 已废弃（不要复活） |
| 单步 hindsight contextual bandit | ⏸️ 暂缓（等测试完成后再说） |
| `tests/test_entities.py`（EV + mock 路网） | ✅ 已落地（含 `remaining_edge_time_h`） |
| `tests/test_charging_station.py` | ✅ 已落地 |
| `tests/test_power_grid_pp.py`（`compute_thevenin=False`） | ✅ 已落地 |
| `tests/test_base_env.py`（决策/BPR/解析等可隔离逻辑） | ✅ 已落地 |
| `tests/test_real_env.py`（offline 合成路网） | ✅ 已落地 |
| `tests/test_real_env_ema_graph_integration.py`（层级 1：T1.1–T1.4） | ✅ 已落地，路网路径 `map_outputs/ema/ema.graphml` |
| `tests/test_real_env_ema_level2_physics.py`（层级 2：T2.1–T2.7） | ✅ 已落地；用例末 **`_diag_print`** 打关键统计（T2.7 abandon 上界已随「无 T2_PENDING」放宽） |
| `tests/test_real_env_ema_level3_performance.py`（层级 3：T3.1–T3.3 耗时 / cProfile / 内存） | ✅ 已落地；**T3.3** 依赖 **psutil**，未装则 skip |
| `tests/test_snapshot_bug.py`（充满后 **会话** 累计字段 + snapshot 语义） | ✅ 已落地（`pytest`） |
| `tests/test_respawn_state_cleanup.py`（充满后 **path/边/目标站** 等路上状态清空） | ✅ 已落地（`pytest`） |
| `tests/test_network.py`（FeatureEncoder / GraphQNetwork 冒烟） | ✅ 已落地 |
| **`agents/dqn_base.py`** | ⬜ **下一个优先单测文件** |
| `agents/FederatedDQN.py` | ⬜ 待定 |
| `training/trainer.py` | ⬜ 待定 |
| git 备份 | ⚠️ 仍建议用户每步 `git commit`（历史状态未强制核验） |

---

## 三、接手协议（看完这一节就能开工）

### 若继续「按文件写单测」

1. **EV 上 `t0_*` / `abandon_*` 等多步遗留字段**：写测时不必当核心契约；**`t2_*` 已从 `entities.EV` 删除**，环境已无 **`T2_PENDING` / `apply_t2_action`**（见第五节）。
2. **单文件 4–7 个用例**为宜；集成/EMA 层级可单独文件、单独命名（`test_real_env_ema_*`）。
3. **测试 FAILED**：先贴 traceback，区分「断言/场景写错」与「实现 bug」；**禁止**为绿而盲改断言或盲改实现（纪律见第四节）。

### 若用户说「继续测试」

下一文件顺序（与早前规划一致，已从 `entities` 推进到 `network` 之后）：

```
… 已完成：entities → charging_station → power_grid_pp → base_env → real_env(offline) → EMA L1/L2/**L3** → agents/network.py → **charging_station 充满语义 pytest**
→ 当前：agents/dqn_base.py
→ 其后：agents/FederatedDQN.py → training/trainer.py
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

**核心活字段**（统计）：`travel_time_h`, `wait_time_h`, `total_fee_paid`, `total_energy_charged`, `charge_sessions` 等。注意 **`travel_steps` / `wait_steps` / `charge_steps`** 在 `base_env.step` 里**按状态每步 +1**，**充满电时未重置**（全寿命计数语义）；是否会话化待单独任务。

### `env/charging_station.py` 充满电分支（`ev.soc >= 95.0`，字面常量）

1. **先**把当前累计写入 snapshot：`charge_fee_snapshot` / `charge_queue_time_h` / `charge_travel_time_h`（值为写入瞬间的 `total_fee_paid` / `wait_time_h` / `travel_time_h`）。  
2. **再**将会话内累计 **`travel_time_h`, `wait_time_h`, `total_fee_paid`, `total_energy_charged` 置 0**（语义改为「本次充电会话内累计」，避免第二次充满 snapshot 混入第一次）。  
3. **再**清空路上状态：`path`, `last_traversed_nodes`, `current_edge_*`, `remaining_edge_time_h`, `target_station_idx`, `assigned_station` 等与 `EV.__init__` 空闲语义对齐。  
4. 然后 `status = "IDLE"`、`charge_sessions += 1`、flags、`respawn_after_full_charge` 随机 SOC 等（与原先一致）。

### `env/base_env.py` 与 **EMA graphml 限速**

- **`TrafficPowerEnv._parse_speed_kph`**：若数值 **`>= 1000`**，按 **m/h→km/h** 除以 1000（修复 graphml 把米/小时误标为 `speed_kph` 导致 **亚秒级穿边**）。单测见 `tests/test_base_env.py` 中 `test_parse_speed_graphml_m_per_h_mislabeled_as_kph`。
- **环境侧已无 t2 决策**：到站即 **`_commit_arrival_to_waiting`**（入队 `WAITING`）；已删 **`apply_t2_action`**、**`T2_PENDING`** 及超时 auto-accept 日志；**`get_pending_decisions`** 的 **`t2_arrival` 恒为 `[]`**；**`info["pending_t2"]`** 仍为「本步到站入队 EV」列表（telemetry），**非**待决策队列。
- **`training/trainer.py`**：`pending_t2` 环已删；`T0` 的 `backfill_snext` 在 **`env.step` 之后**用 `get_graph_state_for_ev` 立即填上。
- **`agents/dqn_base.py`**：混合 `step_type` 时 **T0 的 `next_type`** 从 **`'t2'`** 改为 **`'t0'`**（新轨迹无 T2）；回放缓冲里旧 **T2** 样本仍可被旧分支吃到。
- `real_env`：不调用 `TrafficPowerEnv.__init__()`，自成初始化（已知隐患，HANDOFF 不要求在此修）。
- **`agents/network.py`**：仍含 **`t2_advantage_fc`** 与 **`action_type='t2'`**（单测 `test_network` 仍覆盖）；与「环境无 t2」可并存，属 **RL 网络遗留**。
- 旧脚本 `tests/step*.py` 等：将来可迁到 `tests/experiments/`，**非当前任务**。

### 用户环境（补充）

- 工作区常见为 **`G:\car_charge\car_charge`**（与文档旧载 `d:\car_charge` 可能不一致，以实际打开路径为准）。
- Windows 下 **`osm_loader` 会 print 中文**；EMA 相关测试文件内已对 `stdout/stderr` 做 **UTF-8 reconfigure**，避免 `cp1252` 控制台编码错误。

---

## 六、用户性格 / 协作风格（保留）

1. 偏好**直接结论 + 可执行步骤**  
2. 选项化提问比开放式提问更高效  
3. 会调整方向（如测试驱动）——**接受，不劝回**  
4. 控制 Cursor 自由度，避免「一把梭」改代码  

---

## 七、暂时不要主动展开的话题（保留）

1. **全管线 reward** 是否仍有一处读累计、一处读 snapshot（**充电完成路径**已在桩侧对齐「先 snapshot 再清零累计」）  
2. `should_request_charge_decision` 与 `get_pending_decisions`（**`t2_arrival` 已空**）的命名/文档是否仍易误导  
3. `real_env` 与基类 `__init__` 不一致的长期隐患  
4. `t2_advantage_fc` 是否删除/收敛  
5. 历史训练脚本归档  

---

## 八、下一个 session 的开场建议（已更新）

若用户只说「继续」或「继续测试」：

1. 下一优先文件：**`agents/dqn_base.py`**（写 4–7 条可隔离单测或冒烟）。  
2. **推荐全量**：**`python -m pytest tests/ -v`**（仓库已用 `pytest`：`test_snapshot_bug`、`test_respawn_state_cleanup` 等；EMA 仍依赖 **`map_outputs/ema/ema.graphml`** + **osmnx**；**~1–2 分钟**级）。  
3. **unittest 全量**（可选）：**`python -m unittest discover -s tests -p "test_*.py" -v`**（会收集 `pytest` 风格文件时行为因版本而异，**优先 pytest**）。  
4. **EMA 分层单独跑**：  
   - L1：`python -m pytest tests/test_real_env_ema_graph_integration.py -v -s`  
   - L2：`python -m pytest tests/test_real_env_ema_level2_physics.py -v -s`（可加 **`python -u`** 看 `_diag_print`）  
   - L3：`python -m pytest tests/test_real_env_ema_level3_performance.py -v -s`（**T3.3** 要 **psutil**）  
5. 专项：`python -m pytest tests/test_snapshot_bug.py tests/test_respawn_state_cleanup.py -v`

**不要**在未拿到 traceback 时替用户「猜着改」失败用例。

---

## 九、最后一句话（保留）

项目的核心风险之一是**自由度失控**。测试驱动要赢：**读源码 → 计划 →（用户确认）→ 写测 → 报告**，**不许跳步**。

---

## 十、本会话落地的测试资产（便于索引）

| 文件 | 内容提要 |
|------|-----------|
| `tests/test_entities.py` | `EV` 初始化、`move`、SOC 钳制、`remaining_edge_time_h` |
| `tests/test_charging_station.py` | 电价、`optimize_power` 限幅、`step` 出队等 |
| `tests/test_power_grid_pp.py` | `PPPowerGrid33` 解析母线、潮流、`optimize_power` 透传 |
| `tests/test_base_env.py` | `should_request_charge_decision`、`_find_ev_by_id`、静态解析、`_bpr_time_h` |
| `tests/test_real_env.py` | `offline=True` 合成路网 + `reset`/`step({})` |
| `tests/test_real_env_ema_graph_integration.py` | **层级 1**：T1.1 路网、T1.2 `reset`（50 EV）、T1.3 单步、`T1.4` 50 步；合法状态**无** `T2_PENDING` |
| `tests/test_real_env_ema_level2_physics.py` | **层级 2**：T2.1–T2.7 + 诊断输出 |
| `tests/test_real_env_ema_level3_performance.py` | **层级 3**：T3.1 单步耗时、T3.2 cProfile top15、T3.3 RSS（psutil） |
| `tests/test_snapshot_bug.py` | 充满后 **snapshot = 当次会话累计**；**`total_fee_paid` 等与 snapshot 写入后清零** |
| `tests/test_respawn_state_cleanup.py` | 充满后 **path / 边状态 / target_station_idx / assigned_station** 等清零 |
| `tests/test_network.py` | `FeatureEncoder`、`GraphQNetwork`（`t0`/`t2`、mask、非法 `action_type`） |

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

`EV.__init__` 各字段有**中文行尾注释**（便于读源码写测）。**`t2_*` 已从类中删除**；第五节以当前代码为准。
