# 交接文档 v2 · EV 充电导航项目 · env 修复阶段已完成

> **写给下一个 Claude session**
> 日期：2026-05-12
> 上一份交接文档：HANDOFF.md（v1，测试驱动路线启动前的状态）
> 本次交接：env 修复阶段完成，下一步是 trainer 设计

---

## 一、本次 session 的成果（已完成,不要重做）

用户 Eclipse 在 v1 交接之后完成了完整的 env 测试 + 3 个 bug 修复。当前 env 层处于**最干净状态**，可以基于此写 trainer。

### 已完成的工作

| # | 任务 | 改动 | 测试 | git 状态 |
|---|---|---|---|---|
| 1 | EMA 集成测试（T1/T2/T3 共 14 个） | 新建 3 个测试文件 | PASSED | 已 commit |
| 2 | 子任务 1：snapshot bug 修复 | charging_station.py +4 行 | 2 测试 PASSED | 已 commit |
| 3 | 子任务 2：路上状态清理 | charging_station.py +9 行 | 3 测试 PASSED | 已 commit |
| 4 | 子任务 3：BPR 量纲修复 | base_env.py 改 2 行 | 3 测试 PASSED | 已 commit |

**当前测试总数：65 个，全部 PASSED**。

### 3 个 bug 的本质回顾

**Bug 1 (snapshot)**：充电完成时 `ev.charge_fee_snapshot = ev.total_fee_paid` 是把"累计值"赋给 snapshot，第 2 次充电的 snapshot 错误包含第 1 次的累计。

**修复**：充满电后清零 `travel_time_h / wait_time_h / total_fee_paid / total_energy_charged`。语义变为"会话内累计"。

**Bug 2 (路上状态)**：充满电后 EV 上的 `path / last_traversed_nodes / current_edge_* / target_station_idx / assigned_station / remaining_edge_time_h` 等字段没清，导致下次决策有脏数据。

**修复**：充满电后这 9 个字段全部重置回 init 默认值。`curr_node` 故意不清（EV 留在充电站节点是合理物理语义）。

**Bug 3 (BPR 量纲)**：公式 `ratio = x_flow / (c_capacity * step_duration_h)` 单位不匹配——`x_flow` 是瞬时车数，`c_capacity` 是辆/小时。导致 ratio 永远极小，BPR 几乎不响应流量。

**修复**：改成 `ratio = x_flow / (c_capacity * t0_h)`。`c_capacity * t0_h` 物理意义是"边的瞬时承载能力"，量纲自洽。

---

## 二、env 现状（trainer 设计依赖的事实）

### EV 类生命周期模型

**Respawn 循环模型**：
- 一辆 EV 一个 episode 内可完成多次充电
- 充满电后 SOC 重置到 U[20, 50]，EV 留在充电站节点
- 靠 IDLE 每步 SOC -0.5 拉低，达到 `charge_trigger_soc=30` 时触发新一轮 t0 决策
- `ev.charge_sessions` 每完成一次充电 +1

### Reward 字段读什么

**读 snapshot 字段（B 套），不读累计字段（A 套）**：

```python
# trainer 里完成 EV 的 reward 计算：
def _reward_completed(self, ev) -> float:
    return -(0.3 * ev.charge_travel_time_h    # snapshot, 单次充电的行驶时间
             + 0.5 * ev.charge_queue_time_h    # snapshot, 单次充电的等待时间  
             + 0.03 * ev.charge_fee_snapshot)  # snapshot, 单次充电的费用
```

注意：累计字段（`travel_time_h / wait_time_h / total_fee_paid`）现在在充满电时被清零，trainer **不要直接读累计字段**作为 reward——读 snapshot 才是单次会话的真实代价。

### `info` dict 结构

`env.step()` 返回的 info 包含：
- `info["completed"]`：list[dict]，每个字典含 `ev_id / state / actual_trip_time_h / actual_queue_time_h / charging_fee`
- `info["abandoned"]`：abandon 的 EV 列表

注意：`info["completed"]` 的元素是 **dict 不是 EV 对象**。如果 trainer 需要读 EV 上的字段（比如 charge_travel_time_h），需要通过 `ev_id` 反向找到 EV 实例，或者 dict 里已经有的 `actual_trip_time_h` 这套字段也可以直接用——**两套字段语义查一下确认**。

### 关键接口

```python
env.reset()                          # 返回初始 state（具体 schema 待查）
env.step()                           # 不传 action（环境内部处理）
env.get_pending_decision_evs()       # 返回需要 t0 决策的 EV 列表
env.get_graph_state_for_ev(ev)       # 单 EV 的图状态
env.get_action_mask(ev)              # 单 EV 的合法 action mask
env._completed_evs_this_step         # 直接读字段也能拿到
env._abandoned_evs_this_step
```

具体签名和细节让 Cursor 临时查。

---

## 三、用户的当前决策（已锁定）

1. **不重构项目**（保留这个决定，不要又劝重构）
2. **不加 reroute 创新点**（多次确认过，留 v2）
3. **走单步 hindsight contextual bandit**（done=True，gamma 实际不用，reward = hindsight 算）
4. **env 修复阶段完成，下一步是 trainer**（当前断点）
5. **trainer pending dict key 必须用 `(ev.id, session_idx)`**——见 TRAINER_DESIGN_NOTES.md（如果用户已创建）

---

## 四、悬而未决的问题（不要主动提，记在 TESTING_TODO 里）

按重要性排序：

### 🟡 仿真流量分布过低（影响论文卖点）

EMA 50 EV / 50 step 最忙边只有 3 辆车，新 BPR 公式数学上对了但低流量下增量仍 ≈ 0。**不是 bug，是实验配置问题**。

候选解决方案（等 trainer 跑通后看实际数据决定）：
- A. 增加 num_evs（50 → 200+）
- B. 缩小路网到 sub-graph
- C. 调整 OD 分布让 EV 聚集
- D. 增加 step_duration_h

### 🟡 LMP 缓存可能失效，OPF 占总时间 42%

T3.2 cProfile 显示 `get_lmp` 20 step 内被调 4 次，每次 0.28s。可能 `lmp_update_interval / _cached_lmp` 没生效。当前 11 分钟一个 episode 可接受，**不阻塞 trainer**。

### 🟢 EV 死字段清理（14 个）

- 11 个 t0/t2 死字段：`t0_state, t0_action, t0_step, t2_step, t2_decision_pending, travel_time_at_dispatch, t2_action, t2_state, t2_pending_steps, abandon_reason, just_abandoned_this_step`
- 3 个统计字段（疑似死，待 grep 全仓库确认）：`travel_steps, wait_steps, charge_steps`
- 2 个内部缓存：`_decision_state, _decision_snap`

**触发时机**：trainer 跑通且收敛后，反向确定死字段集合再清。

### 🟢 充满判定字面常量

`charging_station.py L191` 写死 `if ev.soc >= 95.0`，应该改成 `>= ev.target_soc`。**论文做 ablation 时再修**。

### 🟢 T2.6 测试构造问题

T2.6 用 `info["peak_edge_flows_this_step"]` 取 `x_high`，跟 BPR 实际读的 `edge_active_counts` 不是同一变量。BPR 修复后 T2.6 仍然 PASSED 但意义不大（增量在浮点尾巴）。**不阻塞，但如果做 v2 ablation 可能要修**。

---

## 五、接手协议

### 如果用户上来说"继续"或"开始 trainer"

按以下顺序问：

1. **先 git status 确认环境干净**——上一个 session 应该 commit 完了，但确认一次
2. **询问用户偏好**：用 `ask_user_input_v0` 给 3 个选项
   - 直接给 `replay_buffer.py` 的 Cursor 提示词（从 trainer 最底层开始写）
   - 先一起讨论 trainer 整体设计，再开始写
   - 先读 TRAINER_DESIGN_NOTES.md（如果存在）确认设计决策

### 如果用户贴了 Cursor 的 trainer 工作产物

按"双 gate 协作模式"审核：

**Gate 1 检查 Cursor 是否调查了再写**：
- 接口语义查清楚了吗（reward 读哪套字段、info 结构、pending key 设计）
- 没有发明不存在的 API

**Gate 2 检查 TDD 流程**：
- 先写测试看 FAILED 了吗
- 测试覆盖核心不变量（每次充电 reward 与上次充电独立）
- 没有"为了过而写的假测试"

### 如果用户说"修 trainer 跑出来某个问题"

**严禁直接给修复方案**。按 BPR 调查的模式：
1. 先让 Cursor 做 read-only 调查
2. 拿到事实清单，判断根因
3. 才给 TDD 修复提示词

---

## 六、工作纪律（这次 session 印证成功的协作模式）

这一次 session 跟前 5 轮失控完全不同，关键在以下纪律。**不要丢**：

### 纪律 1：每个修复都走完整 TDD（先写测试 → FAILED → 改代码 → PASSED）

3 个 bug 修复都用这个流程。**Step 2 看到 FAILED 是必经环节**，跳过就是失控起点。

### 纪律 2：复杂修复用"双 gate"协作

子任务 2 和 BPR 修复用了 "Step 0 调查 → Step 1 方案确认 → 才能写代码"。这个模式比单 gate 多花 10 分钟，但避免了所有"自由发挥"。

### 纪律 3：Cursor 主动问问题是好事，鼓励它

子任务 2 时 Cursor 主动问"要不要清 _decision_state / _steps 字段"——这是优秀的协作行为。我让用户明确回答"暂不处理"而不是让 Cursor 自己决定。

**不要批评 Cursor 多嘴**，那是它在保持单一职责。

### 纪律 4：每个修复结束立刻 commit

3 个修复都有独立 commit。任何时候用户说"完成了一步"，提醒一句 git commit。

### 纪律 5：发现新问题写 TODO，不切换任务

子任务 2 暴露 `_steps` 字段疑似死代码 → 不立刻清，记 TODO。
子任务 3 暴露 EMA 流量过低 → 不立刻调，记 TODO。
**保持单一任务焦点是这次成功的核心**。

---

## 七、用户性格 / 协作风格更新

（基于本次 session 的观察补充）

1. **执行力强**：贴提示词、跑 Cursor、贴结果，循环很快
2. **会自己改主意但有理由**：本来想改 respawn 选方案 2，听了风险解释后改成方案 1，**这是好的决策灵活性**
3. **偶尔想绕开"调查"直接修**："那还是选项 1 吧"——这时候**让步是对的**，不要再坚持"必须先调查"
4. **不需要长解释**：3-5 段话讲清楚就够，超过 1000 字开始失去注意力
5. **`ask_user_input_v0` 是最有效的工具**：用了 7-8 次都很顺畅
6. **会问"创新点"**（如这次"每节点决策"）——认真分析但用时间预算劝退到 v2

---

## 八、下一个 session 的"开场动作建议"

如果用户上来说"继续"或"开始 trainer"：

1. **不要主动复盘**——本文档已经完整记录，不需要再讲一遍
2. **不要主动建议改方案**——决策都锁定了
3. **不要主动写代码**——先问用户从哪个模块开始

**第一句话推荐**：
> "env 修复完成,65 个测试 PASS。trainer 该从哪里开始?给你 3 个选项..."
> [用 ask_user_input_v0 给:replay_buffer.py / trainer 整体设计讨论 / 读设计备忘]

---

## 九、最后一句话

这个 session 的核心成果不是修了 3 个 bug，是**验证了"调查→方案确认→TDD→commit"这套协作模式真的能避免失控**。

trainer 比 env 复杂得多，但只要保持这套模式，**风险是可控的**。

如果看到 Cursor 跳 gate、自由发挥、"顺手"重构——**立刻让用户打断**。这是接手后最重要的工作。
