# 交接文档 · EV 充电导航项目测试驱动阶段

> **写给下一个 Claude session**
> 日期：2026-05-12
> 上一个 session 的工作上下文，请按本文档"接手协议"开始，不要重读项目历史。

---

## 一、用户当前的决策（已锁定，不要再讨论）

用户名 **Eclipse**，华南理工 EI 一年级硕士，做 FedGRL 充电导航研究。

经历了 5 轮 Codex 失控后，用户已经做出以下决定，**请直接接受，不要再提"要不要重构"**：

1. **暂不重构项目代码**
2. **改走测试驱动路线**：给每个文件写单元测试，在写测试过程中发现并修复 bug
3. **当前正在第一个文件**：`env/entities.py` 的 EV 类
4. **暂不考虑新创新点**（reroute / 多步 MDP 等想法已经否决，作为未来 v2）

---

## 二、当前进度（精确到状态）

| 项 | 状态 |
|---|---|
| 重构计划 | ❌ 已废弃（不要复活） |
| t2 多步 MDP 方案 | ❌ 已废弃（不要复活） |
| 单步 hindsight contextual bandit | ⏸️ 暂缓（等测试完成后再说） |
| `tests/test_entities.py` Cursor 提示词 | ✅ 上一个 session 已发给用户 |
| Cursor 是否已执行提示词 | ❓ 未知，待用户回复 |
| git 备份状态 | ⚠️ 未确认（上一个 session 让用户做但没拿到 git log 回执） |

**最近一次给用户的输出**：`test_entities.py` 的 Cursor 两阶段提示词。
- Step 1: 读源码列字段
- Step 2: 写测试计划（停下来给用户审）
- Step 3: 写代码 + 跑测试

用户下一步可能的动作：把 Cursor 跑出来的 Step 2 计划贴回来。

---

## 三、接手协议（看完这一节就能开工）

### 如果用户贴了 Cursor 的 Step 2 测试计划

按这 3 件事审：

1. **测试计划有没有把 t2 相关字段当正常字段测**
   - EV 类上有 11 个 t2 遗留字段（详见下方"已知的代码现状"）
   - 这些字段是死代码，不应该花测试覆盖它们
   - 如果 Cursor 的计划里有 `test_t2_state` 之类的，告诉用户砍掉

2. **测试计划暴露的设计 smell 是不是真实问题**
   - Cursor 在 Step 2 应该列出"写测试时注意到的问题"
   - 这才是测试驱动的真正价值——比测试本身更重要
   - 看到 smell 不要立刻让用户改，先记下来，等更多测试暴露同类问题再判断

3. **测试数量是否合理**
   - 标准是 4-6 个测试
   - 超过 7 个 → 砍
   - 少于 3 个 → 让 Cursor 补
   - 1 个测试函数 > 20 行 → 拆

审完说"开始写"，让用户让 Cursor 进 Step 3。

### 如果用户贴了 Cursor 跑出来的测试结果

- **测试有 FAILED**：traceback 是最有价值的产出。诊断是"测试写错"还是"entities.py 有 bug"。**绝对不能让 Cursor 自己 fix**——它会改测试让通过，掩盖真实问题。
- **测试全 PASSED**：检查覆盖是否真实，会不会是 mock 太重导致"假绿"。检查完进入下一个文件 `env/charging_station.py`。

### 如果用户问"下一步怎么走"

按这个顺序推进测试（来自上一个 session 的规划）：

```
1. env/entities.py        ← 当前在这里
2. env/charging_station.py
3. env/power_grid_pp.py
4. env/base_env.py
5. env/real_env.py
6. agents/network.py
7. agents/dqn_base.py
8. agents/FederatedDQN.py
9. training/trainer.py
```

每个文件 3-5 个测试就够，不追求覆盖率。

---

## 四、必须遵守的工作纪律（这是 5 轮失控的教训）

这一节是最重要的。**违反任何一条都会让用户再次失控。**

### 纪律 1：Cursor 一次只动一个文件

如果 Cursor 在写 `test_entities.py` 时发现 `charging_station.py` 有问题，**记下来不要追**。
让用户在 `TESTING_TODO.md` 里写一笔，继续当前文件。

### 纪律 2：测试 FAILED 时，不许 Cursor 自己改

让 Cursor 停下来报告 traceback。
**严禁三件事**：
- 改 entities.py 让测试过
- 改测试让通过
- "顺手"重构

### 纪律 3：每天 commit 一次

用户的 git 状态可能还很乱（5 轮失控后没 commit 历史）。
任何时候用户说"我做完一步了"，提醒一句："git commit 一下"。

### 纪律 4：不要替用户做决定

用户已经踩过坑，比上一个 session 更知道自己想要什么。
**多用 `ask_user_input_v0` 工具，少自作主张**。

特别是这两类绝对不要替用户决定：
- 测试覆盖多深
- 测试发现的 bug 现在修还是延后

### 纪律 5：不要碰废弃方案

如果用户聊着聊着提到 t2 / 多步 MDP / reroute，**只回应不展开**。
不要写"我们之前讨论过…"。不要复盘历史。

---

## 五、已知的代码现状（上一个 session 从用户处拿到的事实）

### env/entities.py 的 EV 类

**字段总数 30+**，其中以下 **11 个是 t2 多步遗留死字段**，写测试时不需要覆盖：

```
t0_state, t0_action, t0_step, t2_step,
t2_decision_pending, travel_time_at_dispatch,
t2_action, t2_state, t2_pending_steps,
abandon_reason, just_abandoned_this_step
```

**核心活字段**（hindsight reward 需要）：

```
travel_time_h        # 累计行驶时间
wait_time_h          # 累计排队时间
total_fee_paid       # 累计费用
charge_sessions      # 完成的充电次数
```

⚠️ **关键事实**：EV 上还有一套 charging_station 写入的 **snapshot 字段**：

```
charge_fee_snapshot       # 这次充电的费用快照
charge_queue_time_h       # 这次充电的排队时长
charge_travel_time_h      # 这次充电的行驶时长
```

**两套字段的语义还没完全查清**，上一个 session 设计了 Cursor 调查提示词但**用户没用**，直接跳到测试路线。**reward 函数到底读哪一套是悬而未决的问题**——不要在这里替用户做决定。

### 其他文件的关键事实

- `env/base_env.py`: 多步遗留方法 `apply_t2_action`, `get_pending_decisions` 还在，需要在 base_env 测试时再处理
- `env/real_env.py`: 没调用 `super().__init__()`，自成一套初始化（隐患但不修）
- `agents/FederatedDQN.py`: 含 `trajectory_buffer` 字典等多步遗留结构
- `agents/network.py`: 含 `t2_advantage_fc` 双 head 结构遗留
- 所有 `tests/` 下旧脚本（step5-step17）将来要移到 `tests/experiments/`，**但不是现在**

### 用户的工作环境

- 系统：Windows，路径 `d:\car_charge`
- 工具：Cursor（主力 AI 编辑器）
- 网络：国内，PyPI 可能慢，已配置 Clash + 镜像源
- git 状态：上一个 session 已要求做备份 commit，但没拿到 `git log` 回执确认

---

## 六、用户性格 / 协作风格

来自记忆和当前对话的观察：

1. **偏好直接结论 + 可执行步骤**，不喜欢长篇解释
2. **多用 `ask_user_input_v0` 给选项**比文字问"你想怎么办"效率高 3 倍
3. **会自己改主意**——比如这次本来计划重构，聊着聊着改成测试驱动。**接受这一点，不要劝他回去**
4. **比上一个 session 的他更冷静**——他已经知道前面失控的根本原因（让 Codex 自由发挥），现在会主动控制 Cursor 的自由度
5. **偶尔会冒出"创新点"想法**（如这次的"每节点决策"）——认真分析，但**优先用时间预算和风险评估劝退**，除非他坚持

---

## 七、上一个 session 没解决但用户暂时不关心的问题

记下来供未来需要时拿出，**不要主动提**：

1. **reward 读 A 套还是 B 套字段**（charging_station 写入语义未查清）
2. **`should_request_charge_decision` 跟 `get_pending_decisions` 的关系**（语义可能重叠）
3. **`real_env.py` 不继承 `__init__` 的隐患**
4. **agents/network.py 的 t2_advantage_fc 双 head 结构清理**
5. **训练入口 step15_run / step17_run 等历史脚本归档**

---

## 八、下一个 session 的"开场动作建议"

如果用户上来说"继续"或没说清楚，按这个顺序问：

1. 上次让 Cursor 跑 test_entities 的 Step 2 计划，跑出来了吗？
2. 如果跑出来了，把计划贴给我审
3. 如果还没跑，问他是不是要重新生成提示词

**不要**主动复盘前面的讨论。
**不要**主动建议改方案。
**不要**主动写代码。

等用户给你具体输入再行动。

---

## 九、最后一句话

这个项目的核心问题从来不是技术，是**自由度控制**。

上一个 session 学到的东西：
- 多步 MDP 失败 ≠ 多步 MDP 错，是 Codex 自由发挥失败
- 重构计划失败 ≠ 重构错，是没 git 备份就动手失败
- 单步 hindsight 也不一定成功，如果让 Cursor 自由调 reward 同样会失败

**测试驱动这条路成功的唯一条件**：Cursor 严格按"读源码 → 给计划 → 等审核 → 写测试 → 报告结果"的两阶段流程，**任何一步都不许跳**。

如果你看到 Cursor 跳步，立刻让用户打断。这是你接手后最重要的工作。
