# Testing TODO (新增条目)

## 🔴 T2.6: BPR 拥堵效应失效 (重要,影响论文卖点)
- 现象: 最忙边 t_low == t_high == 0.11114h, 拥堵没增加行驶时间
- 影响: DQN 学不到"避开堵车", 论文"拥堵感知路由"卖点失效
- 处理时机: trainer 写完跑通后, v1 投稿前必须修
- 调查方向: BPR 公式系数 / cap 计算 / 流量统计是否真的喂给了 BPR

## 🟡 T3.2: LMP 缓存可能失效, OPF 占总时间 42%
- 现象: 20 step 内 get_lmp 调了 4 次, 每次 ~0.28s
- 影响: 训练时长增加约 50%, 当前可接受但浪费
- 处理时机: 性能成为瓶颈时, 或 v2 优化阶段
- 调查方向: lmp_update_interval 设置 / _cached_lmp 是否命中


## 🟢 EV 死字段清理 (合并任务)

### 范围
- 11 个 t0/t2 死字段:
    t0_state, t0_action, t0_step,
    t2_step, t2_decision_pending, travel_time_at_dispatch,
    t2_action, t2_state, t2_pending_steps,
    abandon_reason, just_abandoned_this_step
- 3 个统计死字段(子任务 2 附录调查发现):
    travel_steps, wait_steps, charge_steps
- 2 个内部缓存字段(子任务 2 Step 1 标记):
    _decision_state, _decision_snap

### 触发时机
trainer 跑通且收敛后, 确定真正在用的字段集合后, 反向清理

### 执行约束
- 必须全仓库 grep (env/ + agents/ + training/ + evaluation/ + tests/), 
  不能只看 env/
- 必须按 TDD 流程: 先写测试确认字段被引用次数为 0, 再删
- 不许"顺手"合并别的任务


## 🟡 仿真流量分布过低 (论文卖点相关, 非代码 bug)

### 现象
- EMA 路网 50 EV / 50 step, 最忙边只有 3 辆车经过
- 新 BPR 公式数学上正确, 但在如此低流量下增量仍 ≈ 0
- T2.6 测试 PASSED 但"无意义"(增量在浮点尾巴上)

### 影响
- 训练时若 BPR 增量始终 ≈ 0, DQN 学不到"避开拥堵"
- 论文"拥堵感知路由"卖点的实验证据可能不充分

### 不是 bug, 是实验设计问题
解决方案候选(等 trainer 跑通后决定):
- A. 增加 num_evs (50 → 200+), 让边自然出现拥堵
- B. 缩小路网到 sub-graph(~30 节点), 流量自然集中
- C. 调整出行 OD 分布, 让 EV 聚集去特定区域
- D. 增加 step_duration_h, 边占用时间更长

### 触发时机
trainer v1 跑通后, 看实际 episode 里:
  - peak_edge_flows_this_step 的分布
  - BPR 增量 / t0_h 比例分布
据数据决定是否要修, 怎么修