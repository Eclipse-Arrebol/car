# 协调机制消融实验记录

本文档记录围绕"充电导航选站的竞争协调机制"做的三个评估实验:它们想回答什么、
**怎么做的(方法学)**、结果如何、以及结论。三个实验共用同一套公平对照框架。

- 模型:`checkpoints_fed_hindsight_40ev/global_final.pth`(联邦 hindsight 训练,
  40 EV / 4 站 / 每站 8 桩,station-only GNN)
- 场景:`old_city`,UE scale 1.3,40 EV
- 评估规模:10 集 × 144 步,ε=0(确定性 argmax),seed 42
- **所有实验都不重训、不改网络结构**,只在推理时切换"取 state 的方式"

---

## 0. 共用的公平对照框架

三个实验的核心方法是同一个:**用同一份训练好的权重,在推理时切换决策机制,
其余全部固定**,确保指标差异只来自被测变量。

- 同权重:全程加载同一个 checkpoint,不重训。
- 同场景同种子:每集用 `seed + ep` 重建环境,对照组之间初值逐位一致。
- 同指标定义:trip/queue/fee/reward、负载基尼、动作基尼、电网电压等指标的
  计算函数复用同一份代码([evaluate_oracle_wait.py](../evaluate_oracle_wait.py) 内的
  `_mean/_variance/_std/_gini/_pct_change`)。
- 唯一变量:`env.get_graph_state_for_ev(...)` 取状态时,注入什么信息。

奖励口径(训练与评估一致),[trainer/trainer.py](../trainer/trainer.py):

```python
reward = -(0.4 * trip / 0.58 + 0.4 * queue / 0.17 + 0.2 * fee / 65)
```

注意:**该 hindsight 单车奖励已内含拥塞惩罚**(中间项 `0.4·queue/NORM_WAIT`
是真实排队时长)。env.step 里的标量 reward(含 `10·queue_cost`、电压惩罚等)
**不参与训练**,agent 只学这个单车奖励。

### 术语与表头说明

**指标(三个实验共用,括号内为方向)**:

| 指标 | 含义 | 方向 |
|---|---|---|
| 平均 reward | hindsight 单车奖励均值(trip/queue/fee 加权,恒为负) | 越大越好 |
| 平均排队 (h) | 每车实际排队时长均值,小时 | 越小越好 |
| 排队方差 | 各车排队时长的方差(衡量尾部/公平) | 越小越好 |
| 平均 trip / 行程 (h) | 每车实际行驶时长均值,小时 | 越小越好 |
| 负载基尼 | 各站点**负载量**分布的基尼系数(空间均衡度) | 越小越均衡 |
| 动作 / action 基尼 | 各站点**被选次数**分布的基尼系数(决策分散度) | 越小越分散 |
| 服务车次 | 评估期内成功充电的总车次(吞吐量) | 越多越好 |
| 放弃率 | 因等待超时放弃充电的车占比 | 越小越好 |

**各实验结果表的列名约定**:

- **实验1(oracle,§1)**:`baseline` = 用真实 state 的同一模型;`oracle` = 把候选站
  的等待特征就地替换为作弊真值后的**同一模型**;`提升` = oracle 相对 baseline 按
  "好的方向"折算的百分比(正 = oracle 更好)。
- **实验2(拥塞扫描,§2)**:`档位(总桩)` = 4 站 × 每站桩数;`基线排队(分钟)` /
  `基线放弃率` = 该档 baseline 的绝对值;`reward / 排队均值 / 排队方差 提升%` =
  该档 oracle vs baseline 的提升(正 = oracle 更好)。
- **实验3(决策消融,§3)**:`A=B(≈D)` = 基线(顺序无前瞻,作为文献"reward 层惩罚
  D"的代理,A 与 B 机械等价见 §3);`C(本方法)` = 决策层前瞻;`C vs B` = 按"好的
  方向"折算的提升%(服务车次列直接给车次差)。**正值 = C 更好**(排队/行程/基尼虽
  "越小越好",其下降也记为正提升)。

> 同口径的架构消融(mean-pool vs 自注意力)见 [architecture_ablation.md](architecture_ablation.md);
> 本篇是 C-vs-D 决策机制主消融,那篇换网络架构、沿用同一套指标与 C/B 约定。

---

## 1. Oracle 理想到达等待(时间错峰上界实验)

### 问题
当前协调信号(`pending_counts`、mean-field `evs_heading_to`)都是"人头数",
不含时间维度——两辆都驶向 A 站的车,一辆充 5 分钟一辆充 50 分钟,产生相同信号。
系统能"空间分流"但不能"时间错峰"。**值不值得补动态占用建模?** 用一个作弊版
理想等待估计测上界:若几乎没提升,就不值得做。

### 方法
新增三个**纯附加**方法,[env/base_env.py](../env/base_env.py):

- `_ev_eta_to_station_hours(ev, station)`:该车到站的真实剩余行驶时间。在途车
  (MOVING_TO_CHARGE)按"当前边剩余时间 + 沿剩余 path 累加 `get_edge_travel_profile`"
  逆推 `EV.move()`;未上路的候选车按最短路估计(与派发一致)。
- `_oracle_arrival_wait_hours(ev, station)`:**多服务台 FIFO 占用模拟**。
  `num_chargers` 个服务台,当前在桩车按各自剩余充电时间释放;队列车 + 所有
  "比候选更早到"的在途车,按到达时间贪心占用最早空出的台;算出候选车到达
  那一刻真正要等多久。纯真值、确定性,仅评估用(允许读全局真值)。
- `get_graph_state_for_ev_oracle(...)`:调原状态函数后,把每个候选站点的
  feat 11(service_time)与 feat 15(queue_wait_ratio)就地替换为 oracle 等待。
  **归一化方式、输入维度 19、权重、结构均不变。**

> 关键确认:oracle 所需真值(各车 ETA、充电时长、在桩剩余时间)全部能从现有
> 状态算出,无需造数据。充电时长用现成的 `estimate_charge_time_hours`(CC-CV)。

脚本:[evaluate_oracle_wait.py](../evaluate_oracle_wait.py),两模式 baseline / oracle
唯一区别是 model 取 state 调 `get_graph_state_for_ev` 还是 `..._oracle`。

复现:

```powershell
python evaluate_oracle_wait.py --model checkpoints_fed_hindsight_40ev\global_final.pth `
  --episodes 10 --steps-per-episode 144 --num-evs 40 --num-stations 4 `
  --num-chargers-per-station 8 --grid-variant old_city --ue-scale 1.3 `
  --no-use-action-mask --epsilon 0 --seed 42 --save-dir evaluation\oracle_wait_old_city
```

### 结果(old_city,32 桩,5614 次充电事件)

| 指标 | baseline | oracle | 提升 |
|---|---:|---:|---:|
| 平均排队 (h) | 0.0731 | 0.0669 | +8.55% |
| 排队方差 | 0.0196 | 0.0166 | +15.35% |
| 平均 reward | −0.6971 | −0.6855 | +1.66% |
| 平均 trip (h) | 0.3556 | 0.3587 | −0.86% |
| 负载基尼 | 0.1124 | 0.1180 | −4.99% |

### 结论
平均排队仅 4.4 分钟,系统不受排队约束。即使作弊用未来真值,reward 上界也只
+1.66%,且靠牺牲 trip / 负载均匀度换取。**当前规模下动态占用建模不值得做**——
但需用拥塞扫描确认上界是否随负载放大(见实验 2)。

数据:`evaluation/oracle_wait_old_city/oracle_wait_evaluation.json`

---

## 2. 竞争强度扫描(拥塞梯度)

### 问题
验证假设:"时间错峰的价值只在系统真正受排队约束时才放大。"

### 方法
固定 40 EV / 4 站,**只扫描每站桩数**(总桩 32→16→8→4),制造从宽松到饱和的
拥塞梯度,每档跑 baseline vs oracle。固定站数=4 是为了**单变量**:不改动作空间
维度与站点地理(checkpoint 是 4 站训练的),总桩=32 档正好等于实验 1 的锚点。

外层包装脚本 [run_congestion_sweep.py](../run_congestion_sweep.py) 循环调用实验 1 的
`run_mode`,**单点对照逻辑一字未改**;额外汇总每档的绝对排队(分钟)、放弃率,
以及 oracle 的 reward / 排队均值 / 排队方差提升%。

复现:

```powershell
python run_congestion_sweep.py --episodes 10 --steps-per-episode 144 `
  --seed 42 --save-dir evaluation\congestion_sweep
```

### 结果

| 档位(总桩) | 基线排队(分钟) | 基线放弃率 | reward 提升% | 排队均值改善% | 排队方差改善% |
|---|---:|---:|---:|---:|---:|
| 4×8 (=32) 宽松 | 4.4 | 0.0% | +1.66% | +8.55% | +15.35% |
| 4×4 (=16) | 52.8 | 0.5% | +0.28% | +1.16% | +24.84% |
| 4×2 (=8) ⚠️ | 131.4 | 14.1% | −2.31% | −2.38% | −0.87% |
| 4×1 (=4) ⚠️ | 170.8 | 48.6% | −0.93% | −0.99% | +5.52% |

⚠️ 放弃率 >10% 的档位:`avg_queue` 仅统计充上电的车(幸存者偏差),`max_queue`
钉在 4h 超时上限,排队指标失真,反映"谁进得去"而非真排队改善。

### 结论
**假设被证伪。** oracle 的 reward 提升不随拥堵单调上升,反而在最宽松档最高
(+1.66%),随拥堵一路转负。三段式解读:
1. 宽松档:提升最大但绝对排队仅 4 分钟,无实际意义。
2. 真正受约束档(16 桩):reward 提升塌到 +0.28%——时间错峰只是把成本从"排队"
   挪到"行驶"(trip −4.8%),净目标几乎不动。
3. 饱和档(8/4 桩):瓶颈是**总容量不足**,所有站堵死、无错峰空间,oracle 转负。

瓶颈是充电容量,该投扩容/选址,不是调度智能。

数据:`evaluation/congestion_sweep/congestion_sweep.json`

---

## 3. A/B/C 决策机制消融(贡献2)

### 问题
证明本文的"决策层空间分流"(C)优于文献主流的"reward 层事后拥塞惩罚"(D)。

### 四个范式
- **A 纯并发**:同 tick 全部车基于冻结状态独立决策,不更新 `pending_counts`。
- **B 顺序无耦合**:顺序决策,但 state 不含 `pending_counts`(传 None)。
- **C 本方法**:顺序 + `pending_counts` 注入 state,后决策车看到本 tick 前面已选
  各站的车数。
- **D reward 层惩罚**:决策时不前瞻,把拥塞作为 reward 惩罚项。**D 的区别在
  训练目标,不在推理,严格需重训。**

### 方法与一个关键论证
脚本 [evaluate_decision_ablation.py](../evaluate_decision_ablation.py),A/B/C 唯一区别
是取 state 时用不用 `pending_counts`(C 在决策循环内累加并注入;A/B 传 None)。

**为何 B 可作为 D 的代理(免重训)**:
1. 本架构里一个 tick 的决策先收集、再统一 `env.step`,决策循环中 env 冻结,
   **唯一的 tick 内耦合通道就是 `pending_counts`**。故关掉前瞻后 **A 与 B 机械
   等价**——实验确认两档下 A、B 动作分布与所有指标**逐位相同**。
2. 现有训练奖励已内含基于真实排队的拥塞惩罚(比 D 的占用计数代理更直接),
   故"基线模型 + B 决策方式"是 D 语义的合理近似。

> **决策记录(2026-06-14):** 经评估后决定**不重训精确 D**,采用 B 作为 D 的
> 合理代理,将贡献2 如实写成"温和的设计优势 + 边界讨论",省去 ~5h 重训。

复现(两个 config):

```powershell
# 32 桩(宽松,标准 config)
python evaluate_decision_ablation.py --model checkpoints_fed_hindsight_40ev\global_final.pth `
  --episodes 10 --steps-per-episode 144 --num-stations 4 --num-chargers-per-station 8 `
  --grid-variant old_city --ue-scale 1.3 --no-use-action-mask --epsilon 0 --seed 42 `
  --save-dir evaluation\decision_ablation_32

# 16 桩(受约束,tick 内竞争最密)
python evaluate_decision_ablation.py --model checkpoints_fed_hindsight_40ev\global_final.pth `
  --episodes 10 --steps-per-episode 144 --num-stations 4 --num-chargers-per-station 4 `
  --grid-variant old_city --ue-scale 1.3 --no-use-action-mask --epsilon 0 --seed 42 `
  --save-dir evaluation\decision_ablation_16
```

### 结果(A == B,故只列 C vs B)

**① 32 桩(宽松,排队 ~4.6 分钟)**

| 指标 | A=B(≈D) | C(本方法) | C vs B |
|---|---:|---:|---:|
| 平均 reward | −0.7098 | −0.6971 | +1.79% |
| 平均排队 (h) | 0.0773 | 0.0731 | +5.44% |
| 排队方差 | 0.0217 | 0.0196 | +9.42% |
| 平均 trip (h) | 0.3590 | 0.3556 | +0.95% |
| 负载基尼 | 0.1176 | 0.1124 | +4.48% |
| 服务车次 | 5567 | 5614 | +47 |

**② 16 桩(受约束,排队 ~54 分钟,放弃 <0.5%)**

| 指标 | A=B(≈D) | C(本方法) | C vs B |
|---|---:|---:|---:|
| 平均 reward | −2.6785 | −2.6341 | +1.66% |
| 平均排队 (h) | 0.8965 | 0.8808 | +1.75% |
| 平均 trip (h) | 0.4385 | 0.4274 | +2.53% |
| 负载基尼 | 0.0580 | 0.0516 | +11.10% |
| 服务车次 | 3363 | 3391 | +28 |
| 排队方差 | 1.075 | 1.212 | −12.74%(C 更差) |
| 放弃率 | 0.33% | 0.50% | −53%(C 更差) |

### 结论(温和的设计优势 + 边界讨论)
C 一致地、温和地赢 B(≈D):reward **+1.79%(宽松)/ +1.66%(受约束)**,
伴随更均匀负载、更短行程、更高吞吐。机制符合预期:决策层前瞻打破 tick 内同时
决策的对称性,后决策车主动避开本 tick 已选站点;reward 层惩罚只能教"平均避堵",
无法协调具体 tick。

诚实的三点边界:
1. 提升温和且**不随拥堵放大**(+1.79%→+1.66%),与实验 2 一致。
2. **非帕累托占优**:受约束档 C 改善 reward/吞吐/负载均匀度,但排队方差更大、
   放弃略多——C 是更激进的空间分流,把成本由均值转向尾部。
3. **效应量为上界**:C 的 checkpoint 带 `pending_counts` 训练,B 模式将其训练时
   依赖的特征位置零,存在 train/test 不匹配;故 +1.7% 含"特征剥夺"成分,纯机制
   效应应略小于此。本贡献定位为**一项温和但一致的设计优势**,非数量级改进。

数据:`evaluation/decision_ablation_32/`、`evaluation/decision_ablation_16/`

---

## 文件索引

| 文件 | 作用 |
|---|---|
| [env/base_env.py](../env/base_env.py) | 新增 oracle 三方法(`_ev_eta_to_station_hours` / `_oracle_arrival_wait_hours` / `get_graph_state_for_ev_oracle`),纯附加 |
| [evaluate_oracle_wait.py](../evaluate_oracle_wait.py) | 实验 1:baseline vs oracle |
| [run_congestion_sweep.py](../run_congestion_sweep.py) | 实验 2:拥塞梯度扫描(外层包装) |
| [evaluate_decision_ablation.py](../evaluate_decision_ablation.py) | 实验 3:A/B/C 决策机制消融 |
| `evaluation/oracle_wait_old_city/` | 实验 1 数据 |
| `evaluation/congestion_sweep/` | 实验 2 数据 |
| `evaluation/decision_ablation_32/`、`_16/` | 实验 3 数据 |
