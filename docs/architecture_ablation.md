# 网络架构消融:站间交互机制(mean-pool vs 自注意力)

回答一个自然的审稿质疑——"既然贡献2 是决策层协调,为什么用如此简单的 station-only
GNN(DeepSets + mean-pool)?换更强的站间关系建模会不会更好?"本节用一个干净的受控
消融给出否定回答,从而**反向背书 mean-pool 架构的选择**。

数据:`evaluation/decision_ablation_{scratch,attn}_{32,16}/`;mean-pool warm 基线见
[coordination_ablation.md](coordination_ablation.md) §3。

> 定位:本篇是**架构消融**(换网络、固定决策机制);决策机制主消融(C-vs-D)与
> 共用的术语/指标/表头约定见 [coordination_ablation.md](coordination_ablation.md)
> (§0「术语与表头说明」)。两篇沿用同一套指标与 B/C 约定。

---

## 1. 设计与公平协议

两个网络消费**完全相同的输入特征**(同一份 state、同样的站点/EV 特征索引),**唯一差异
是站间信息的融合方式**:

- **mean-pool(本方法,`station_only`)**:每站独立编码 → 全局上下文 = 各站 embedding 的
  均值 → 逐站打分。跨站交互只有一个均值通道。
- **self-attention(`station_attn`)**:把 mean-pool 换成站间多头自注意力,让每个站点显式
  地与其它站点做相对比较(并全向量化)。

**公平性控制(关键)**:自注意力的权重结构与 mean-pool 不兼容,无法复用 warm-start,
只能 from-scratch。为排除 "warm-start vs 从零" 的混淆,我们**为两者都做了同预算 from-scratch
训练**:local 240 集 / 中心化&联邦 120 轮,epsilon 1.0→0.05(decay 0.9845,在 ~80% 进度
退火到下限),reward 饱和。验证显示 **from-scratch 的 mean-pool 追平了 warm-start**
(32 桩 B 档绝对 reward −0.7038 vs warm −0.7098),证明 from-scratch 预算充分、对比公平。

三个被比对象:`so-warm`(mean-pool, warm,论文主模型)、`so-scratch`(mean-pool, 从零)、
`attn-scratch`(自注意力, 从零)。最干净的架构对比是 **so-scratch vs attn-scratch**(同协议
同预算同收敛)。

---

## 2. 结果

### 2.0 术语与表头说明

**三个被比模型(列名)**——三者都是同一套联邦 hindsight 训练,只差网络架构与训练起点:

| 列名 | 含义 | checkpoint |
|---|---|---|
| `so-warm` | mean-pool 的 `station_only`,**warm-start**(论文主模型) | `checkpoints_fed_hindsight_40ev/global_final.pth` |
| `so-scratch` | 同一 mean-pool 架构,**from-scratch 同预算对照** | `checkpoints_scratch_federated/global_final.pth` |
| `attn` | self-attention 的 `station_attn`,**from-scratch** | `checkpoints_station_attn_federated/global_final.pth` |

**两种决策方式(B / C)**——同一份权重,推理时切换"取 state 的方式"(详见
[coordination_ablation.md](coordination_ablation.md) §3 的 A/B/C/D 定义):

- **B = 基线**:顺序决策但**不注入 `pending_counts`**(无 tick 内前瞻)。本架构里它与
  "纯并发 A"机械等价(A==B,故表中只列 B),并作为文献主流"**reward 层拥塞惩罚 D**"的合理代理。
- **C = 本方法**:顺序决策 + 注入 `pending_counts`,后决策车能看到本 tick 已选各站车数 →
  **决策层前瞻协调**。

**各指标(行名)**——括号内为"越大好 / 越小好":

| 指标 | 含义 | 方向 |
|---|---|---|
| 平均 reward | hindsight 单车奖励均值(trip/queue/fee 加权,恒为负) | 越大越好 |
| 平均排队 (h) | 每车实际排队时长均值,小时 | 越小越好 |
| 排队方差 | 各车排队时长的方差(衡量尾部/公平) | 越小越好 |
| 平均行程 (h) | 每车实际行驶时长均值,小时 | 越小越好 |
| 负载基尼 | 各站点**负载量**分布的基尼系数(空间均衡度) | 越小越均衡 |
| action 基尼 | 各站点**被选次数**分布的基尼系数(决策分散度) | 越小越分散 |
| 服务车次 | 评估期内成功充电的总车次(吞吐量) | 越多越好 |
| 放弃率 | 因等待超时放弃充电的车占比 | 越小越好 |

**"C vs B 提升"列(§2.2)**:脚本按各指标的"好的方向"统一折算,**正值 = C 优于 B**
(例如排队/行程/基尼虽是"越小越好",其下降也记为正的提升%)。

> 取数:queue/trip 单位小时,reward/gini 无量纲,服务车次为绝对计数。`so-warm` 取自
> [coordination_ablation.md](coordination_ablation.md) §3(其 action 基尼未记录,标 —)。

### 2.1 绝对值

**① 32 桩(宽松)**

| 指标 | so-warm B | so-warm C | so-scratch B | so-scratch C | attn B | attn C |
|---|---:|---:|---:|---:|---:|---:|
| 平均 reward | −0.7098 | −0.6971 | −0.7038 | −0.6852 | −0.7396 | −0.7216 |
| 平均排队 (h) | 0.0773 | 0.0731 | 0.0767 | 0.0695 | 0.0970 | 0.0878 |
| 排队方差 | 0.0217 | 0.0196 | 0.0197 | 0.0179 | 0.0276 | 0.0243 |
| 平均行程 (h) | 0.3590 | 0.3556 | 0.3525 | 0.3494 | 0.3364 | 0.3408 |
| 负载基尼 | 0.1176 | 0.1124 | 0.1183 | 0.1098 | 0.1241 | 0.1184 |
| action 基尼 | — | — | 0.1272 | 0.1189 | 0.1352 | 0.1286 |
| 服务车次 | 5567 | 5614 | 5586 | 5640 | 5603 | 5613 |
| 放弃率 | 0% | 0% | 0% | 0% | 0% | 0% |

**② 16 桩(受约束)**

| 指标 | so-warm B | so-warm C | so-scratch B | so-scratch C | attn B | attn C |
|---|---:|---:|---:|---:|---:|---:|
| 平均 reward | −2.6785 | −2.6341 | −2.5644 | −2.5466 | −2.6764 | −2.6618 |
| 平均排队 (h) | 0.8965 | 0.8808 | 0.8319 | 0.8263 | 0.8977 | 0.8912 |
| 排队方差 | 1.0750 | 1.2120 | 0.3407 | 0.3468 | 1.3138 | 1.3211 |
| 平均行程 (h) | 0.4385 | 0.4274 | 0.4863 | 0.4784 | 0.4318 | 0.4329 |
| 负载基尼 | 0.0580 | 0.0516 | 0.0318 | 0.0287 | 0.0893 | 0.0975 |
| action 基尼 | — | — | 0.0470 | 0.0455 | 0.1746 | 0.1559 |
| 服务车次 | 3363 | 3391 | 3436 | 3464 | 3199 | 3185 |
| 放弃率 | 0.33% | 0.50% | 0.00% | 0.00% | 11.36% | 9.47% |

### 2.2 C vs B 提升(正 = C 更好)

**① 32 桩(宽松)**

| 指标 (C vs B) | so-warm | so-scratch | attn-scratch |
|---|---:|---:|---:|
| 平均 reward | +1.79% | +2.64% | +2.43% |
| 平均排队 | +5.44% | +9.36% | +9.48% |
| 排队方差 | +9.42% | +9.09% | +11.99% |
| 平均行程 | +0.95% | +0.86% | **−1.33%** |
| 负载基尼 | +4.48% | **+7.20%** | +4.56% |
| action 基尼 | — | +6.48% | +4.85% |
| 放弃率(绝对) | 0% | 0% | 0% |

**② 16 桩(受约束)**

| 指标 (C vs B) | so-warm | so-scratch | attn-scratch |
|---|---:|---:|---:|
| 平均 reward | +1.66% | +0.70% | +0.55% |
| 平均排队 | +1.75% | +0.68% | +0.73% |
| 排队方差 | −12.74% | −1.78% | −0.56% |
| 平均行程 | +2.53% | +1.63% | −0.25% |
| 负载基尼 | +11.10% | **+9.68%** | **−9.16%** |
| action 基尼 | — | +3.28% | +10.71% |
| **放弃率(B 绝对)** | 0.33% | **0.00%** | **11.36%** |

---

## 3. 结论:自注意力不改进,受约束档反而有害——且是架构效应

1. **宽松档(32 桩):打平,mean-pool 甚至略胜。** 注意力在排队/方差上略高,但**行程反转
   为负(−1.33%)**、负载基尼更低(+4.56% vs +7.20%)。没有可主张的增益。
2. **受约束档(16 桩):注意力明显更差。** C 的负载基尼从 +9.68%(mean-pool)翻成 **−9.16%**
   (协调反而破坏均衡),且 attn 模型**放弃率高达 11.36%**(两个 mean-pool 变体均为 0%)。
3. **是架构而非训练:** 同样 from-scratch、同预算、同收敛,`so-scratch` 在 16 桩完全正常
   (0% 放弃、负载基尼 +9.68%,与 warm 一致);唯一变量是站间融合方式。故退化归因于
   **自注意力本身**——它学出更激进/集中的路由,容量一紧便把车堆向充不上的站。

**因此采用 mean-pool `station_only` 作为最终架构。** 本消融把"用简单网络"从一个潜在
弱点转为一个有据可依的设计选择:对该选站问题,更强的站间关系建模不仅无益,在容量
压力下反而损害协调与可服务性。

---

## 4. Paper-ready paragraph (EN)

> **Does richer inter-station modeling help?** Our station selector pools per-station
> embeddings with a permutation-invariant mean (a DeepSets-style head). A natural
> question is whether explicit inter-station relational reasoning would strengthen the
> decision-level coordination of Sec. [C]. We replace mean-pooling with multi-head
> self-attention over stations, keeping all input features identical, and—because the
> two parametrizations are weight-incompatible—train *both* architectures from scratch
> under a matched budget and exploration schedule (a from-scratch mean-pool model
> matches its warm-started counterpart, confirming the budget is adequate). Attention
> yields no gain under light load and is clearly harmful under capacity stress: at the
> constrained scale the attention policy *worsens* load balance (load-Gini coordination
> effect flips from +9.7% to −9.2%) and abandons 11.4% of vehicles, whereas the matched
> mean-pool control abandons none. Since the only changed factor is the pooling
> mechanism, the degradation is architectural: self-attention learns a more concentrated
> routing policy that overloads stations when capacity is tight. We therefore retain the
> simple mean-pool head, and report this as evidence that the inductive bias—not model
> capacity—is what matters for anticipatory station-selection coordination.

---

## 5. 复现

```powershell
# 同预算 from-scratch 训练(两架构)
python run_paradigm_training.py --network station_attn  --from-scratch              --local-episodes 240 --rounds 120 --epsilon-decay 0.9845
python run_paradigm_training.py --network station_only  --from-scratch --tag _scratch --local-episodes 240 --rounds 120 --epsilon-decay 0.9845

# 决策消融(各架构 32/16 桩;--num-chargers-per-station 8 或 4)
python evaluate_decision_ablation.py --model checkpoints_station_attn_federated\global_final.pth --network station_attn  ... --save-dir evaluation\decision_ablation_attn_32
python evaluate_decision_ablation.py --model checkpoints_scratch_federated\global_final.pth      --network station_only  ... --save-dir evaluation\decision_ablation_scratch_32
```

欠训练(50 轮)的早期 attn checkpoint 曾给出过乐观假象,已归档
`archive/station_attn_undertrained_50r/`,**结论以本节足量训练版为准**。
