# 从当前代码提取的论文方法公式

本文件只保留当前工作区可直接支持的数学模型。每节均给出代码、变量、单位和实现程度。`paper_draft_cn_v3.md` 当前缺失不影响本文件的代码提取，但影响逐句论文校对。

## 3 交通—配电网耦合系统模型

### 3.1 交通网络与背景流（当前不是 UE）

当前背景流空间基线为边端点启发式函数。令节点分数归一化值为

$$
\bar s_v=\frac{s_v-s_{\min}}{s_{\max}-s_{\min}},
$$

则代码等价于

$$
h_{uv}=0.5+2.5\left(1-\left|\frac{\bar s_u+\bar s_v}{2}-0.5\right|\right)
\left(0.5+0.5|\bar s_u-\bar s_v|\right).
$$

日变化背景量为

$$
x^{\mathrm{bg}}_{e,t}=h_e d_t
\left[1+0.05\sin\left(\frac{2\pi(t\bmod144)}{144}\right)\right],
$$

其中 $d_t$ 是由 8:00、17:30 两个高斯峰和午间宽峰组成、再按日最大值归一化至 $(0,1]$ 的曲线。

- 代码：`env/background_traffic.py:12-25,28-61`；`env/base_env.py:403-409`。
- 变量：`background_edge_base_flows`、`background_daily_profile`、`background_edge_flows`。
- 单位：$h_e$ 和 $x^{bg}$ 没有被代码赋予可靠交通流单位，按后续 BPR 用法近似为边上并发车辆量。
- 实现程度：完全对应当前可复现执行。

不能写 UE 的 Wardrop、Beckmann 或 Frank–Wolfe 公式。虽然 `build_base_background_flows()` 有可选导入分支，当前 `env.background.ue_assignment` 不可导入，执行会回退上式（`env/background_traffic.py:83-105`）。TNTP OD 文件存在并声明 74 zones、总流 65576.375431，但当前执行不解析 OD。

### 3.2 BPR 道路时间

自由流时间：

$$
t_e^0=\frac{\ell_e/1000}{v_e},
$$

其中 $\ell_e$ 为 m，$v_e$ 为 km/h，$t_e^0$ 为 h。

受控 EV 边占用为

$$
n^{\mathrm{EV}}_{e,t}=n_{(u,v),t}+n_{(v,u),t}.
$$

对候选路径估计或车辆即将进入边时，实际输入量是

$$
x_{e,t}=x^{\mathrm{bg}}_{e,t}+n^{\mathrm{EV}}_{e,t}+1.
$$

代码把容量换算为自由流穿越期间可容纳的并发车辆数：

$$
\widetilde C_e=\max\{1,C_e t_e^0\},
$$

并计算

$$
t_{e,t}=\max\left\{10^{-6},\;t_e^0
\left[1+0.15\left(\frac{x_{e,t}}{\widetilde C_e}\right)^4\right]\right\}.
$$

路径预计时间和距离为

$$
\widehat T_{i,s,t}=\sum_{e\in P^*_{i,s,t}}t_{e,t},\qquad
D_{i,s,t}=\sum_{e\in P^*_{i,s,t}}\ell_e/1000,
$$

其中 $P^*$ 是以动态 BPR 时间为权重的 NetworkX 最短路。

- 代码：`env/base_env.py:366-439,483-508,888-904`。
- 变量：`length_m`、`speed_kph`、`capacity_vehph`、`edge_active_counts`、`trip_time_h`、`trip_dist_km`。
- 单位：m、km/h、veh/h、近似并发车辆数、h、km。
- 实现程度：公式完全对应代码；但背景量单位未校准，因此物理量纲只对受控 EV 和 $C_et_e^0$ 部分自洽。

### 3.3 EV 行驶能耗与 SOC

代码意图采用固定单位里程能耗：

$$
E^{\mathrm{drive}}_{i,t}=0.18D_{i,t}\quad[\mathrm{kWh}],
$$

$$
SOC_{i,t+1}=\max\left\{0,SOC_{i,t}-
100\frac{E^{\mathrm{drive}}_{i,t}}{60}\right\}\quad[\%].
$$

充电请求条件：

$$
SOC_i<30\%,
$$

目标 SOC 为 $95\%$。

- 代码：`env/entities.py:9-15,66-108`；`env/base_env.py:51,205-226`。
- 变量：`drive_kwh_per_km=0.18`、`battery_capacity_kwh=60`、`soc`、`charge_trigger_soc=30`、`target_soc=95`。
- 单位：kWh/km、kWh、百分数。
- 实现程度：固定里程耗电关系已实现，但 `EV.move()` 的 $D_{i,t}$ 只用最后/当前边近似；一个时步跨多边时不能准确累计。论文须标“离散近似”。此外 `IDLE` 每 10 min 固定 SOC−0.5 且随机跳邻居（`env/base_env.py:879-923`），不属于上式。

### 3.4 排队与充电桩服务

实际队列是离散时间 FCFS：

$$
q_s(t^+)=q_s(t)\oplus i
$$

表示车辆到站后 append 队尾；每个时隙站点把队首车辆依次分配给空闲桩：

$$
i=\operatorname{popFront}(q_s),\qquad
|\mathcal C_s|<M_s.
$$

实际等待时间由离散步累计：

$$
W_i=N_i^{\mathrm{wait}}\Delta t,\qquad \Delta t=\frac16\ \mathrm h,
$$

并在 $W_i\ge4$ h 时放弃。

状态中的预计等待不是多服务台空闲时刻公式。令

$$
L_s=|q_s|+z_s,
$$

其中 $z_s$ 为当前时隙前序提交数。代码计算

$$
\widehat W_s=
\begin{cases}
0,&L_s\le0,\\
\min_{i\in\mathcal C_s}\frac{E_i^{\rm rem}}{\eta_i p_i}
+\dfrac{L_s}{M_s}\overline T_s^{\rm ch},&L_s>0,
\end{cases}
$$

没有在充车辆时第一项按 0 处理。

- 代码：`env/base_env.py:239-249,954-974`；`env/charging_station.py:52-86,195-206,251-255`。
- 变量：`queue`、`connected_evs`、`num_chargers`、`wait_time_h`、`incoming_count`。
- 单位：车辆数、h。
- 实现程度：离散 FCFS 完全实现；预计等待为启发式近似。不能写成维护每桩 $b_{s,k}$ 的连续事件模型。

### 3.5 分段近似 CC-CV 充电

单桩额定功率 $P^{\max}=60$ kW，CC/CV 分界 $SOC^{cc}=80\%$，CV 下限 $P^{floor}=1$ kW，目标 $SOC^*=95\%$：

$$
P_i^{\rm req}(SOC)=
\begin{cases}
60,&SOC<80,\\
\max\left\{1,60\dfrac{95-SOC}{95-80}\right\},&80\le SOC<95.
\end{cases}
$$

站级上限 $P_s^{grid}=600$ kW：

$$
\rho_s(t)=\min\left\{1,
\frac{600}{\sum_{i\in\mathcal C_s}P_i^{\rm req}(t)}\right\},
\qquad
P_i^{\rm ch}(t)=\rho_s(t)P_i^{\rm req}(t).
$$

SOC 离散更新：

$$
SOC_i(t+1)=\min\left\{100,
SOC_i(t)+100\frac{0.92P_i^{\rm ch}(t)(1/6)}{60}\right\}.
$$

当 $SOC_i\ge95\%$ 时结束。

- 代码：`env/charging_station.py:6-14,133-171,208-255`；`env/entities.py:10-12`。
- 变量：`max_charger_power`、`max_grid_power`、`_CC_CV_THRESHOLD`、`_CV_FLOOR_KW`、`charge_efficiency`。
- 单位：kW、h、kWh、百分数。
- 实现程度：分段离散近似完全实现；无电池电压/电流状态、无独立车辆接收上限、无配电网反馈限功率。

### 3.6 pandapower 交流潮流

代码在 pandapower 中构造平衡三相等值 IEEE33 网络并调用 AC `runpp`。可在论文中写标准节点功率平衡：

$$
P_i=V_i\sum_jV_j\left(G_{ij}\cos\theta_{ij}+B_{ij}\sin\theta_{ij}\right),
$$

$$
Q_i=V_i\sum_jV_j\left(G_{ij}\sin\theta_{ij}-B_{ij}\cos\theta_{ij}\right),
$$

但必须说明方程由 `pandapower.runpp` 求解，而不是项目手写求解器。

站点 EV 负荷映射：

$$
P_{b,t}^{EV}[\mathrm{MW}]=\frac1{1000}
\sum_{s:\phi(s)=b}P_{s,t}^{ch}[\mathrm{kW}],
\qquad Q_{b,t}^{EV}=0.
$$

总节点负荷为代码表内基础 $P_b^0,Q_b^0$ 加 EV 负荷。电压审计集合：

$$
\mathcal V_t=\{b:V_{b,t}<0.95\ \text{or}\ V_{b,t}>1.05\}.
$$

有功线损：

$$
P_t^{loss}=1000\sum_{\ell} \texttt{res\_line.pl\_mw}_\ell\quad[\mathrm{kW}].
$$

- 代码：`env/power_grid_pp.py:21-93,101-194,196-252`；`env/real_env.py:171-191`。
- 网络：33 buses、32 lines、0 transformers；12.66 kV；站点 0–3 映射 buses 6/9/12/18。
- 单位：MW/Mvar、pu、kW。
- 实现程度：AC 潮流真实调用。未读取线路负载率、变压器结果或无功损耗；电压阈值只用于统计，不是优化约束。

### 3.7 条件式 OPF 电价与费用

每 6 个仿真步（约 1 h）代码尝试 `pandapower.runopp`，并取

$$
\lambda_{b,t}=\frac{\texttt{res\_bus.lam\_p}_{b,t}}{1000}.
$$

若 OPF 成功：

$$
\lambda^{energy}_{s,t}=\max\{0.1,\lambda_{\phi(s),t}(1+\xi_t)\}.
$$

若失败：

$$
\lambda^{energy}_{s,t}=\max\{0.1,
1.0\,m_t^{ToU}(1+\xi_t)\}.
$$

其中 $\xi_t\sim U[-0.1,0.1]$ 且全站共享。面向用户的当前价：

$$
\lambda^{user}_{s,t}=\max\{0.1,
\lambda^{energy}_{s,t}+0.15+0.08(|q_s|+|\mathcal C_s|)\}.
$$

状态预计费用：

$$
\widehat F_{i,s,t}=\lambda^{user}_{s,t}
\frac{(SOC_i^*-SOC_i)B_i}{100\eta_i}.
$$

训练奖励所用费用快照：

$$
F_i^{start}=\lambda^{energy}_{s,t_i^{start}}
\frac{(SOC_i^*-SOC_i(t_i^{start}))B_i}{100\eta_i}.
$$

代码另累计但不用于当前 reward 的实际费用：

$$
F_i^{actual}=\sum_t\lambda^{user}_{s,t}P_{i,t}^{ch}\Delta t.
$$

- 代码：`env/power_grid_pp.py:299-309`；`env/base_env.py:983-1008`；`env/charging_station.py:37-50,117-123,181-187,213-222`。
- 单位：代码意图为货币/kWh和货币；币种没有在核心环境统一声明，不能仅凭变量名确定 CNY/EUR。
- 实现程度：三种费用公式分别完全对应；OPF 电价是条件分支，失败静默回退，不能无条件称 DLMP。ToU 实现把 10 分钟 step 当小时，现状不宜给出标准分时段公式。

### 3.8 状态、动作、奖励与目标异构

完整图状态：

$$
X_{i,t}\in\mathbb R^{N\times19},\qquad
E\in\mathbb N^{2\times 2|\mathcal E|},\qquad
A_e\in\mathbb R^{2|\mathcal E|\times2}.
$$

当前 EMA 图 $N=74,|\mathcal E|=129$，所以 `x:[74,19]`、`edge_index:[2,258]`、`edge_attr:[258,2]`。完整特征定义见 `code_model_audit.md` 第 11 节。

默认策略实际输入是每站

$$
f_{i,s,t}=[\lambda^{user},\,P_s/600,\,
\widehat T^{trip},\,\widehat T^{service},\,
\min(2,\widehat W/4),\,
(M_s-|\mathcal C_s|)/M_s,\,h_s]^{\top}\in\mathbb R^7,
$$

以及 EV 特征

$$
e_i=[SOC_i/100]\in\mathbb R.
$$

动作：

$$
a_{i,t}\in\{0,\ldots,S-1\}.
$$

训练奖励：

$$
r_i=-\left(
w_T\frac{T_i^{actual}}{0.58}
+w_W\frac{W_i^{actual}}{0.17}
+w_F\frac{F_i^{start}}{65}
\right).
$$

默认 $(w_T,w_W,w_F)=(0.4,0.4,0.2)$；异构目标开启时：

$$
w^{old}=(0.15,0.70,0.15),\quad
w^{new}=(0.15,0.15,0.70),\quad
w^{suburb}=(0.70,0.15,0.15).
$$

- 代码：`env/base_env.py:595-661`；`agents/network_station_only.py:20-30`；`trainer/trainer.py:6-17,77-93`；`reward_profiles.py:18-35`。
- 单位：时间 h，费用为代码内部货币值，SOC 比例。
- 实现程度：完全实现。0.58/0.17/65 的来源未确认；放弃车辆无 reward transition；无电网奖励项。

## 4 FedSetRL 方法

### 4.1 站点集合 Q 网络

对每站：

$$
u_s=\operatorname{ReLU}(W_{s2}\operatorname{ReLU}(W_{s1}f_s+b_{s1})+b_{s2})\in\mathbb R^{32},
$$

$$
v_i=\operatorname{ReLU}(W_e e_i+b_e)\in\mathbb R^8,
$$

$$
h_{i,s}=\operatorname{ReLU}
\left(\operatorname{LN}(W_f[u_s\Vert v_i]+b_f)\right)\in\mathbb R^{32}.
$$

均值集合上下文：

$$
\bar h_i=\frac1S\sum_{s=1}^S h_{i,s}.
$$

每站 Q 值：

$$
Q_\theta(X_i,s)=W_{q2}\operatorname{ReLU}
(W_{q1}[h_{i,s}\Vert\bar h_i]+b_{q1})+b_{q2}.
$$

- 代码：`agents/network_station_only.py:33-57,71-98,126-167`。
- 维度：station 7→32→32，EV 1→8，fusion 40→32，head 64→32→1。
- 实现程度：完全实现。它是 DeepSets 式均值集合网络，不用道路图卷积或边特征。

### 4.2 Double DQN 框架及当前退化

代码的一般 Double DQN 目标：

$$
a^*=\arg\max_a Q_{\theta}(s',a),
$$

$$
y=r+0.99(1-d)Q_{\bar\theta}(s',a^*),
$$

$$
\mathcal L=\frac1B\sum_i(Q_\theta(s_i,a_i)-y_i)^2.
$$

online 参数每 batch 用 Adam(lr=$3\times10^{-4}$) 更新，target 每 100 个梯度步硬同步；replay 50000，batch 64。

但当前 HindsightTrainer 对每条有效样本均置 $d=1,s'=\varnothing$，故

$$
y=r.
$$

- 代码：`agents/dqn_base.py:35-92,138-212`；`agents/hindsight_dqn_agent.py:18-36`；`trainer/trainer.py:86-93`。
- 实现程度：Double DQN 框架存在；当前有效训练是终端 Q 回归，target/gamma 不影响目标。

### 4.3 FedAvg

令 $n_k$ 为客户端 k 本轮新增的 charge-start terminal transition 数，代码使用

$$
\alpha_k=\frac{\max(1,n_k)}{\sum_j\max(1,n_j)},
$$

$$
\theta^{r+1}=\sum_k\alpha_k\theta_k^{r}.
$$

若 `aggregation_momentum=m<1`：

$$
\theta^{r+1}\leftarrow(1-m)\theta^r+m\sum_k\alpha_k\theta_k^r.
$$

- 代码：`train_federated_hindsight.py:98-158`。
- 聚合：policy 浮点 state；target 由 policy 同步；optimizer/replay 不聚合。
- 实现程度：完全实现。

### 4.4 FedProx

本地目标等价于：

$$
\mathcal L_k^{prox}(\theta)=\mathcal L_k(\theta)+
\frac\mu2\|\theta-\theta^r\|_2^2.
$$

代码不显式把标量加到 loss，而在反向后注入梯度

$$
g\leftarrow g+\mu(\theta-\theta^r),
$$

数学上等价。

- 代码：`train_federated_hindsight.py:80-95,400-405`。
- 参数：CLI 默认 $mu=0.01$；`run_fl_methods.py` 正式对比传 $mu=0.1$。
- 实现程度：对 policy 全参数每 batch 实现；target 不直接受近端项。

### 4.5 个性化部分参数共享

将网络参数分为

$$
\theta_k=(\phi_k,\psi_k),
$$

其中 $\phi$ 对应 `encoder.*`，$\psi$ 对应 `head.*`。每轮只聚合

$$
\phi^{r+1}=\sum_k\alpha_k\phi_k^r,
$$

并在下轮覆盖客户端 encoder；head 保留：

$$
\psi_k^{r+1}\leftarrow\psi_k^r.
$$

- 代码：`train_federated_hindsight.py:60-78,98-158`。
- 参数量：station_only 全部 4817，共享 encoder 2704（56.13%），本地 head 2113（43.87%）；相对 FedAvg 单向通信浮点参数量减少约 43.87%。
- 实现程度：部分参数聚合完全实现；不是严格原始 FedRep，因为没有 head/encoder 分阶段冻结优化。最后一轮聚合后未重新下发，保存的本地最终 encoder 也不是最终聚合 encoder（`:417,437-450`）。

### 4.6 同一调度时隙已提交计数前瞻

同一时隙内车辆按 SOC 从低到高编号为 $i_1,\ldots,i_J$：

$$
z_{s,t}^{(j)}=\sum_{k<j}\mathbf1(a_{i_k,t}=s).
$$

对第 j 辆车，代码用 $z$：

$$
\widetilde q_s^{(j)}=|q_s|+z_{s,t}^{(j)},
$$

$$
\pi_s^{(j)}=\frac{|q_s|+|\mathcal C_s|+widehat A_s+z_{s,t}^{(j)}}{20+M_s},
$$

$$
h_s^{(j)}=\frac{n_s^{heading}+z_{s,t}^{(j)}}
{\max(1,\sum_r n_r^{heading}+\sum_rz_{r,t}^{(j)})}.
$$

$z$ 还作为 `incoming_count` 改变预计等待，进而改变 service time 与 wait ratio。

- 代码：`trainer/trainer.py:114-125`；`env/base_env.py:627-660`。
- 单位：车辆数及归一化比例。
- 实现程度：完全实现。真实 queue/load/price 在决策循环内保持冻结；前序动作只改变合成特征。顺序不随机，存在低 SOC 优先偏置。

### 4.7 整体训练与推理流程

训练第 t 步：

$$
\mathcal U_t=\operatorname{sort}_{SOC\uparrow}
\{i:status_i=IDLE,\ SOC_i<30\},
$$

$$
s_i\leftarrow\operatorname{State}(i,z^{(j)}),\quad
a_i\leftarrow\epsilon\text{-greedy}(Q_\theta(s_i,\cdot)),
$$

$$
\text{pending}[i]\leftarrow(s_i,a_i),\qquad
z_{a_i}\leftarrow z_{a_i}+1.
$$

环境统一执行动作并推进交通、队列、充电、AC 潮流。当 i 开始充电：

$$
r_i\leftarrow r(T_i^{actual},W_i^{actual},F_i^{start}),
$$

$$
\mathcal D_k\leftarrow\mathcal D_k\cup(s_i,a_i,r_i,\varnothing,1).
$$

每环境步从本地 replay 抽 batch 更新；联邦轮末按 4.3–4.5 聚合。推理令 $\epsilon=0$，仍按 SOC 升序和 submitted-count 前瞻逐车 argmax；论文主评估默认不启用 action mask。

- 代码：`trainer/trainer.py:30-136`；`env/base_env.py:849-1093`；`train_federated_hindsight.py:294-450`。
- 实现程度：完全对应当前主路径。

## 不可直接写入当前论文的方法公式

1. Wardrop UE、Beckmann 目标、Frank–Wolfe 更新：当前模块缺失。
2. 标准 $t=t_0[1+\alpha(x/C)^\beta]$：当前实际分母是 $Ct_0$。
3. 每桩绝对空闲时刻 $b_{s,k}$ 的连续事件排队：代码不维护。
4. 连续电化学 CC-CV 电压/电流模型：只有分段功率近似。
5. 无条件 LMP/DLMP：`runopp` 失败会 fallback，且未记录成功率。
6. 训练费用 $\sum_t\lambda_tP_t\Delta t$：该累计量存在，但 reward 不用。
7. 电压/线路/变压器安全约束或电网奖励：当前 hindsight policy/reward 无。
8. 严格 FedRep 两阶段优化：未实现。
9. 有效的多步 Double DQN bootstrap：当前所有训练样本 terminal。
