# FedGRL 交接文档 — 2026-06-19

充电导航选站项目(GNN-based DQN + hindsight reward + 联邦 FedAvg)。本文档记录
本阶段(贡献1/2 验证)的完整状态、产出、待办与复跑方式。前序背景见
`docs/handoff_fedgrl_current.md`,方法细节见 `docs/coordination_ablation.md`。

---

## 0. 一句话现状

- **贡献2(决策层协调机制)**:实验完成、结论定稿,无待办。
- **贡献1(联邦价值)**:同质场景证伪了"联邦赢本地",已重定向为**异构 reward 实验(B 故事)**;
  代码就绪、冒烟通过,**正式训练待运行/运行中**,跑完做评估即可收口。

---

## 1. 关键约定与环境(必读,踩过坑)

- **基础模型/warm-start**:`checkpoints_40ev_32piles_continue/model_final.pth`,所有对比实验
  都从它续训,保证公平。
- **统一规模**:40 EV / 4 站 / 每站 8 桩;评估 10 集 × 144 步,ε=0,seed 42。
- **三客户端**:`old_city:1.3, new_city:1.0, suburb:0.7`(grid_variant:UE 强度);unseen 用 `ieee33:1.1`。
- **奖励**:`compute_hindsight_reward(trip,queue,fee)`(trainer/trainer.py),默认权重 0.4/0.4/0.2。
- **HETERO_REWARD 开关(核心)**:`reward_profiles.py` 定义三城差异化画像
  (old=厌排队 0.15/0.70/0.15,new=厌花费 0.15/0.15/0.70,suburb=厌行程 0.70/0.15/0.15)。
  **仅当环境变量 `HETERO_REWARD=1` 时生效**,否则全是默认均匀权重 → 历史实验零污染。
  训练(HindsightTrainer)和评估(run_mode)都按 `env.grid_variant` 自动查表。
- **可靠性坑(重要)**:
  - 长任务**务必在 VSCode 终端前台跑**;经由本工具的后台进程会被 harness/会话清理在
    ~40min 杀掉(已多次踩坑,非睡眠、非崩溃,日志无报错)。
  - 机器睡眠/休眠已设为永不(`powercfg /change standby-timeout-ac 0` 等);保持别休眠。
  - 查进程用 PowerShell `Get-Process python`,**不要用 git-bash 的 `ps`**(看不到原生进程,会误报已死)。
  - 日志多为 UTF-16,中文可能乱码,数字/英文清楚。

---

## 2. 贡献2:决策层 vs reward 层协调(已完成)

三组实验,均不重训、不改网络,推理时切换取 state 方式。

| 实验 | 脚本 | 数据 | 结论 |
|---|---|---|---|
| Oracle 理想到达等待(时间错峰上界) | `evaluate_oracle_wait.py` | `evaluation/oracle_wait_old_city/` | 上界仅 +1.66%,系统不受排队约束 |
| 竞争强度扫描 | `run_congestion_sweep.py` | `evaluation/congestion_sweep/` | 收益不随拥堵放大、反转负 → **动态占用建模不值得做** |
| A/B/C 决策机制消融 | `evaluate_decision_ablation.py` | `evaluation/decision_ablation_32/`、`_16/` | C(决策层前瞻)一致领先 B≈D 约 +1.7% reward |

- 关键设计:env 内新增 oracle 三方法(`_ev_eta_to_station_hours` / `_oracle_arrival_wait_hours`
  / `get_graph_state_for_ev_oracle`),纯附加,见 `env/base_env.py`。
- A≡B 论证使 B 成为 D 的合理代理 → **决定不重训精确 D**;贡献2 定位"温和但一致的设计优势 + 边界讨论"
  (+1.7% 含 train/test 特征剥夺成分,为上界)。
- 全部方法与数据表已写入 `docs/coordination_ablation.md`。

---

## 3. 贡献1:联邦 vs 集中式 vs 本地

### 3a. 同质场景(已完成,结论:三范式打平)
- 复用框架评估了同质 reward 下的三范式;federated 收敛曲线极平(warm-start 饱和)。
- 跨城矩阵 `evaluation/paradigm/matrix.json`(5 模型 × 4 城):
  - 本地模型**跨城不崩**,联邦**未跑赢本地**(local_new 平均甚至最优)。
  - 根因:三城同为 IEEE33 变体、reward 相同 → **最优策略同质**,联邦无可汇聚增量。
- 同质模型 checkpoint:`checkpoints_local_{old_city,new_city,suburb}/`、`checkpoints_centralized/`、
  `checkpoints_fed_hindsight_40ev/`(federated,含 round10–50 快照)。
- 站点节点迁移测试 `test_station_node_transfer.py`(`evaluation/station_transfer_test.log`):
  换"同样分散"的节点策略照样迁移(+6%);聚集型掉点是**难度变化非策略错配**。
  → 结论:**仅换站点节点不足以制造异质性**。

### 3b. 异构 reward 重定向(进行中,B 故事)
- 决策:与导师确认走 **B 故事**——"联邦=跨城鲁棒单模型,本地专家换城即崩",
  而非"联邦赢本地"(后者需数据稀缺 + from-scratch,见下)。
- 机制:用 `HETERO_REWARD=1` 让三城各自 reward 画像不同 → 最优策略真正分化。
- 代码已就绪、冒烟通过(5 job 全通)。**待运行正式训练 + 评估。**

---

## 4. 立即要做的(接手从这里开始)

### 步骤1:训练(VSCode 前台,~15–17h,可断点续)
```powershell
python run_paradigm_training.py --hetero
```
- 顺序训 5 个:local_old/new/suburb + centralized + federated,全程 HETERO_REWARD=1。
- 输出到 `checkpoints_hetero_*`(不碰同质模型)。中断重跑同命令即自动续。
- 日志:`evaluation/paradigm_train_logs/*_hetero.log`。

### 步骤2:跨城矩阵评估(训完跑,务必带 HETERO_REWARD=1)
```powershell
$env:HETERO_REWARD='1'; python eval_paradigms.py `
  --models local_old=checkpoints_hetero_local_old_city\model_final.pth `
           local_new=checkpoints_hetero_local_new_city\model_final.pth `
           local_sub=checkpoints_hetero_local_suburb\model_final.pth `
           centralized=checkpoints_hetero_centralized\central_final.pth `
           federated=checkpoints_hetero_federated\global_final.pth `
  --cities old_city:1.3,new_city:1.0,suburb:0.7 `
  --episodes 10 --steps-per-episode 144 --seed 42 `
  --out evaluation\paradigm\matrix_hetero.json; Remove-Item Env:\HETERO_REWARD
```

### 步骤3:判读 `matrix_hetero.json`
预期(B 成立):**对角线**本地专家最强;**非对角线**本地换城明显掉点;
**federated/centralized 整行稳**。→ 证明"本地专家不可跨城,联邦一模型走天下"。

⚠️ 若非对角线仍不崩 → 即便 reward 画像也没拉开策略(站点 trade-off 太弱),
需转 **数据稀缺 + from-scratch** 路线(同目标、每客户端少数据、不 warm-start;
预期 Local<Fed≈Centralized,但要调参、成本更高)。

---

## 5. 文件索引

**新增脚本**
- `reward_profiles.py` — 三城 reward 画像 + HETERO_REWARD 开关
- `evaluate_oracle_wait.py` — oracle 时间错峰上界(baseline/oracle)
- `run_congestion_sweep.py` — 拥塞梯度扫描
- `evaluate_decision_ablation.py` — A/B/C 决策机制消融
- `eval_paradigms.py` — 任意模型 × 多城评估(增量落盘),三范式对比用
- `train_centralized_hindsight.py` — 集中式训练入口(共享 agent,无 FedAvg)
- `run_paradigm_training.py` — 训练流水线(`--hetero` 开异构;job 级断点续 + 保活)
- `test_station_node_transfer.py` — 站点节点迁移测试

**改动(纯附加/开关式,默认行为不变)**
- `env/base_env.py` — oracle 三方法
- `trainer/trainer.py` — reward 支持权重 + 按城市自动套画像
- `evaluate_oracle_wait.py` `_build_env` — 支持传 `station_node_ids`

**文档/记忆**
- `docs/coordination_ablation.md` — 贡献2 三实验完整方法与数据
- 记忆 `coordination-ablation-decisions`(项目决策:不重训 D 等)

---

## 6. 待办 / 开放问题
1. 跑完异构训练 + 评估,出跨城矩阵表 + 出版级图(可用 nature-figure 技能)。
2. 据矩阵定稿贡献1 立论(联邦的隐私 + 泛化价值);若打平则转 from-scratch 路线。
3. 补 mean field / 全局上下文等模块的消融(导师周报里提到的下一步)。
4. (可选)若审稿需要,补精确 D 的短轮次重训以坐实贡献2。
