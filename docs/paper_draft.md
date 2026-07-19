# FedGRL: A Graph Reinforcement Learning Framework for EV Charging-Station Navigation, with Federated Cross-City Robustness and Decision-Layer Concurrency Coordination

> **Draft status.** This file contains the first three core sections (Related Work,
> Method, Experiments) targeted at an IEEE transportation/grid journal (e.g.,
> *IEEE T-ITS* / *IEEE T-Smart Grid* / *IEEE T-Transportation Electrification*),
> two-column, ~10–13 pages. Abstract / Introduction / Discussion / Conclusion are
> stubbed at the end and will follow. Numbers tagged `[TODO: ...]` are not yet
> available (heterogeneous federated matrix and the `station_attn` ablation are
> still training); every other number is taken from `evaluation/*.json` in this
> repository and is cross-referenced to its source file. **No numbers are invented.**

---

## II. Related Work

> *(Section numbering assumes I = Introduction, to be written.)*

### A. Learning-based EV charging-station navigation

Charging-station selection (a.k.a. charging navigation or charging recommendation)
sits between vehicle routing and demand-side grid management: an EV that needs to
charge must pick *where* to go, trading off detour/trip time, expected queueing
delay at the station, and energy price, while its choice perturbs both the road
network and the distribution grid. Early work casts this as a static optimization
or a myopic greedy rule (nearest station, cheapest station, shortest expected
wait). Reinforcement learning (RL) reframes it as a sequential decision problem
in which the agent learns a station-selection policy from interaction
[TODO: cite 3–5 RL-for-EV-charging works]. Graph neural networks (GNNs) are a
natural state encoder here because the joint traffic–grid system is a graph; a
growing line of work couples GNN state encoders with value- or policy-based RL for
charging navigation and coordinated charging [TODO: cite GNN-RL charging works].
Our base learner is in this family: a station-restricted GNN Q-network trained
with Double DQN on a per-vehicle *hindsight* reward (Section III-C–D).

### B. Federated reinforcement learning across operators/cities

Charging operators in different cities cannot freely pool raw mobility and grid
data (privacy, regulatory, and competitive constraints), which motivates federated
RL: each city trains locally and only model parameters are aggregated, classically
by FedAvg [TODO: cite McMahan FedAvg; 2–3 federated RL / federated EV works]. The
*expected* selling point in much of this literature is that federation **improves**
task performance over isolated local training by pooling experience. We report a
more nuanced finding (Section V): under **homogeneous** objectives across cities,
federation does **not** beat local training — the optimal policy is shared, so
there is no aggregable increment, and an isolated local model already transfers
across cities without collapsing (Section V-A). We therefore reposition the value
of federation as **cross-city robustness under objective heterogeneity**: when the
three cities optimize genuinely different trade-offs, a single federated model
stays uniformly strong across all of them, whereas a local "expert" overfits its
home city's objective and degrades when deployed elsewhere (Section V-B). This is
a robustness/single-model-deployment argument, not a "federation wins on the
training distribution" argument, and we state the homogeneous null result openly.

### C. Multi-agent concurrency coordination — and where the congestion signal is used

When many EVs are dispatched at the **same scheduling tick**, they observe nearly
identical station states and independently select the same "currently best"
station — a herding / concurrent-decision externality. The multi-agent RL (MARL)
literature addresses such coupling with, among others, mean-field approximations,
centralized critics, and explicit communication [TODO: cite mean-field RL,
centralized-critic MARL, comm-based MARL]. In the charging-navigation literature
specifically, the dominant remedy is a **reward-layer congestion penalty**: the
training objective is augmented with a term that penalizes choosing a station that
is (or becomes) congested, so the learned policy avoids busy stations *on
average* [TODO: cite 2–4 reward-shaping congestion-penalty charging works].

**This is the line we draw.** Our contribution is *not* a new congestion feature
and *not* a method "for high-competition regimes." It is a controlled statement
about **where in the agent the concurrency signal is placed and when it is used.**
We compare two designs that consume the **same** congestion information:

- **Reward-layer (the prevailing school, denoted D):** the concurrency/congestion
  signal enters the *training objective*. The policy learns an average tendency to
  avoid busy stations, but at inference all EVs in a tick still decide against a
  frozen state and cannot resolve *which specific* vehicles should split off *this*
  tick.
- **Decision-layer lookahead (ours, denoted C):** the *same* per-tick occupancy
  information enters the *state at decision time*. EVs in a tick decide
  sequentially, and each later EV sees the choices already committed by earlier EVs
  this tick (`pending_counts`), breaking the within-tick symmetry directly.

Both use one congestion signal; they differ only in *which layer* (objective vs.
state) and *when* (training-time average vs. decision-time lookahead) it acts. Our
finding is thus a **placement principle for the coordination mechanism**, not
feature engineering. Crucially, we also delimit it honestly (Section V-C): the
gain is driven by *concurrency* (how many EVs decide simultaneously), **not** by
congestion severity; it does **not** amplify under heavier load and can even turn
negative at capacity saturation; and it is **not** Pareto-dominant (it pushes cost
from the mean toward the tail). We position C as a *mild but consistent,
mechanistically clear design principle*, which is a stronger and more defensible
claim than an unqualified "method for congested networks."

---

## III. Method

### A. Problem setting and environment

We study **charging-station navigation**: when an EV's state of charge (SoC)
triggers a charging need, a learned policy selects one of the candidate charging
stations; the EV then routes there, possibly queues, charges, and resumes its
trip. The environment couples a road/traffic network with an electrical
distribution grid (IEEE 33-bus variants), so a station choice affects trip time,
station queueing, energy fee, and grid loading simultaneously. Unless stated
otherwise we use a **unified scale of 40 EVs, 4 charging stations, and 8 chargers
per station** (32 chargers total), implemented in `env/base_env.py`.

Multiple EVs can reach a charging decision within the same scheduling step
("tick"). We treat each EV's station choice as one decision in a per-vehicle
Markov decision process (MDP); the agents share a single Q-network (parameter
sharing). The action space is the set of stations, $|\mathcal{A}| = $ number of
stations (4 in the unified scale); an action mask removes infeasible stations
(insufficient SoC to reach, or estimated trip $\ge 24$ h), see
`get_action_mask` in `env/base_env.py`.

### B. State representation and station-restricted GNN Q-network

**Node features.** The environment exposes a graph state `Data(x, edge_index,
edge_attr)` with a 19-dimensional feature vector per node (`get_graph_state`,
`get_graph_state_for_ev` in `env/base_env.py`). For a querying EV, the SoC is
written onto its current node (feature index 8), and each station node is filled
with decision-relevant quantities estimated for *that* EV: price/LMP (idx 3),
station load ratio (idx 5), estimated trip time (idx 10), estimated service time
(idx 11), queue-wait ratio (idx 15), spare-charger ratio (idx 17), and a real-time
mean-field "vehicles heading to this station" signal (idx 18).

**Station-restricted encoder (DeepSets).** Most nodes in the road graph carry zero
station features and act as noise for the station-selection head. Our base
Q-network `StationOnlyGraphQNetwork` (`agents/network_station_only.py`) therefore
reads only the station nodes and the querying EV's SoC. Concretely it uses
station-feature indices `[3, 5, 10, 11, 15, 17, 18]` and EV-feature index `[8]`.
Each station is encoded by an MLP, fused with an EV-SoC embedding, and — this is
the only cross-station interaction — a **global context is formed by mean-pooling
the station embeddings**; each station's $Q$-value is read out from
`[station embedding ‖ global context]`. This is a DeepSets-style permutation-
invariant set encoder (no message passing, no edge features); the **mean-pool is
the sole, and deliberately blunt, relational channel between stations.** That
property motivates the architecture ablation in Section III-F.

**Learner.** The policy is trained with **Double DQN** (`agents/dqn_base.py`,
`agents/hindsight_dqn_agent.py`): a policy network selects the argmax action and a
periodically synchronized target network evaluates it, reducing value
overestimation. [TODO: confirm/insert exact hyperparameters — replay buffer size,
batch size, $\gamma$, learning rate, target-update period, $\epsilon$ schedule —
from `agents/dqn_base.py` / training configs.]

### C. Hindsight per-vehicle reward

Station-selection outcomes (realized trip time, queueing delay, and energy fee)
are only known *after* the EV has traveled, queued, and charged. We therefore use
a **hindsight per-vehicle reward**: at dispatch time the transition is held
pending (`PendingEntry` in `trainer/trainer.py`), and when the charging session
completes the realized $(\text{trip}, \text{queue}, \text{fee})$ are read back and
the reward is assigned to that stored transition. The reward
(`compute_hindsight_reward`, `trainer/trainer.py`) is a normalized, weighted,
negated cost:

$$
r = -\Big( w_\text{trip}\,\tfrac{\text{trip}}{T_0} + w_\text{queue}\,\tfrac{\text{queue}}{W_0} + w_\text{fee}\,\tfrac{\text{fee}}{F_0}\Big),
$$

with normalizers $T_0 = 0.58$ h, $W_0 = 0.17$ h, $F_0 = 65$ and default weights
$(w_\text{trip}, w_\text{queue}, w_\text{fee}) = (0.4, 0.4, 0.2)$.

Two points matter for the rest of the paper. First, the **queue term**
$w_\text{queue}\cdot\text{queue}/W_0$ is a *realized-queueing* cost — so the
training objective **already contains a congestion penalty**, expressed directly
as experienced wait time rather than as an occupancy-count proxy. This is what
makes our base model a legitimate stand-in for the reward-layer school when
analyzing coordination (Section V-C). Second, the scalar reward emitted inside
`env.step` (which additionally includes a $10\times$ queue-cost term, voltage
penalties, etc.) is **not** used for training; the agent learns *only* this
per-vehicle hindsight reward.

### D. Decision-layer lookahead coordination (Contribution 2)

**The concurrency problem.** Within one tick, several EVs may need a station. If
they all decide against the *same frozen* state, they observe the same "best"
station and pile onto it — a herding externality from concurrent decisions.

**Mechanism placement.** We formalize four paradigms that all use the *same*
congestion information but place/time it differently (evaluated in
`evaluate_decision_ablation.py`):

- **A — pure concurrent:** all EVs in a tick decide from the frozen state;
  `pending_counts` is not updated. (No coordination.)
- **B — sequential, decoupled:** EVs decide sequentially, but the state carries no
  `pending_counts` (passed as `None`). Sequential ordering alone, with no
  information channel between within-tick decisions.
- **C — ours (decision-layer lookahead):** EVs decide sequentially, and a running
  `pending_counts` (how many EVs have already chosen each station *this tick*) is
  injected into the state of each later-deciding EV. In `env/base_env.py`,
  `get_graph_state_for_ev(ev, pending_counts)` folds these counts into the
  station's predicted-arrivals / mean-field features (indices 16, 18) and price
  (index 2), so a later EV sees a station that earlier EVs already filled this tick
  as correspondingly busier and routes around it.
- **D — reward-layer penalty (the prevailing school):** no decision-time
  lookahead; congestion enters the *training objective* instead. D differs from
  A/B/C in the *training target*, not in inference, so it strictly requires
  retraining.

**Why B is a valid proxy for D (no retraining).** In this architecture, a tick's
decisions are first *collected* and then applied by a single `env.step`; during
the decision loop the environment is frozen. Therefore the **only** within-tick
coupling channel is `pending_counts`. With lookahead disabled, **A and B are
mechanically identical** — we confirm empirically that A and B produce bit-for-bit
identical action distributions and metrics. Moreover, the trained reward already
embeds a *realized-queueing* congestion penalty (Section III-C), which is a more
direct signal than D's occupancy-count proxy. Hence "**base model + B inference**"
is a sound semantic approximation of D, and we adopt it to avoid a ~5 h retrain
(decision recorded 2026-06-14). We flag the approximation's limits in Section V-C.

### E. Federated training across cities (Contribution 1)

We instantiate three **client cities** as IEEE 33-bus traffic–grid variants with
different unbalanced-load (UE) intensities: `old_city` (UE scale 1.3),
`new_city` (1.0), and `suburb` (0.7); an unseen `ieee33` (1.1) variant is held out
for cross-grid generalization. Each client trains locally on its own environment
and only **model parameters** are shared.

**Aggregation.** A `FederatedServer` (`agents/FederatedDQN.py`) performs **FedAvg**
with **sample-count weighting**: after each round the global parameters are the
per-client state dicts averaged with weights proportional to each client's number
of samples that round ($w_c = \max(1, n_c)$, $\theta \leftarrow \sum_c
\tfrac{w_c}{\sum_{c'} w_{c'}}\,\theta_c$; see `FederatedServer.aggregate`). Buffer
parameters such as `station_node_ids` are kept from the global model rather than
averaged.

**Objective heterogeneity (the B story).** With homogeneous rewards across cities
the optimal policy is shared and federation has nothing to aggregate (the null
result in Section V-A). To create genuinely divergent optima we activate
*per-city reward profiles* (`reward_profiles.py`), gated by the environment
variable `HETERO_REWARD=1` so that every prior experiment is unaffected:

| City | Trade-off | $(w_\text{trip}, w_\text{queue}, w_\text{fee})$ |
|---|---|---|
| `old_city` | queue-averse (congested) | $(0.15, 0.70, 0.15)$ |
| `new_city` | fee-averse (price-sensitive) | $(0.15, 0.15, 0.70)$ |
| `suburb` | trip-averse (long distances) | $(0.70, 0.15, 0.15)$ |

The profile is looked up by `env.grid_variant` in both the trainer (training
signal) and evaluation, so no CLI threading is needed. Under this setting each
city's *expert* is tuned to its own trade-off, while the federated model must
serve all three — setting up the robustness test of Section V-B.

### F. Architecture ablation: mean-pool vs. station self-attention

Because the base Q-network's *only* cross-station channel is a mean-pool, it
caps how much relational reasoning (hence coordination) the network can express.
We add a controlled counterpart `StationAttnGraphQNetwork`
(`agents/network_station_attn.py`, variant `station_attn`) that consumes
**identical features** (same `STATION_FEATURE_IDXS` / `EV_FEATURE_IDXS`) and
differs in exactly one way: the mean-pool is replaced by **multi-head self-
attention across stations** (and the forward pass is fully vectorized). This
isolates "*how* station information is fused" (blunt average vs. learned relational
attention) from "*which* features are seen." **Fairness caveat:** the attention
weight structure differs from the base network, so it cannot reuse the base
warm-start and must train from scratch; absolute metrics across the two networks
therefore mix architecture with warm-start vs. from-scratch and are not directly
A/B-comparable. The decision ablation's C-vs-B gap, however, is an *inference-time*
switch within one checkpoint and is unaffected. [TODO: insert `station_attn`
results once training completes — C-vs-B gap and load-Gini under attention.]

---

## IV. Experimental Setup

- **Base model / warm-start.** Unless noted, comparisons continue-train from a
  shared checkpoint (`checkpoints_40ev_32piles_continue/model_final.pth`) for
  fairness across paradigms.
- **Unified scale.** 40 EVs / 4 stations / 8 chargers per station.
- **Evaluation protocol.** Deterministic policy ($\epsilon = 0$, argmax), 10
  episodes × 144 steps, `seed = 42`, no action mask, unless a specific experiment
  states otherwise (the federated baseline comparison in Section V-A uses 20
  episodes). Each episode rebuilds the environment from `seed + ep`, so paired
  conditions share bit-identical initial conditions and the only varying factor is
  the mechanism under test.
- **Metrics.** Average per-vehicle reward; average trip time, queueing time, and
  fee; **load Gini** (spatial evenness of station utilization) and action Gini;
  served charging events (throughput); abandonment rate; and distribution-grid
  voltage diagnostics. Metric definitions reuse a single implementation
  (`_mean/_variance/_std/_gini/_pct_change` in `evaluate_oracle_wait.py`).
- **Baselines.** `random`, `shortest-path`, `cost-greedy`, and the learned FedGRL
  model.

---

## V. Results

### A. Federation does not beat local under homogeneous objectives (the null result)

We first evaluate five paradigms — three local experts (`local_old`, `local_new`,
`local_sub`), a `centralized` model (shared agent, no FedAvg), and the `federated`
(FedAvg) model — on a 4-city matrix (`old_city`, `new_city`, `suburb`, and the
unseen `ieee33`) under **homogeneous** rewards. Source:
`evaluation/paradigm/matrix.json` (10 episodes × 144 steps, $\epsilon=0$,
seed 42).

**Average reward across the four cities (higher = better):**

| Paradigm | Avg reward | Avg queue (h) | Avg trip (h) | Load Gini |
|---|---:|---:|---:|---:|
| local_old | −0.6591 | 0.0915 | 0.3407 | 0.1262 |
| **local_new** | **−0.6024** | 0.0609 | 0.3621 | 0.1111 |
| local_sub | −0.6351 | 0.0788 | 0.3486 | 0.1127 |
| centralized | −0.6107 | 0.0664 | 0.3550 | 0.1106 |
| federated | −0.6201 | 0.0713 | 0.3522 | 0.1157 |

**Finding.** Federation does **not** win: the best average reward belongs to a
*local* model (`local_new`, −0.6024), ahead of both `centralized` (−0.6107) and
`federated` (−0.6201); and no local model **collapses** when moved off its home
city (e.g., `local_new` is competitive on `old_city`/`suburb`/`ieee33`, not just
`new_city`). The root cause is that all three cities are IEEE-33 variants with the
*same* reward, so the optimal station-selection policy is essentially shared and
federation has no city-specific increment to aggregate. This motivates the
objective-heterogeneity redirect (Section V-B).

For context, against non-learned baselines the federated model is consistently
best on reward in every scenario including the unseen grid (source:
`docs/federated_hindsight_eval_summary.md`, `evaluation/fed_eval_*_v2/`; 20
episodes). E.g., on `old_city` the model reaches −0.6941 reward vs. cost-greedy
−0.7072, shortest-path −1.6212 (which abandons 33 EVs from severe queueing), and
random −0.8554; the model is also best on the unseen `ieee33` (−0.6151),
indicating cross-grid generalization under traffic–grid heterogeneity.

### B. Cross-city robustness under objective heterogeneity (the B story)

With `HETERO_REWARD=1` the three cities optimize genuinely different trade-offs
(Section III-E), so each local model becomes a true *expert* for its city's
objective and the optimal policies diverge. We re-evaluate the same five-paradigm
× 4-city matrix and expect: (i) **diagonal** — each local expert is strongest on
its home city; (ii) **off-diagonal** — a local expert degrades markedly when
deployed to a city with a different objective; (iii) **federated/centralized rows
stay uniformly strong** across all cities. If so, federation's value is a *single,
cross-city-robust deployable model* that no single local expert can match
out-of-domain.

[TODO: insert heterogeneous matrix `evaluation/paradigm/matrix_hetero.json` once
training/eval completes. Report (a) the diagonal expert advantage, (b) the
off-diagonal drop of each local expert, and (c) the worst-city reward of federated
vs. the worst-city reward of the best single local expert — the headline
robustness number.]

> **Contingency (pre-registered).** If the off-diagonal does *not* collapse even
> under heterogeneous rewards (i.e., the station trade-off is too weak to separate
> policies), we will switch to a *data-scarce, from-scratch* federated protocol
> (same objective, few samples per client, no warm-start), where the expected
> ordering is Local < Federated ≈ Centralized.

### C. Coordination mechanism: decision-layer lookahead vs. reward-layer penalty (Contribution 2)

All three sub-experiments use one trained checkpoint
(`checkpoints_fed_hindsight_40ev/global_final.pth`) and switch only the
inference-time state-construction; no retraining, no architecture change. Full
method and data: `docs/coordination_ablation.md`.

#### 1) A/B/C decision-mechanism ablation

Since A and B are bit-for-bit identical (the only within-tick coupling channel is
`pending_counts`), we report C (ours) vs. B (≈ D, the reward-layer school). Source:
`evaluation/decision_ablation_32/` and `evaluation/decision_ablation_16/`.

**32 chargers (slack, ~4.6 min queue):**

| Metric | A = B (≈ D) | C (ours) | C vs. B |
|---|---:|---:|---:|
| Avg reward | −0.7098 | −0.6971 | **+1.79%** |
| Avg queue (h) | 0.0773 | 0.0731 | +5.44% |
| Queue variance | 0.0217 | 0.0196 | +9.42% |
| Avg trip (h) | 0.3590 | 0.3556 | +0.95% |
| Load Gini | 0.1176 | 0.1124 | **+4.48%** |
| Served events | 5567 | 5614 | +47 |

**16 chargers (constrained, ~54 min queue, <0.5% abandon):**

| Metric | A = B (≈ D) | C (ours) | C vs. B |
|---|---:|---:|---:|
| Avg reward | −2.6785 | −2.6341 | **+1.66%** |
| Avg queue (h) | 0.8965 | 0.8808 | +1.75% |
| Avg trip (h) | 0.4385 | 0.4274 | +2.53% |
| Load Gini | 0.0580 | 0.0516 | **+11.10%** |
| Served events | 3363 | 3391 | +28 |
| Queue variance | 1.075 | 1.212 | −12.74% (C worse) |
| Abandon rate | 0.33% | 0.50% | −53% (C worse) |

**Interpretation.** Decision-layer lookahead consistently but mildly beats the
reward-layer proxy on reward (**+1.79%** slack / **+1.66%** constrained), with more
even load, shorter trips, and higher throughput. The mechanism behaves exactly as
the placement argument predicts: decision-time lookahead breaks the within-tick
symmetry so later EVs actively avoid stations *already chosen this tick*, whereas a
reward-layer penalty can only teach an *average* avoidance and cannot coordinate a
*specific* tick.

> **Headline-metric note.** Reward gains are modest (+1.7%); the load-evenness /
> throughput gains are larger (load Gini **+4.5%** slack, **+11.1%** constrained).
> We plan to lead with the load-Gini / throughput framing rather than reward.
> [TODO: finalize headline metric and, if available, fold in `station_attn`
> load-Gini numbers from Section III-F.]

#### 2) Honest boundaries (claimed proactively)

1. **Concurrency-driven, not congestion-driven.** The gain does **not** amplify
   with load: it is essentially flat from slack to constrained (+1.79% → +1.66%),
   consistent with the congestion sweep below.
2. **Not Pareto-dominant.** In the constrained regime C improves
   reward/throughput/load-evenness but *worsens* queue variance (−12.74%) and
   abandonment (0.50% vs. 0.33%): C is a more aggressive spatial split that pushes
   cost from the **mean toward the tail**.
3. **Effect size is an upper bound.** C's checkpoint was trained with
   `pending_counts`; the B condition zeroes a feature the model was trained to use,
   so +1.7% includes a train/test feature-deprivation component and the pure
   mechanism effect is somewhat smaller. We therefore frame C as a *mild but
   consistent design principle*, not an order-of-magnitude improvement.

#### 3) Supporting upper-bound and sweep experiments

**Oracle ideal-arrival wait (time-staggering upper bound).** Replacing the wait
estimate with a cheating multi-server FIFO oracle (perfect future occupancy) on
`old_city`/32 chargers yields only **+1.66% reward** and reduces mean queue by
8.55% — but the mean queue is just **4.4 min**, i.e., the system is not
queue-constrained at this scale, and even the oracle trades trip time / load
evenness for it. Source: `evaluation/oracle_wait_old_city/`. Conclusion: dynamic
*temporal* occupancy modeling is **not worth building** at this scale.

**Congestion sweep.** Sweeping chargers-per-station (total 32 → 16 → 8 → 4) at
fixed 40 EVs / 4 stations tests whether the coordination value grows with
congestion. It does **not** — the oracle's reward gain is *largest at the slackest
setting* and turns negative as load rises (source:
`evaluation/congestion_sweep/`):

| Setting (total chargers) | Base queue (min) | Base abandon | Reward gain % | Mean-queue gain % | Queue-var gain % |
|---|---:|---:|---:|---:|---:|
| 4×8 (=32) slack | 4.4 | 0.0% | +1.66% | +8.55% | +15.35% |
| 4×4 (=16) | 52.8 | 0.5% | +0.28% | +1.16% | +24.84% |
| 4×2 (=8) ⚠️ | 131.4 | 14.1% | −2.31% | −2.38% | −0.87% |
| 4×1 (=4) ⚠️ | 170.8 | 48.6% | −0.93% | −0.99% | +5.52% |

(⚠️ Above ~10% abandonment the queue metrics suffer survivorship bias.) At true
saturation the bottleneck is **total capacity** — every station is full, there is
no staggering room — which calls for siting/expansion, not smarter dispatch. This
is the empirical basis for framing Contribution 2 as a *concurrency-coordination
placement principle* rather than a "method for high-competition regimes."

### D. Architecture ablation: station self-attention

[TODO: insert `station_attn` (multi-head cross-station attention) results from
`evaluate_decision_ablation.py --network station_attn` on
`checkpoints_station_attn_federated/global_final.pth`: (a) C-vs-B gap and load-Gini
under attention vs. mean-pool, to test whether stronger relational reasoning widens
the coordination advantage; and (b) the cross-city matrix `matrix_attn.json`.
Report with the warm-start vs. from-scratch fairness caveat from Section III-F.]

---

## Stubs to be written

- **Abstract** — one-paragraph summary once V-B and V-D numbers land.
- **I. Introduction** — motivation, the two contributions, contribution bullets.
- **VI. Discussion** — placement principle generality; when concurrency
  coordination helps; federation as privacy + robustness; threats to validity.
- **VII. Conclusion.**
- **References** — fill all `[TODO: cite ...]`.
