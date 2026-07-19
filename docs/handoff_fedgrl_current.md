# FedGRL Charging Navigation Handoff

## Current Status

The current branch has a working hindsight contextual bandit training flow for
EV charging-station selection, plus a federated training script over
traffic-grid heterogeneous clients.

The main validated model is:

```text
checkpoints_fed_hindsight_40ev/global_final.pth
```

It was trained with:

```powershell
python train_federated_hindsight.py --rounds 50 --local-episodes 2 --steps-per-episode 144 --batch-size 64 --num-evs 40 --num-stations 4 --num-chargers-per-station 8 --client-specs old_city:1.3,new_city:1.0,suburb:0.7 --no-use-action-mask --load-model checkpoints_40ev_32piles_continue\model_final.pth --epsilon 0.3 --save-dir checkpoints_fed_hindsight_40ev
```

## Main Code Changes

### Station-only GNN policy

File: `agents/network_station_only.py`

- Uses station-focused features.
- EV feature remains SOC only.
- Adds station global context by averaging all station embeddings and
  concatenating it with each station embedding before Q-value prediction.
- Current station feature set includes real-time mean field:

```python
STATION_FEATURE_IDXS = [3, 5, 10, 11, 15, 17, 18]
```

### Real-time mean field feature

Files:

- `env/base_env.py`
- `env/real_env.py`

The environment tracks `evs_heading_to`, i.e. how many EVs are currently heading
to each station. This is written to node feature index `18`. Dispatch-time
`pending_counts` is also included so same-tick decisions are coupled.

### Reward settlement

Files:

- `env/charging_station.py`
- `trainer/trainer.py`

Reward is now settled when the EV starts charging, not when charging completes.
The reward is:

```python
reward = -(
    0.4 * (trip / NORM_TRIP)
    + 0.4 * (queue / NORM_WAIT)
    + 0.2 * (fee / NORM_FEE)
)
```

Current constants:

```python
NORM_TRIP = 0.58
NORM_WAIT = 0.17
NORM_FEE = 65.0
```

### Grid variants

Files:

- `env/grid_variants.py`
- `env/power_grid_pp.py`
- `env/real_env.py`

Available grid variants:

```text
ieee33
old_city
new_city
suburb
```

These variants are used to create heterogeneous clients for federated training.
`old_city` has higher load and weaker lines, `new_city` adds PV, and `suburb`
has lower load with PV/storage.

### New scripts

- `train_hindsight.py`: single-environment hindsight training.
- `train_federated_hindsight.py`: FedAvg over multiple hindsight clients.
- `evaluate_hindsight.py`: evaluation with `random`, `shortest`, `greedy`, and
  `model` strategies.

## Important Commands

### Single model evaluation

```powershell
python evaluate_hindsight.py --model checkpoints_fed_hindsight_40ev\global_final.pth --episodes 20 --steps-per-episode 144 --num-evs 40 --num-stations 4 --num-chargers-per-station 8 --grid-variant old_city --ue-scale 1.3 --no-use-action-mask --epsilon 0 --compare-baselines --save-dir evaluation\fed_eval_old_city_v2
```

Change `--grid-variant`, `--ue-scale`, and `--save-dir` for other scenarios.

### Evaluated scenarios

```powershell
python evaluate_hindsight.py --model checkpoints_fed_hindsight_40ev\global_final.pth --episodes 20 --steps-per-episode 144 --num-evs 40 --num-stations 4 --num-chargers-per-station 8 --grid-variant old_city --ue-scale 1.3 --no-use-action-mask --epsilon 0 --compare-baselines --save-dir evaluation\fed_eval_old_city_v2

python evaluate_hindsight.py --model checkpoints_fed_hindsight_40ev\global_final.pth --episodes 20 --steps-per-episode 144 --num-evs 40 --num-stations 4 --num-chargers-per-station 8 --grid-variant new_city --ue-scale 1.0 --no-use-action-mask --epsilon 0 --compare-baselines --save-dir evaluation\fed_eval_new_city_v2

python evaluate_hindsight.py --model checkpoints_fed_hindsight_40ev\global_final.pth --episodes 20 --steps-per-episode 144 --num-evs 40 --num-stations 4 --num-chargers-per-station 8 --grid-variant suburb --ue-scale 0.7 --no-use-action-mask --epsilon 0 --compare-baselines --save-dir evaluation\fed_eval_suburb_v2

python evaluate_hindsight.py --model checkpoints_fed_hindsight_40ev\global_final.pth --episodes 20 --steps-per-episode 144 --num-evs 40 --num-stations 4 --num-chargers-per-station 8 --grid-variant ieee33 --ue-scale 1.1 --no-use-action-mask --epsilon 0 --compare-baselines --save-dir evaluation\fed_eval_unseen_ieee33_v2
```

## Evaluation Summary

Full result table:

```text
docs/federated_hindsight_eval_summary.md
```

Short summary:

| Scenario | Best Strategy | Model Reward | Greedy Reward | Shortest Issue |
|---|---|---:|---:|---|
| old_city, UE 1.3 | model | -0.6941 | -0.7072 | high queue, 33 abandoned |
| new_city, UE 1.0 | model | -0.5825 | -0.6073 | high queue, 33 abandoned |
| suburb, UE 0.7 | model | -0.5812 | -0.5991 | high queue, 33 abandoned |
| unseen_ieee33, UE 1.1 | model | -0.6151 | -0.6498 | high queue, 33 abandoned |

Main conclusion:

```text
The FedGRL model consistently beats random, shortest-path, and cost-greedy
baselines across the three training clients and the unseen IEEE33 scenario.
Shortest-path minimizes trip time but causes severe queueing. The model learns a
better tradeoff: shorter trip than greedy, slightly higher but acceptable queue,
and better total reward.
```

## Notes on Decision Semantics

Within one simulation tick, all pending EVs are collected first. They are then
assigned sequentially, but `pending_counts` and real-time mean field are updated
inside the tick. Therefore the current setup is:

```text
sequentialized batch decision with intra-step coupling
```

It is not a fully simultaneous game, but it is also not independent static
single-EV decision making.

## Known Caveats

- `main.py` and the old `training.*` entrypoints are legacy and should not be
  used for the current hindsight experiments.
- `agents/FederatedDQN.py` is also legacy for the current workflow. Use
  `train_federated_hindsight.py` instead.
- `__pycache__` files may show as modified because Python writes bytecode. Use
  `$env:PYTHONDONTWRITEBYTECODE='1'` when running tests if needed.
- The current validated scale is 40 EVs, 4 stations, and 32 chargers. 80 EVs
  with immediate low-SOC respawn is often overloaded and harder to train.

## Recommended Next Steps

1. Add ablations:
   - without mean field feature
   - without global station context
   - without grid heterogeneity
   - local-only vs centralized vs federated
2. Convert the evaluation table into a paper-ready table.
3. Run repeated seeds for the four main evaluation scenarios.
4. Consider a more realistic respawn model for final evaluation, where EVs do
   not immediately respawn with low SOC after full charge.
