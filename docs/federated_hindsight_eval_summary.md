# Federated Hindsight Evaluation Summary

## Setup

- Model: `checkpoints_fed_hindsight_40ev/global_final.pth`
- Training clients: `old_city:1.3`, `new_city:1.0`, `suburb:0.7`
- Scale: 40 EVs, 4 stations, 8 chargers per station
- Evaluation: 20 episodes, 144 steps per episode, no action mask, epsilon = 0
- Baselines: random, shortest-path, cost-greedy, FedGRL model

## Results

### old_city, UE scale 1.3

| Strategy | Events | Abandoned | Avg Trip (h) | Avg Queue (h) | Avg Fee | Avg Reward |
|---|---:|---:|---:|---:|---:|---:|
| random | 9153 | 0 | 0.6874 | 0.0401 | 93.2813 | -0.8554 |
| shortest | 9167 | 33 | 0.2765 | 0.4941 | 87.0818 | -1.6212 |
| greedy | 10856 | 0 | 0.4163 | 0.0590 | 91.4435 | -0.7072 |
| model | 11309 | 0 | 0.3540 | 0.0723 | 90.9955 | -0.6941 |

### new_city, UE scale 1.0

| Strategy | Events | Abandoned | Avg Trip (h) | Avg Queue (h) | Avg Fee | Avg Reward |
|---|---:|---:|---:|---:|---:|---:|
| random | 9153 | 0 | 0.6874 | 0.0401 | 59.7387 | -0.7522 |
| shortest | 9167 | 33 | 0.2765 | 0.4941 | 55.7369 | -1.5248 |
| greedy | 10829 | 0 | 0.4235 | 0.0575 | 58.4940 | -0.6073 |
| model | 11351 | 0 | 0.3500 | 0.0694 | 57.8319 | -0.5825 |

### suburb, UE scale 0.7

| Strategy | Events | Abandoned | Avg Trip (h) | Avg Queue (h) | Avg Fee | Avg Reward |
|---|---:|---:|---:|---:|---:|---:|
| random | 9153 | 0 | 0.6874 | 0.0401 | 54.9146 | -0.7373 |
| shortest | 9167 | 33 | 0.2765 | 0.4941 | 51.4672 | -1.5117 |
| greedy | 10811 | 0 | 0.4235 | 0.0600 | 53.8902 | -0.5991 |
| model | 11328 | 0 | 0.3491 | 0.0749 | 53.3418 | -0.5812 |

### unseen_ieee33, UE scale 1.1

| Strategy | Events | Abandoned | Avg Trip (h) | Avg Queue (h) | Avg Fee | Avg Reward |
|---|---:|---:|---:|---:|---:|---:|
| random | 9153 | 0 | 0.6874 | 0.0401 | 71.9090 | -0.7896 |
| shortest | 9167 | 33 | 0.2765 | 0.4941 | 67.2930 | -1.5603 |
| greedy | 10832 | 0 | 0.4179 | 0.0613 | 70.6542 | -0.6498 |
| model | 11380 | 0 | 0.3499 | 0.0672 | 70.0978 | -0.6151 |

## Main Findings

1. The FedGRL model achieves the best average reward in all evaluated scenarios.
2. The shortest-path baseline has the lowest trip time but causes severe queueing and 33 abandoned EVs in every scenario.
3. Compared with cost-greedy, the FedGRL model consistently reduces trip time by about 0.06-0.07 h while keeping queue time within a small increase.
4. The model also performs best on the unseen `ieee33` scenario, suggesting cross-grid generalization under traffic-grid heterogeneity.

## Result Files

- `evaluation/fed_eval_old_city_v2/hindsight_evaluation.json`
- `evaluation/fed_eval_new_city_v2/hindsight_evaluation.json`
- `evaluation/fed_eval_suburb_v2/hindsight_evaluation.json`
- `evaluation/fed_eval_unseen_ieee33_v2/hindsight_evaluation.json`
