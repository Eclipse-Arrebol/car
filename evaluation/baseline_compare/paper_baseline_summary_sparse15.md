# Paper Baseline Summary

Average reward (higher is better).

| Method | 40 EV | 60 EV | 80 EV |
|---|---:|---:|---:|
| Best single local | -0.6305 | -0.9839 | -1.9708 |
| Centralized | -0.6336 | -1.0433 | -1.9579 |
| FedAvg | -0.7051 | -1.2483 | -2.2288 |
| FedProx | -0.6700 | -1.2083 | -2.2262 |
| FedRep diagonal | -0.6328 | -1.0156 | -1.9511 |
| Greedy generalized cost | -0.6826 | -1.2731 | -2.0955 |
| Local diagonal | -0.6453 | -0.9630 | -1.9583 |
| Random | -0.7891 | -1.3151 | -2.1425 |
| Shortest trip | -1.4867 | -2.3660 | -2.9777 |

Average queue time in minutes (lower is better).

| Method | 40 EV | 60 EV | 80 EV |
|---|---:|---:|---:|
| Best single local | 2.84 | 9.11 | 40.66 |
| Centralized | 6.02 | 19.91 | 47.68 |
| FedAvg | 5.95 | 22.55 | 51.91 |
| FedProx | 5.75 | 21.59 | 51.74 |
| FedRep diagonal | 6.36 | 18.94 | 47.31 |
| Greedy generalized cost | 3.65 | 20.00 | 44.45 |
| Local diagonal | 6.88 | 18.23 | 47.62 |
| Random | 2.04 | 17.93 | 43.02 |
| Shortest trip | 31.26 | 58.16 | 76.84 |

Sources are listed in the CSV file.
