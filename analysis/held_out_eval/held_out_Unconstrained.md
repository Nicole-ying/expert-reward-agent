# Held-Out Evaluation: Unconstrained

- episodes: 100
- seeds: 50000..50099
- env: LunarLander-v3 (original reward)
- threshold: 200.0

| seed | dev_score | held_out_mean | held_out_std | min | max | len | term | solved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_0 | 169.90 | 167.89 | 101.59 | -174.6 | 279.9 | 424.3 | 91/100 | no |
| seed_1 | 130.64 | 84.22 | 106.49 | -160.5 | 210.2 | 712.2 | 83/100 | no |
| seed_2 | 71.06 | 82.08 | 50.08 | -11.3 | 194.9 | 955.0 | 27/100 | no |
| seed_3 | 59.18 | 64.44 | 111.46 | -88.0 | 258.9 | 744.1 | 53/100 | no |
| seed_4 | 140.27 | 115.72 | 131.24 | -180.7 | 275.8 | 437.1 | 97/100 | no |

## Summary

- solved: 0/5
- held_out_mean: 102.87
- held_out_std: 40.77
