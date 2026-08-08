# Held-Out Evaluation: BipedalWalker (CREATE)

- env: BipedalWalker-v3 (original reward)
- episodes: 100
- seeds: 50000..50099
- threshold: 300.0

| seed | dev_score | held_out_mean | held_out_std | min | max | len | term | solved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_0 | 320.02 | 317.20 | 27.68 | 43.2 | 321.3 | 1163.5 | 100/100 | yes |
| seed_1 | 313.73 | 313.89 | 0.57 | 311.8 | 315.3 | 1083.6 | 100/100 | yes |
| seed_2 | 307.92 | 307.59 | 0.81 | 305.0 | 309.4 | 1046.1 | 100/100 | yes |
| seed_3 | 311.12 | 310.99 | 0.92 | 308.5 | 313.9 | 1051.2 | 100/100 | yes |
| seed_4 | 304.92 | 304.45 | 0.76 | 303.0 | 306.5 | 1131.1 | 100/100 | yes |

## Summary

- solved: 5/5
- held_out_mean: 310.82
- held_out_std: 5.03
