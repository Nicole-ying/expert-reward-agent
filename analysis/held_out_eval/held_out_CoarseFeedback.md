# Held-Out Evaluation: CoarseFeedback

- episodes: 100
- seeds: 50000..50099
- env: LunarLander-v3 (original reward)
- threshold: 200.0

| seed | dev_score | held_out_mean | held_out_std | min | max | len | term | solved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_0 | 239.52 | 227.99 | 50.58 | -1.4 | 277.7 | 390.1 | 95/100 | yes |
| seed_1 | 170.40 | 156.36 | 84.68 | -41.1 | 278.2 | 608.9 | 73/100 | no |
| seed_2 | -110.09 | -109.52 | 10.87 | -135.0 | -81.0 | 68.9 | 100/100 | no |
| seed_3 | 115.51 | 107.66 | 90.37 | -130.1 | 234.9 | 691.5 | 71/100 | no |
| seed_4 | 259.50 | 251.06 | 45.01 | 67.2 | 313.7 | 288.2 | 95/100 | yes |

## Summary

- solved: 2/5
- held_out_mean: 126.71
- held_out_std: 143.85
