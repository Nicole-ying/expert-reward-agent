# Held-Out Evaluation: CREATE

- episodes: 100
- seeds: 50000..50099
- env: LunarLander-v3 (original reward)
- threshold: 200.0

| seed | dev_score | held_out_mean | held_out_std | min | max | len | term | solved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_0 | 224.21 | 231.29 | 29.24 | 101.2 | 285.1 | 408.6 | 100/100 | yes |
| seed_1 | 240.60 | 231.61 | 67.03 | -176.2 | 289.1 | 373.4 | 100/100 | yes |
| seed_2 | 220.24 | 214.14 | 22.57 | 113.7 | 255.2 | 526.8 | 99/100 | yes |
| seed_3 | 253.71 | 248.71 | 20.94 | 202.7 | 289.5 | 363.4 | 100/100 | yes |
| seed_4 | 206.14 | 232.40 | 50.16 | 123.7 | 303.1 | 522.1 | 77/100 | yes |

## Summary

- solved: 5/5
- held_out_mean: 231.63
- held_out_std: 12.23
