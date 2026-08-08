# Held-Out Evaluation: IndependentGen

- episodes: 100
- seeds: 50000..50099
- env: LunarLander-v3 (original reward)
- threshold: 200.0

| seed | dev_score | held_out_mean | held_out_std | min | max | len | term | solved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_0 | -13.11 | -14.88 | 29.39 | -111.3 | 80.7 | 990.1 | 6/100 | no |
| seed_1 | 170.78 | 184.68 | 61.56 | -101.2 | 256.6 | 603.9 | 93/100 | no |
| seed_2 | 55.07 | 58.08 | 104.42 | -83.9 | 289.8 | 153.1 | 98/100 | no |
| seed_3 | -106.69 | -109.41 | 19.43 | -144.5 | -67.3 | 69.1 | 100/100 | no |
| seed_4 | -109.73 | -109.19 | 10.98 | -141.4 | -81.0 | 68.9 | 100/100 | no |

## Summary

- solved: 0/5
- held_out_mean: 1.86
- held_out_std: 124.08
