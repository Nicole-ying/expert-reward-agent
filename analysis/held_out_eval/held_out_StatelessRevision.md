# Held-Out Evaluation: StatelessRevision

- episodes: 100
- seeds: 50000..50099
- env: LunarLander-v3 (original reward)
- threshold: 200.0

| seed | dev_score | held_out_mean | held_out_std | min | max | len | term | solved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_0 | 25.37 | 25.76 | 15.01 | -14.9 | 64.4 | 216.4 | 100/100 | no |
| seed_1 | -5.05 | -6.46 | 113.77 | -537.2 | 253.6 | 809.3 | 42/100 | no |
| seed_2 | -10.11 | -24.25 | 129.80 | -210.6 | 194.1 | 748.6 | 82/100 | no |
| seed_3 | 151.32 | 148.44 | 52.88 | 17.4 | 304.5 | 869.6 | 17/100 | no |
| seed_4 | 58.86 | 78.09 | 128.50 | -215.1 | 229.1 | 592.5 | 93/100 | no |

## Summary

- solved: 0/5
- held_out_mean: 44.32
- held_out_std: 70.01
