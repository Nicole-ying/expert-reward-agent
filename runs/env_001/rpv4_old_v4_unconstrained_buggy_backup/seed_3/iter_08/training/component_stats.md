# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 14.800597 | 14.800597 | 0.227701 | 65.000000 | 65.000000 | 0.000000 | 65.000000 | 1003520 |
| component.goal_proximity | 0.014304 | 0.015926 | 0.998575 | 0.014324 | 0.015949 | -0.073564 | 0.080763 | 1003520 |
| component.orientation_penalty | -0.006745 | 0.006745 | 1.000000 | -0.006745 | 0.006745 | -0.326467 | -0.000000 | 1003520 |
| component.total_reward | 4.486639 | 4.621442 | 1.000000 | 4.486639 | 4.621442 | -1.093853 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.077670 | 0.077670 | 0.998390 | -0.077795 | 0.077795 | -1.057634 | -0.000000 | 1003520 |
| generated_reward | 4.486639 | 4.621442 | 1.000000 | 4.486639 | 4.621442 | -1.093853 | 20.000000 | 1003520 |
| original_env_reward | -0.271450 | 2.359915 | 1.000000 | -0.271450 | 2.359915 | -100.000000 | 128.588741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 2285.413306 | 2285.413306 | 0.000000 | 56420.000000 | 6448 |
| goal_proximity | 2.224946 | 2.224946 | 0.029242 | 2.814522 | 6448 |
| orientation_penalty | -1.048934 | 1.048934 | -20.141650 | -0.017337 | 6448 |
| total_reward | 692.717548 | 700.172169 | -28.885299 | 17355.937784 | 6448 |
| velocity_penalty | -12.085536 | 12.085536 | -30.842703 | -5.240037 | 6448 |
