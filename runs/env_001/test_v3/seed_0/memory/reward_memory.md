# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -111.26 | -111.26 | 0.00 | 74.10 | progress_reward=0.160 soft_landing_bonus=0.011 stability_penalty=-0.340 | new_best |
| 2 | progress_reward + soft_landing_bonus + stability_penalty | -111.79 | -111.26 | -0.53 | 74.10 | progress_reward=0.242 soft_landing_bonus=0.010 stability_penalty=-0.331 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | 158.82 | 158.82 | 0.00 | 728.40 | landing_shaping=1.625 progress_reward=0.204 stability_penalty=-0.037 | new_best |
| 4 | landing_shaping + progress_reward + stability_penalty | -142.80 | 158.82 | -301.62 | 88.10 | landing_shaping=0.012 progress_reward=1.599 stability_penalty=-0.122 | no_meaningful_improvement |
| 5 | landing_shaping + progress_reward + stability_penalty | -190.67 | 158.82 | -349.50 | 84.30 | landing_shaping=0.010 progress_reward=5.119 stability_penalty=-0.068 | no_meaningful_improvement |
| 6 | distance_reward + progress_reward + stability_penalty | -109.08 | 158.82 | -267.91 | 83.60 | distance_reward=-0.093 progress_reward=8.255 stability_penalty=-0.133 | unsolved_stagnation_fresh_restart |
| 7 | progress_reward + soft_landing_bonus + stability_penalty | 146.57 | 158.82 | -12.25 | 564.60 | progress_reward=0.091 soft_landing_bonus=0.397 stability_penalty=-0.146 | no_meaningful_improvement |
| 8 | landing_shaping + progress_reward + stability_penalty | 100.05 | 158.82 | -58.77 | 846.40 | landing_shaping=1.370 progress_reward=0.228 stability_penalty=-0.022 | no_meaningful_improvement |

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- soft_landing_bonus trigger rate <1% indicates sparse proxy; consider continuous shaping or relaxed conditions
- landing_shaping with nonzero_rate < 1% is ineffective and should be replaced with denser proxy or removed
- stability_penalty coefficients should be low (angle_penalty <=0.05, angular_penalty <=0.02, speed_penalty <=0.1) to avoid dominance
- soft_landing_bonus should be continuous shaping (e.g., product of near_target, low_speed, stable_angle, both_contact) rather than sparse binary bonus
