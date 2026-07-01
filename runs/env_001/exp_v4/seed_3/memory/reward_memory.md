# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -111.98 | -111.98 | 0.00 | 70.60 | energy_penalty=-0.012 landing_bonus=0.005 progress_reward=0.032 stability_penalty=-0.243 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -112.65 | -111.98 | -0.67 | 70.60 | energy_penalty=-0.010 landing_bonus=0.014 progress_reward=0.161 stability_penalty=-0.120 | no_meaningful_improvement |
| 3 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -110.51 | -111.98 | 1.47 | 70.60 | energy_penalty=-0.013 landing_bonus=0.008 progress_reward=0.032 stability_penalty=-0.242 | no_meaningful_improvement |
| 4 | energy_penalty + landing_shaping + progress_reward + stability_penalty | -113.68 | -111.98 | -1.70 | 70.40 | energy_penalty=-0.008 landing_shaping=0.009 progress_reward=0.032 stability_penalty=-0.060 | unsolved_stagnation_fresh_restart |

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- stability_penalty coefficients should be tuned to avoid dominating the total reward
- progress_reward coefficient increase from 2.0 to 10.0 did not improve score, likely due to goal_near_oscillation
- stability_penalty coefficients reduction from (0.5,0.3,0.2) to (0.2,0.1,0.1) reduced penalty but did not improve overall performance
- landing_bonus condition relaxation (dist<0.8, speed<0.5, angle<0.3) and reward increase to 2.0 did not increase landing rate (nonzero rate still very low)
- progress_reward coefficient may need to be increased to drive learning
