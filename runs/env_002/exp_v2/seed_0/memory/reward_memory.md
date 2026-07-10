# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -102.55 | -102.55 | 0.00 | 33.00 | energy_penalty=-0.017 progress_delta_reward=0.003 soft_landing_bonus=0.000 stability_penalty=-0.137 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | 11.95 | 11.95 | 0.00 | 863.30 | energy_penalty=-0.015 progress_delta_reward=0.012 soft_landing_bonus=0.073 stability_penalty=-0.004 | new_best |
| 3 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -76.01 | 11.95 | -87.96 | 104.40 | energy_penalty=-0.016 progress_delta_reward=0.051 soft_landing_bonus=0.008 stability_penalty=-0.006 | no_meaningful_improvement |

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- progress_delta_reward coefficient should be >= 5 to drive learning
- stability_penalty should be stronger to prevent instability
