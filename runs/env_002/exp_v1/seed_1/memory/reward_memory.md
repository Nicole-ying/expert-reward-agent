# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + stability_penalty | -102.17 | -102.17 | 0.00 | 33.00 | energy_penalty=-0.016 progress_delta_reward=-0.005 stability_penalty=-0.134 | new_best |
| 2 | energy_penalty + progress_delta_reward + stability_penalty | -26.51 | -26.51 | 0.00 | 1600.00 | energy_penalty=-0.010 progress_delta_reward=0.009 stability_penalty=-0.049 | new_best |

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
