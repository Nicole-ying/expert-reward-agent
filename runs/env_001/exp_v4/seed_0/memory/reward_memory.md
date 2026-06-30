# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_reward + soft_landing_bonus + stability_penalty | -111.31 | -111.31 | 0.00 | 74.10 | energy_penalty=-0.008 progress_reward=0.161 soft_landing_bonus=0.010 stability_penalty=-0.241 | new_best |

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
