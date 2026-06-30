# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|| 10 | distance_anchor + landing_quality_shaping + progress_delta_reward + stability_penalty | 28.04 | 188.21 | -160.17 | 735.00 | distance_anchor=-0.107 landing_quality_shaping=2.180 progress_delta_reward=0.237 stability_penalty=-0.197 | no_meaningful_improvement |

## Stable Lessons

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
