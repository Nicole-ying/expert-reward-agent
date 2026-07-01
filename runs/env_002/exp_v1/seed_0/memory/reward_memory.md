# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + angular_penalty + energy_penalty + forward_reward | 302.90 | 302.90 | 0.00 | 994.80 | angle_penalty=-0.053 angular_penalty=-0.000 energy_penalty=-0.020 forward_reward=1.043 | target_solved_new_best |
| 2 | angle_penalty + angular_penalty + energy_penalty + forward_reward + landing_bonus | 263.20 | 302.90 | -39.70 | 969.70 | angle_penalty=-0.020 angular_penalty=-0.000 energy_penalty=-0.021 forward_reward=0.468 landing_bonus=0.133 | stop_after_solved_drop_keep_best |

## Stable Lessons

- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- forward_reward指数形式在速度高时产生大奖励，需注意震荡风险
- 当前骨架有效，无需大幅改动
