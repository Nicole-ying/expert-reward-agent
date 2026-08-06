# Subagent Research Signal

**训练过程**: Early (2749 eps): avg_len 111, avg_score 480, crash_rate 94%. Late (2751 eps): avg_len 138, avg_score 802, crash_rate 90%. Score rose but episodes remained short; evaluation fully consistent at len 1000, no crash, but mean eval reward -28.3, indicating convergence to stable hovering, not landing.

**组件健康**: landing_bonus nonzero 27% (active only on contact), others: total_reward, generated_reward, radial_reward, angle_penalty, vel_penalty, time_penalty all 100%. proximity_bonus 50.7%, engine_penalty 69.4%. No fully dead components.

**奖励对齐**: generated_reward per step mean 4.94 vs original_env_reward per step mean -0.64. Shaped reward is strongly positive while env reward is near zero, driving exploitation. Evaluation mean generated reward -28.3; training score high (802 lat,e) but evaluation negative, showing misalignment.

**异常检测**: Training monitor reports >90% crash rate and short episodes, but final evaluation runs all 1000 steps without crash. This gap suggests the deterministic policy learned to avoid early termination by hovering indefinitely, not by landing successfully. Training score increase (480->802) while evaluation score negative indicates possible overfitting to shaped bonuses during exploration.

**置信度**: `medium`
