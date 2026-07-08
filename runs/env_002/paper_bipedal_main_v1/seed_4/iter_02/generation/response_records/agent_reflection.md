# Response Record

**evidence**: score=103 with range [-80,287] and all 20 episodes terminated (0 truncated) indicates high-variance walking where some episodes nearly reach target but others fail early; stability_penalty magnitude is only 5.6% of total reward (ratio to forward ≈ 0.06), an order of magnitude below the "light constraint" reference point for dense signals.

**behavior_diagnosis**: The agent has learned an aggressive, inconsistent gait — when it works, it scores near 287; when it fails, the robot likely lunges forward and falls early (score ≈ -80). The weak stability penalty fails to penalize risky postures and bouncing enough to discourage fall-prone behavior.

**signal_completeness**: Forward progress and stability are both present and reachable. Energy penalty is explicitly mentioned in the task secondary objective but not yet implemented. The stability component has the correct sign and mathematical form (quadratic) but its coefficient is too weak to function as an effective constraint — the signal is present in name but negligible in practice.

**selected_level**: Level 1 — the stability_penalty component has correct sign and quadratic shape (penalizes large deviations heavily), but its effective magnitude relative to forward_reward is too small (~6%) to shape behavior. Single coefficient adjustment is sufficient.

**selected_intervention**: Increase all three sub-coefficients within stability_penalty by approximately 5×: tilt_penalty from -0.5 to -2.5, angular_vel_penalty from -0.1 to -0.5, vertical_vel_penalty from -0.5 to -2.5. This targets a |stability|/forward ratio of ~0.3, providing meaningful constraint without dominating the forward signal.

**falsifiable_hypothesis**: A 5× stronger stability penalty will suppress the aggressive, fall-prone behaviors that cause the low-scoring episodes, reducing score variance and raising the mean. If the hypothesis is correct, the score minimum will rise (fewer catastrophic early falls) and the mean will increase toward the current maximum. If incorrect, either the agent becomes overcautious (mean drops, max drops, episode length increases with low forward speed) or the penalty remains insufficient (no change).

**expected_next_round**: Stability penalty magnitude_share should rise from ~5.6% to ~25-35%. Score variance should decrease (tighter range). Mean score should increase, primarily from eliminating the negative-score tail. Max score may stay similar or slightly decrease if the agent is slightly less aggressive. Episode length should become more consistent.

**main_risk**: Over-constraint — the agent may converge to a "stand still" local optimum (zero tilt, zero velocity → zero penalty, zero reward) if the stability penalty overwhelms the forward incentive during early exploration.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 前进速度奖励（正值表示向前，直接作为主学习信号）
    forward_speed = next_obs[2]
    forward_reward = forward_speed  # 系数 1.0

    # 稳定性惩罚：惩罚大倾角、高角速度和明显垂直速度（弹跳）
    # 系数从 0.5/0.1/0.5 提升到 2.5/0.5/2.5（~5x），使稳定性约束更有效
    tilt_angle = next_obs[0]
    angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    tilt_penalty = -2.5 * (tilt_angle ** 2)
    angular_vel_penalty = -0.5 * (angular_vel ** 2)
    vertical_vel_penalty = -2.5 * (vertical_vel ** 2)
    stability_penalty = tilt_penalty + angular_vel_penalty + vertical_vel_penalty

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```
