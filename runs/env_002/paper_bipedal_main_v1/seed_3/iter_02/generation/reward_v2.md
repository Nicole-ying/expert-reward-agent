## 1. evidence
score=289.99, len=1026.4, all 20 episodes terminated by body-fall without reaching terrain end; forward_reward dominates at 99.2% magnitude share while angle_penalty sits at only 0.6% (ratio |angle|/|forward| ≈ 0.0065, far below even the 0.1 "light constraint" reference), and the vert_penalty is negligible at 0.2%; the agent maintains good average posture (hull_angle ≈ 3.3°) but still falls every time, suggesting a missing signal rather than mere scale imbalance.

## 2. behavior_diagnosis
The agent walks forward stably for ~1026 steps then invariably falls — it has learned a "go fast until you collapse" strategy because the reward cares only about forward speed with effectively invisible stability constraints; there is no incentive whatsoever for efficient, sustainable gait patterns.

## 3. signal_completeness
The reward has forward progress (forward_velocity) and stability constraints (angle, vertical), but entirely lacks the energy-efficiency objective that the task explicitly states — "最小化能耗" has zero representation in the reward function, and this missing responsibility likely prevents the agent from discovering smooth, sustainable gaits that avoid gradual instability buildup.

## 4. selected_level
Level 2 — a necessary task responsibility (energy minimization) is completely absent from the reward, qualifying as "必要职责缺失."

## 5. selected_intervention
Add a new `energy_penalty` component: `-0.05 * sum(action²)`, penalizing large joint torques to encourage efficient movement; all three existing components remain untouched.

## 6. falsifiable_hypothesis
By making action cost visible, the agent should learn smoother, lower-torque gaits that dissipate less energy per step — these more controlled motion patterns should reduce the gradual instability accumulation that currently causes the universal collapse at ~1026 steps, leading to longer survival and higher scores.

## 7. expected_next_round
`energy_penalty` should show ~100% active_rate with episode_sum_mean roughly in the range of -50 to -100 (given average |action| ≈ 0.5 yielding ~1.0 raw sum per step × 0.05 × ~1026 steps ≈ -51); episode length should increase if the hypothesis holds, and score should eventually exceed 290; a temporary score dip is acceptable as the agent adapts.

## 8. main_risk
If w_energy=0.05 is too aggressive relative to the forward signal, the agent may converge to a "freeze" or minimal-movement strategy that avoids penalties but makes no forward progress; conversely if too weak, the new component will have no behavioral effect.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 核心进步信号：鼓励沿前进方向快速移动
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # 姿态稳定约束：惩罚躯干倾角偏离直立
    hull_angle = next_obs[0]
    w_angle = 1.0
    angle_penalty = -w_angle * (hull_angle ** 2)

    # 垂直稳定约束：抑制跳跃或剧烈起伏
    vertical_velocity = next_obs[3]
    w_vert = 0.5
    vert_penalty = -w_vert * (vertical_velocity ** 2)

    # 能耗约束：惩罚过大的关节力矩，鼓励高效低能耗步态
    energy_penalty = -0.05 * sum(a ** 2 for a in action)

    total_reward = forward_reward + angle_penalty + vert_penalty + energy_penalty

    components = {
        'forward_reward': forward_reward,
        'angle_penalty': angle_penalty,
        'vert_penalty': vert_penalty,
        'energy_penalty': energy_penalty
    }

    return float(total_reward), components
```