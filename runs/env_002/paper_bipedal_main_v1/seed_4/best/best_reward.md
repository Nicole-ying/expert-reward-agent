**evidence**: All 20 episodes terminate (reach terrain end), tight score range [262.9, 265.8], forward_reward dominates at 97.9% magnitude share (504.2 ep_sum), stability_penalty at 2.1% (−10.9), episode length 1153. External score is 264.4 vs target 300, gap of 35.6. No energy or action-cost signal exists in training reward.

**behavior_diagnosis**: Robot consistently completes the terrain with a stable gait at moderate speed (avg ~0.44). The persistently incomplete external score despite full terrain traversal suggests the agent uses unnecessarily forceful joint torques — energetically inefficient locomotion that an external metric likely penalizes.

**signal_completeness**: Forward progress (forward_speed) and stability constraints (tilt, angular velocity, vertical velocity) are present and well-scaled. Energy penalty (action magnitude cost) is absent — a standard locomotion component that plausibly contributes to the external evaluation metric. Action smoothness is also absent but not directly computable from current function signature.

**selected_level**: Level 2 — necessary signal (energy penalty) is missing from the reward structure.

**selected_intervention**: Add a single new component `energy_penalty = -0.05 * sum(action²)`, penalizing the sum of squared joint torques. All existing components and coefficients remain unchanged.

**falsifiable_hypothesis**: Introducing a modest action-magnitude cost will encourage the agent to discover more energy-efficient joint coordination patterns. If the external score includes an energy-efficiency term, this alignment should reduce the score gap without reducing forward speed or stability. The penalty is deliberately small (~4-11% of forward_reward) to avoid suppressing exploration of the existing successful gait.

**expected_next_round**: energy_penalty active_rate 100%, magnitude_share ~3-8% (comparable to or slightly above stability_penalty). External score should increase toward 280-295. Episode length may decrease slightly (faster, more efficient gait) or stay similar. Stability and forward_reward should remain at similar levels; if forward speed drops notably, the coefficient is too high.

**main_risk**: If the energy penalty coefficient is too strong relative to the forward incentive, the agent may reduce action magnitudes below what's needed for effective locomotion, slowing down and reducing external score. The 0.05 coefficient is deliberately conservative to mitigate this.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 前进速度奖励
    forward_speed = next_obs[2]
    forward_reward = forward_speed

    # 稳定性惩罚
    tilt_angle = next_obs[0]
    angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    tilt_penalty = -2.5 * (tilt_angle ** 2)
    angular_vel_penalty = -0.5 * (angular_vel ** 2)
    vertical_vel_penalty = -2.5 * (vertical_vel ** 2)
    stability_penalty = tilt_penalty + angular_vel_penalty + vertical_vel_penalty

    # 能量消耗惩罚（动作力矩平方和）
    energy_penalty = -0.05 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty
    }

    return float(total_reward), components
```