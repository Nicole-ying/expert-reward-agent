```json
{
  "evidence": "score=-7.77, len=972, terminated=2/20(truncated=18), goal_proximity mean=-116.87→avg dist≈0.35(agent IS near target), contact_reward active_rate仅1.1%且signed_share仅10.7%, speed_quality×angle_quality乘积使中速/中角度时奖励塌缩至≈0, velocity_penalty和orientation_penalty已独立约束速度与姿态, 上一轮添加quality乘子后contact_reward变得更稀疏。",
  "behavior_diagnosis": "agent已学会靠近目标垫(均距≈0.35)并保持低速度, 但因contact_reward在产品塌缩下几乎不可达, agent无法将双足触垫与正奖励关联, 导致在目标上方悬停至truncated, 未能完成着陆。",
  "signal_completeness": "goal_proximity、velocity_penalty、orientation_penalty职责完备且可达; contact_reward职责正确(both_legs×proximity)但speed_quality×angle_quality乘子使信号不可达, 速度与姿态约束已由其独立惩罚项覆盖, 无需在contact中重复。",
  "selected_level": "Level 2",
  "selected_intervention": "product_to_noncollapsing_joint: 移除contact_reward中的speed_quality和angle_quality乘子, 保留both_legs_contact×proximity结构, w_contact从3.0维持不变(实际有效权重因去除塌缩因子将大幅提升)。",
  "falsifiable_hypothesis": "去除导致乘积塌缩的quality乘子后, contact_reward的active_rate应显著上升(不再要求速度与角度同时近乎完美), agent能发现双足触垫的正奖励, 从悬停转为主动完成着陆, terminated比例和score应同步改善。",
  "expected_next_round": "contact_reward的active_rate从~1%升至>5%, episode_sum_mean正向增大, terminated比例上升, score均值提高, goal_proximity仍保持接近目标(均距<0.5)。",
  "main_risk": "去除速度/角度质量门控后, agent可能在高速或大角度下触垫获得contact_reward, 但velocity_penalty和orientation_penalty的独立约束应抑制此行为; 若出现冲撞着陆需后续调整velocity_penalty尺度。"
}
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # Hyperparameters
    w_goal = 1.0
    alpha_proximity = 5.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 3.0

    # Distance to target center (squared)
    dist_sq = x**2 + y**2

    # Soft proximity weight: ~1 when close to target, ~0 when far
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: reward both legs touching the pad, gated by proximity
    # Removed speed_quality and angle_quality multipliers that caused product collapse;
    # speed and angle are independently constrained by velocity_penalty and orientation_penalty.
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```