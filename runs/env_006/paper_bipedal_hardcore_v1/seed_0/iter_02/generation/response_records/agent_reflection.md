# Response Record

# 设计理由
当前策略的外部得分为 -57.47，但内部奖励中前进速度组件平均每回合有 80.12（per‑step 约 0.48），而姿势惩罚仅 -1.88（per‑step 约 -0.011），惩罚强度严重不足。  
终止模式：20/20 的 episode 以 terminated 结束（无 truncation），平均长度 166.9 步；结合环境事实（摔倒终止）可推断，**大多数 episode 因 hull 倾倒终止**。  
信号覆盖审计：未使用 leg_contact、vertical_speed、lidar 等观测，但直接导致失败的核心信号（hull_angle、hull_angular_velocity）已在惩罚组件中使用，只是**惩罚的数学形式和量级无法阻止危险姿态**。  
因此本轮的修改思路是**保持主奖励（gated_forward）不变**，将 **posture_hinge_penalty 组件从“线性 hinge”变换为“无门槛的二次惩罚”**，使惩罚在角度或角速度稍有增大时就明显提高，而在正常行走时仍控制在安全范围。  
**量级校准**：  
- 主信号 per‑step ≈ 0.48  
- 新惩罚的 per‑step 目标：正常姿态时 < 0.24（0.5 倍以内），危险姿态时 > 0.5  
- 系数选取：`w_angle = 2.0`，`w_angvel = 1.0`  
- 假设正常 hull_angle ≈ 0.1 rad、angvel ≈ 0.2 rad/s → 惩罚 ≈ 2*0.01 + 1*0.04 = 0.06，远低于上限  
- 当 hull_angle = 0.5 时惩罚 ≈ 0.5，能强烈抑制危险行为  

**变换依据**：证据为“约束在无关阶段妨碍探索（惩罚太弱等于没有）”，采用 **水平 2：结构变换（线性 hinge → 二次无界惩罚）**。  
**副作用控制**：惩罚过强可能导致 agent 过度保守，因此保持 gated_forward 对速度的凸奖励，维持前进动力。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with soft health gate based on hull stability
    - Constraint: quadratic stability penalty on tilt and angular velocity
    """

    # ==================== Extract signals ====================
    # Next hull state (immediate future)
    next_hull_angle = next_obs[0]
    next_hull_angvel = abs(next_obs[1])   # angular velocity magnitude

    # Forward velocity (next step)
    horizontal_speed = next_obs[2]

    # ==================== Constants ====================
    # Gate thresholds (unchanged)
    TILT_CRITICAL = 0.6
    TILT_WARNING_START = 0.25
    TILT_WARNING_MARGIN = 0.35

    # Weights
    FORWARD_WEIGHT = 2.0
    STABILITY_ANGLE_WEIGHT = 2.0
    STABILITY_ANGVEL_WEIGHT = 1.0

    # ==================== Component A: Forward progress with soft health gate ====================
    abs_tilt = abs(next_hull_angle)
    if abs_tilt <= TILT_WARNING_START:
        gate = 1.0
    elif abs_tilt >= TILT_CRITICAL:
        gate = 0.0
    else:
        gate = (TILT_CRITICAL - abs_tilt) / TILT_WARNING_MARGIN

    forward_reward = FORWARD_WEIGHT * horizontal_speed ** 2
    gated_forward = gate * forward_reward

    # ==================== Component B: Quadratic stability penalty ====================
    # Penalize any deviation from upright and any angular velocity
    # Quadratic form gives mild penalty near zero and rapid growth as tilt/velocity increase
    angle_penalty = STABILITY_ANGLE_WEIGHT * (next_hull_angle ** 2)
    angvel_penalty = STABILITY_ANGVEL_WEIGHT * (next_hull_angvel ** 2)
    stability_penalty = -(angle_penalty + angvel_penalty)

    # ==================== Total reward ====================
    total_reward = gated_forward + stability_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'stability_quad_penalty': stability_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 摔倒信号已在惩罚中使用，但量级和数学形态（线性 hinge）不足以阻止危险姿态；leg_contact/vertical_speed 等辅助信号暂未引入，属于“信号齐全但校准问题” → 优先修改现有组件的惩罚强度和形状。
- **behavior**: agent 在前进鼓励下频繁出现过大倾斜导致摔倒终止，外部得分因此极负；内部 punish 极弱，未能教导稳定性。
- **signal**: posture_hinge_penalty 过弱（per‑step ~0.011 vs 主信号 0.48），无法阻止 hull_angle 超限。
- **level**: Level 2
- **hypothesis**: 改用二次惩罚后，即使小角度也会产生可感知的 penalty，训练会学会将 tilt 和 angvel 维持在低水平，减少摔倒，从而延长 episode 并提高外部得分。
- **risk**: 惩罚过强可能抑制正常行走所需的微小倾角，导致进步速度下降；但 gated forward 仍提供强动力，预计能平衡。
