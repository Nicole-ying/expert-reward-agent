# 设计理由
**改动组件**：将 `angle_penalty` 替换为 `angvel_penalty`，使用之前被忽略的 `angular_velocity`（obs[5]）信号。  
**为什么改**：第0步审计发现 `angular_velocity` 未被使用，而原有的 `angle_penalty` 触发率仅 1.8%（绝大部分时间角度在安全区内），实际作用接近于零。引入角速度监督可以填补信号缺口，在训练后期进一步优化姿态平稳性（减少不必要的旋转震荡），同时避免影响已收敛的着陆行为。  
**数学形式**：采用 hinge 惩罚 `-w * max(0, abs(angular_velocity) - safe_threshold)`，阈值设在不影响正常调整但能抑制剧烈旋转的范围（`safe_angvel=0.5`），系数 `0.05` 使 per‑step 惩罚量级远小于主奖励（预计 <0.01）。  
**系数校准**：
- 主信号 `landing_incentive` 的 per‑step ≈ 134.3/416.7 ≈ 0.322
- 新惩罚预计最大 per‑step（在角速度很大时）≈ 0.05 * 假设 max(0, high_value) ≤ 0.05，远低于 0.3×0.322 = 0.097
- 即使全时激活，总惩罚负担仍远低于 0.5×主信号，安全

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Progress: distance reduction ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = dist - next_dist

    # --- Landing incentive: only when legs touch ground ---
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    speed = (next_vx ** 2 + next_vy ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    w_landing = 1.0
    landing_incentive = leg_contact * w_landing / (1.0 + next_dist * 5.0) * speed_factor

    # --- Angular velocity penalty (replaces body angle penalty) ---
    w_angvel = 0.05
    safe_angvel = 0.5
    angvel_error = abs(next_angvel) - safe_angvel
    angvel_penalty = -w_angvel * angvel_error if angvel_error > 0 else 0.0

    # --- Total reward ---
    total_reward = w_progress * progress + landing_incentive + angvel_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angvel_penalty": angvel_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: `angular_velocity`(obs[5]) 未被使用，`angle_penalty` 几乎不触发（1.8%），存在姿态震荡可能未被约束的信号缺口。  
- **behavior**: agent 已学会快速安全着陆（score>250，len≈417），主要靠 `landing_incentive` 引导，但过程中可能会有一些不必要的旋转。  
- **signal**: 缺少对角速度的约束；原有的角度惩罚过于宽松且覆盖不足。  
- **level**: Level 2 — 结构变换（替换组件，使用新观测）  
- **hypothesis**: 通过轻微抑制高角速度，可让着陆姿态更平稳，减少无用震荡，且不会损害已学会的快速着陆策略。  
- **risk**: 阈值/系数若不当可能限制必要的姿态调整，导致着陆过程变慢；但 0.5 的容忍度足够保守，系数 0.05 量级很小，副作用概率低。