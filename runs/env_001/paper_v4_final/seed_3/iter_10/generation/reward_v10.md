# 设计理由
本轮修改组件：将僵尸组件 `angle_term_penalty`（hinge，阈值为 0.7 rad，从未触发，active_rate=0%）替换为连续二次角度惩罚 `angle_penalty`。  
**数学形式**：`-w_angle * (abs(next_obs[4]) ** 2)`，系数 `w_angle=0.5`。  
**校准依据**：  
- 当前主信号 `progress_gated` 每步约 0.21，`angle_penalty` 在角度 0.3 rad（常见安全值）时每步约 -0.045，惩罚/主信号比 ≈0.21 < 0.3；角度 0.5 rad 时每步 -0.125，仍可控。总惩罚负担（含 `angvel` 和 `lateral`）保持在主信号的 0.3x 以内。  
- 原 hinge 在终止边界附近才激活，而 agent 角度往往直接越过预警区并触发 episode 终止，未能获得有效梯度。连续二次惩罚从角度为 0 即开始提供负梯度，使 agent 在角度轻微增大时就能感知代价，更早学习收敛。  

保留原有的 `angle_gate` 不变（在 progress 奖励上乘性抑制大角度），双重信号确保角度控制。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_lc = next_obs[6]
    n_rc = next_obs[7]

    # ---------- 1. progress towards origin ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 30.0

    # ---------- 2. attitude gate ----------
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty ----------
    w_lat_pos = 0.08
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact landing proxy ----------
    mean_contact = (n_lc + n_rc) / 2.0
    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs_angle)
    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.5
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- 6. angle penalty (continuous quadratic, replaces inactive hinge) ----------
    w_angle = 0.5
    angle_penalty = -w_angle * (abs_angle ** 2)

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward
                    + angle_penalty)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测已被使用，但 `angle_term_penalty` 僵尸组件从未激活，因其 hinge 阈值（0.7 rad）在终止前无法提供有效梯度。  
- **behavior**: agent 在早期（143 步）因角度过大终止，期间 progress signal 被 gate 压制，而角度本身无直接惩罚导致角度失控。  
- **signal**: 缺少从零角度开始的连续角度代价，角度控制仅依赖 gate 的间接抑制和稀疏着陆奖励的形状因子。  
- **level**: Level 2  
- **hypothesis**: 连续二次角度惩罚将角度控制转化为每步都有的直接负梯度，使 agent 更早学会压低角度，减少终止率，逐步恢复生存长度并稳定进步。  
- **risk**: 角度惩罚可能轻微削弱 progress 的吸引力，如果系数过强可能导致 agent 为避免任何角度而停滞；当前系数使惩罚与主信号比例安全，风险可控。