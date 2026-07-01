现在我来分析当前状态：

**当前表现：** score=242.09, len=441.3, 0% early terminal, 0 errors ✅

**组件分析：**
- `soft_landing_proxy`: mean=0.829, nonzero_rate=98.3%, ratio_to_progress=0.829 — **严重主导**总奖励
- `potential_shaping`: mean=0.021, nonzero_rate=100% — 被淹没
- `angular_vel_penalty`: mean=-0.0006 — 几乎无影响
- `energy_penalty`: mean=-0.0075 — 几乎无影响

**问题：** soft_landing_proxy 贡献了 98% 的总奖励（0.829/0.843），而 shaping 信号只有 2.5%。这可能导致 agent 只关注 landing proxy 而忽略真正的 landing 质量。虽然 score 已经为正，但 original_env_reward 仍为负（-0.117），说明 agent 在 exploit proxy 而非真正完成任务。

**修改方案：**
1. 降低 soft_landing_proxy 权重，从 1.5 降到 0.5
2. 提高 potential_shaping 权重，从 2.0 提到 5.0，让 shaping 信号更有影响力
3. 保持其他不变，观察是否改善 original_env_reward

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断与修改说明：
    # 当前 score=242.09，但 original_env_reward=-0.117（负值），
    # 说明 agent 在 exploit soft_landing_proxy 而非真正完成任务。
    # 
    # 问题：soft_landing_proxy 贡献 98% 总奖励（ratio_to_progress=0.829），
    # potential_shaping 仅贡献 2.5%。agent 被 proxy 信号主导，
    # 忽略了真正的 landing 质量（减速、姿态稳定）。
    #
    # 修改方案：
    # 1. 降低 soft_landing_proxy 权重 1.5 → 0.5，减少 proxy 主导
    # 2. 提高 potential_shaping 权重 2.0 → 5.0，让 shaping 信号更有影响力
    # 3. 保持 angular_vel_penalty 和 energy_penalty 不变
    # 
    # 目标：让 shaping 信号（接近+减速+稳定）与 landing proxy 平衡，
    # 引导 agent 学习真正的 soft landing。
    # ============================================================
    
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    
    current_angle = abs(obs[4])
    next_angle = abs(next_obs[4])
    
    # ============================================================
    # 1. 主学习信号: potential_based_shaping（权重提高）
    #    势能 Φ = -(dist + 0.8*speed + 0.5*|angle|)
    #    shaping = γ * Φ(next) - Φ(obs), γ=0.99
    # ============================================================
    gamma = 0.99
    phi_obs = -(current_dist + 0.8 * current_speed + 0.5 * current_angle)
    phi_next = -(next_dist + 0.8 * next_speed + 0.5 * next_angle)
    
    shaping_scale = 5.0  # 从 2.0 提高到 5.0
    potential_shaping = shaping_scale * (gamma * phi_next - phi_obs)

    # ============================================================
    # 2. 稳定约束: angular_vel_penalty（距离门控）
    # ============================================================
    angular_vel = abs(next_obs[5])
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * angular_vel * distance_gate

    # ============================================================
    # 3. 任务完成proxy: soft_landing_proxy（权重降低）
    #    加权和形式，每个因子独立贡献梯度
    # ============================================================
    dist_factor = max(0.0, 1.0 - next_dist / 0.8)
    speed_factor = max(0.0, 1.0 - next_speed / 0.5)
    angle_factor = max(0.0, 1.0 - next_angle / 0.3)
    contact_factor = min(1.0, (next_obs[6] + next_obs[7]) / 2.0)
    
    landing_bonus_weight = 0.5  # 从 1.5 降低到 0.5
    soft_landing_proxy = landing_bonus_weight * (
        0.4 * dist_factor + 
        0.3 * speed_factor + 
        0.2 * angle_factor + 
        0.1 * contact_factor
    )

    # ============================================================
    # 4. 动作代价: energy_penalty（小权重）
    # ============================================================
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * engine_use

    # ============================================================
    # 总奖励
    # ============================================================
    total_reward = potential_shaping + angular_vel_penalty + soft_landing_proxy + energy_penalty

    components = {
        "potential_shaping": potential_shaping,
        "angular_vel_penalty": angular_vel_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```