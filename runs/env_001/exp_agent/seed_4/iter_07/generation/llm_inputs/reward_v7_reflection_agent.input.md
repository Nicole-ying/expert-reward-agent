# 上一轮奖励函数代码（该轮得分: 242.224287）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.03        # 【恢复】0.01→0.03，回到历史最佳配置

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_max = 2.0             # 距离阈值：超过此距离 proximity 归零
    V_max = 3.0             # 速度阈值：超过此速度 smooth_speed 归零
    A_max = 0.5             # 角度阈值（弧度，~28°）：超过此角度 upright 归零

    # 距离门控：stability penalty 只在接近地面时生效
    D_gate = 3.0            # 距离阈值：超过此距离 stability penalty 被门控衰减
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty (DISTANCE-GATED) ---
    # 门控因子：距离远 → 0（自由机动），距离近 → 1（精细控制）
    # 这样 agent 在高空下降时不会被速度/角度惩罚束缚
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    raw_penalty = w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel
    gate = max(0.0, 1.0 - dist_next / D_gate)  # ∈ [0, 1]，距离越近门越开

    stability_penalty = -raw_penalty * gate

    # --- Soft landing proxy (CONTINUOUS) ---
    # 三个 bounded 因子，每个 ∈ [0, 1]，乘积提供连续梯度
    proximity    = max(0.0, 1.0 - dist_next / D_max)
    smooth_speed = max(0.0, 1.0 - abs_v_sum / V_max)
    upright      = max(0.0, 1.0 - abs_angle / A_max)

    landing_bonus = w_landing * proximity * smooth_speed * upright

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "total_reward": total_reward,
        "progress_delta_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=242.224287, len=624.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002184 | 0.002364 | 0.999483 | 1.000000 |
| soft_landing_proxy | 0.024061 | 0.024061 | 0.993512 | 11.016659 |
| stability_penalty | -0.001867 | 0.001867 | 1.000000 | -0.854788 |
| total_reward | 0.024379 | 0.024603 | 1.000000 | 11.161870 |
| generated_reward | 0.024379 | 0.024603 | 1.000000 | 11.161870 |
| original_env_reward | 0.055818 | 1.986136 | 1.000000 | 25.556424 |

## Distribution
- score: mean=242.224287, min=137.361742, max=293.484305
- episode_length: mean=624.500000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.33 | -107.33 | 0.00 | 71.50 | progress_delta_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.014 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | 149.94 | 149.94 | 0.00 | 1000.00 | progress_delta_reward=0.002 soft_landing_proxy=0.234 stability_penalty=-0.002 | new_best |
| 3 | progress_delta_reward + soft_landing_proxy + stability_penalty | 165.26 | 165.26 | 0.00 | 1000.00 | progress_delta_reward=0.003 soft_landing_proxy=0.024 stability_penalty=-0.003 | new_best |
| 4 | progress_delta_reward + soft_landing_proxy + stability_penalty | 157.71 | 165.26 | -7.55 | 1000.00 | progress_delta_reward=0.003 soft_landing_proxy=0.023 stability_penalty=-0.004 | no_meaningful_improvement |
| 5 | progress_delta_reward + soft_landing_proxy + stability_penalty | -54.34 | 165.26 | -219.60 | 76.20 | progress_delta_reward=0.015 soft_landing_proxy=0.003 stability_penalty=-0.013 | no_meaningful_improvement |
| 6 | progress_delta_reward + soft_landing_proxy + stability_penalty | 242.22 | 242.22 | 0.00 | 624.50 | progress_delta_reward=0.002 soft_landing_proxy=0.024 stability_penalty=-0.002 | target_solved_new_best |
