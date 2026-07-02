# 上一轮奖励函数代码（该轮得分: 165.264900）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.03        # 软着陆连续代理奖励（从 0.3 大幅降低，避免劫持 progress）

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_max = 2.0             # 距离阈值：超过此距离 proximity 归零
    V_max = 3.0             # 速度阈值：超过此速度 smooth_speed 归零
    A_max = 0.5             # 角度阈值（弧度，~28°）：超过此角度 upright 归零
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty ---
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    stability_penalty = -(w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel)

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
score=165.264900, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002542 | 0.002752 | 0.999666 | 1.000000 |
| soft_landing_proxy | 0.023730 | 0.023730 | 0.991623 | 9.335099 |
| stability_penalty | -0.002815 | 0.002815 | 1.000000 | -1.107501 |
| total_reward | 0.023457 | 0.023986 | 1.000000 | 9.227598 |
| generated_reward | 0.023457 | 0.023986 | 1.000000 | 9.227598 |
| original_env_reward | 0.023377 | 1.835628 | 1.000000 | 9.195853 |

## Distribution
- score: mean=165.264900, min=134.698208, max=195.139252
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.33 | -107.33 | 0.00 | 71.50 | progress_delta_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.014 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | 149.94 | 149.94 | 0.00 | 1000.00 | progress_delta_reward=0.002 soft_landing_proxy=0.234 stability_penalty=-0.002 | new_best |
| 3 | progress_delta_reward + soft_landing_proxy + stability_penalty | 165.26 | 165.26 | 0.00 | 1000.00 | progress_delta_reward=0.003 soft_landing_proxy=0.024 stability_penalty=-0.003 | new_best |
