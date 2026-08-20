# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: runs\env_001\exp_agent\seed_0\iter_04\generation\validations\reward_v4.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -1771.110673）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 提取当前状态 ──
    x, y = obs[0], obs[1]
    curr_vx, curr_vy = obs[2], obs[3]
    curr_angle = obs[4]

    # ── 提取下一时刻状态（反映动作后果）──
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 距离 ──
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # ── 速率标量 ──
    speed = (vx**2 + vy**2) ** 0.5
    curr_speed = (curr_vx**2 + curr_vy**2) ** 0.5

    # ── 主学习信号：potential-based shaping ──
    # Φ = -(distance + α*speed + β*|angle|)
    # F = γ * Φ(next) - Φ(curr)，天然引导接近+减速+姿态稳定
    alpha = 0.1
    beta = 0.2
    gamma = 0.99

    phi_curr = -(dist + alpha * curr_speed + beta * abs(curr_angle))
    phi_next = -(next_dist + alpha * speed + beta * abs(angle))
    potential_shaping = gamma * phi_next - phi_curr

    # ── 软着陆 proxy（连续乘积，阈值放宽）──
    # 用 bounded max(0, 1-x/threshold) 提供连续梯度
    prox_factor = max(0.0, 1.0 - next_dist / 1.0)               # 距离 < 1.0
    vel_factor = max(0.0, 1.0 - speed / 0.5)                    # 速率 < 0.5
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.3)             # 倾角 < 0.3
    ang_vel_factor = max(0.0, 1.0 - abs(ang_vel) / 0.3)         # 角速度 < 0.3
    contact_factor = min(left_contact, right_contact)            # 双脚触地

    soft_landing_proxy = (
        prox_factor * vel_factor * angle_factor * ang_vel_factor * contact_factor
    )

    # ── 轻量辅助惩罚 ──
    # 角速度惩罚：抑制剧烈旋转
    angular_vel_penalty = -0.01 * abs(ang_vel)
    # 能量惩罚：鼓励高效移动
    energy_penalty = -0.01 * (abs(vx) + abs(vy))

    # ── 总奖励 ──
    total_reward = (
        1.0 * potential_shaping +
        2.0 * soft_landing_proxy +
        1.0 * angular_vel_penalty +
        1.0 * energy_penalty
    )

    components = {
        "potential_shaping": potential_shaping,
        "soft_landing_proxy": soft_landing_proxy,
        "angular_vel_penalty": angular_vel_penalty,
        "energy_penalty": energy_penalty,
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一时刻的位置
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    # 速度、姿态、接触信息（使用 next_obs 更合理，反映动作导致的后果）
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：进度增量奖励 ----------
    # 每一步奖励“当前到目标的距离”与“下一步到目标的距离”之差
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5
    progress_delta = dist - next_dist   # 正值表示更靠近目标

    # ---------- 稳定/安全惩罚 ----------
    # 惩罚水平、垂直速度，以及姿态角和角速度，鼓励稳定接近
    stability_penalty = -(
        0.1 * abs(vx) +
        0.1 * abs(vy) +
        0.2 * abs(angle) +
        0.1 * abs(ang_vel)
    )

    # ---------- 任务完成近似信号（软着陆 proxy） ----------
    # 同时满足：靠近中心、低速、姿态稳定、双支撑脚接触，则给予小奖励
    dist_thresh = 0.5
    vel_thresh = 0.2
    angle_thresh = 0.1
    ang_vel_thresh = 0.1

    if (next_dist < dist_thresh and
        abs(vx) < vel_thresh and
        abs(vy) < vel_thresh and
        abs(angle) < angle_thresh and
        abs(ang_vel) < ang_vel_thresh and
        left_contact > 0.5 and right_contact > 0.5):
        soft_landing_proxy = 1.0
    else:
        soft_landing_proxy = 0.0

    # ---------- 总奖励 ----------
    w_progress = 5.0
    w_stab = 1.0      # stability_penalty 内部已含负号，直接加
    w_soft = 1.0

    total_reward = (
        w_progress * progress_delta +
        w_stab * stability_penalty +
        w_soft * soft_landing_proxy
    )

    components = {
        "progress_delta_reward": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-1771.110673, len=213.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angular_vel_penalty | -0.001080 | 0.001080 | 0.999997 | -0.001080 |
| energy_penalty | -0.020271 | 0.020271 | 0.999996 | -0.020271 |
| potential_shaping | 0.130193 | 0.130394 | 0.999998 | 0.130193 |
| soft_landing_proxy | 0.000705 | 0.000705 | 0.001510 | 0.000705 |
| total_reward | 0.110252 | 0.113534 | 1.000000 | 0.110252 |
| generated_reward | 0.110252 | 0.113534 | 1.000000 | 0.110252 |
| original_env_reward | -4.382563 | 4.911105 | 1.000000 | -4.382563 |

## Distribution
- score: mean=-1771.110673, min=-5945.060097, max=-574.418700
- episode_length: mean=213.500000
- early_terminal (<150 steps + score<-50): 6/10 (60%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.10 | -107.10 | 0.00 | 74.20 | progress_delta_reward=0.016 soft_landing_proxy=0.005 stability_penalty=-0.142 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | -111.45 | -107.10 | -4.35 | 74.10 | progress_delta_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.058 | no_meaningful_improvement |
| 3 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | -1771.11 | -107.10 | -1664.01 | 213.50 | angular_vel_penalty=-0.001 energy_penalty=-0.020 potential_shaping=0.130 soft_landing_proxy=0.001 | no_meaningful_improvement |
| 4 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | 179.44 | 242.09 | -62.64 | 812.60 | angular_vel_penalty=-0.001 energy_penalty=-0.007 potential_shaping=0.045 soft_landing_proxy=0.336 | stop_after_solved_drop_keep_best |
