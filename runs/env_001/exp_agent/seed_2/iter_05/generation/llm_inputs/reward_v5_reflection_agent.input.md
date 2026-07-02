# ⚠️ 上一版代码验证失败
错误信息：Reward v5 failed validation: runs\env_001\exp_agent\seed_2\iter_05\generation\validations\reward_v5.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 137.073696）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励（不动）──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束（不动）──
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y)) \
                        -0.002 * abs(angle) \
                        -0.002 * abs(angular_vel)

    # ── 4. 连续软着陆引导信号 ──
    # 上一轮：coefficient=0.5, prox_threshold=0.3 → mean=0.237, ratio=83x progress
    # 本轮：coefficient→0.08, prox_threshold→0.2，目标 ratio≈5-10x
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5

    prox_factor = max(0.0, 1.0 - dist_new / 0.2)       # dist=0→1, dist≥0.2→0（收紧）
    speed_factor = max(0.0, 1.0 - speed / 0.5)          # speed=0→1, speed≥0.5→0
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.3)    # angle=0→1, |angle|≥0.3→0
    leg_factor = 0.5 * (left_contact + right_contact)   # 两腿→1, 单腿→0.5, 无→0

    # 系数从 0.5 降至 0.08，大幅削弱 landing 信号的主导地位
    soft_landing_continuous = 0.08 * prox_factor * speed_factor * angle_factor * leg_factor

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_continuous

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_continuous": soft_landing_continuous,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励 ──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束（系数从 0.01 → 0.002，降低 5 倍）──
    # 上一轮：penalty ratio = -0.88，严重压制 progress 信号
    # 本轮目标：ratio ≈ -0.18，让 agent 敢于机动
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y)) \
                        -0.002 * abs(angle) \
                        -0.002 * abs(angular_vel)

    # ── 4. 软着陆近似信号（本轮不动）──
    near_target = dist_new < 0.15
    low_speed = (vel_x ** 2 + vel_y ** 2) ** 0.5 < 0.2
    upright = abs(angle) < 0.1
    both_legs_down = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.5 if (near_target and low_speed and upright and both_legs_down) else 0.0

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_proxy

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=137.073696, len=921.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.003239 | 0.003473 | 0.999699 | 0.003239 |
| soft_landing_continuous | 0.041987 | 0.041987 | 0.630045 | 0.041987 |
| stability_penalty | -0.000805 | 0.000805 | 1.000000 | -0.000805 |
| total_reward | 0.044421 | 0.044768 | 1.000000 | 0.044421 |
| generated_reward | 0.044421 | 0.044768 | 1.000000 | 0.044421 |
| original_env_reward | -0.032718 | 1.397779 | 1.000000 | -0.032718 |

## Distribution
- score: mean=137.073696, min=35.761577, max=183.794941
- episode_length: mean=921.600000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -105.90 | -105.90 | 0.00 | 72.00 | progress=0.016 soft_landing_proxy=0.002 stability_penalty=-0.014 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | 187.93 | 187.93 | 0.00 | 694.50 | progress=0.003 soft_landing_proxy=0.257 stability_penalty=-0.001 | new_best |
| 3 | progress + soft_landing_continuous + stability_penalty | 143.84 | 187.93 | -44.08 | 1000.00 | progress=0.003 soft_landing_continuous=0.237 stability_penalty=-0.001 | no_meaningful_improvement |
| 4 | progress + soft_landing_continuous + stability_penalty | 137.07 | 187.93 | -50.85 | 921.60 | progress=0.003 soft_landing_continuous=0.042 stability_penalty=-0.001 | no_meaningful_improvement |
