# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: runs\env_001\exp_agent\seed_2\iter_08\generation\validations\reward_v8.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 234.795902）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 读取当前状态
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 从 obs 读取上一步位置，计算距离变化
    px = obs[0]
    py = obs[1]

    prev_dist = (px**2 + py**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5

    # === 1. 主学习信号：进度增量（向目标靠近为正） ===
    # 这是 potential-based shaping 的特例：Φ = -distance, γ = 1
    # 正奖励 = 距离减小，天然零均值，提供清晰的梯度方向
    progress = prev_dist - curr_dist

    # === 2. 连续着陆质量信号（始终激活，峰值在着陆条件） ===
    # 三个 bounded 因子：proximity, speed, upright，都是 1/(1+kx) 形式
    # 自动 bounded 在 [0,1]，无需手动调尺度

    # 接近度：距离越近越高，k=5 使 distance=1 时约为 0.17
    proximity = 1.0 / (1.0 + 5.0 * curr_dist)

    # 速度因子：越慢越高，k=3 使 speed=1 时约为 0.25
    speed_val = (vx**2 + vy**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed_val)

    # 姿态因子：越正越高，k=3 使 angle=0.3rad 时约为 0.53
    upright_factor = 1.0 / (1.0 + 3.0 * abs(angle))

    # 接触加成：有腿部接触时额外奖励（0.5 到 1.0 之间）
    contact_bonus = 0.5 + 0.5 * (left_contact + right_contact) / 2.0

    # 连续乘积：每个因子 ∈ (0,1]，乘积提供密集梯度
    landing_quality = proximity * speed_factor * upright_factor * contact_bonus

    # === 3. 稳定性惩罚：距离门控，远处不罚 ===
    # 只在目标附近（<2 单位）施加弱稳定性约束
    gate = max(0.0, 1.0 - curr_dist / 2.0)
    stability_penalty = -gate * 0.02 * (abs(vx) + abs(vy) + abs(angular_vel))

    # === 组合 ===
    # progress: 方向性梯度，均值接近 0
    # landing_quality: 密集正信号，引导精细着陆行为
    # stability_penalty: 弱背景约束，仅在近端生效
    total_reward = progress + 1.0 * landing_quality + stability_penalty

    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=234.795902, len=503.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_quality | 0.485144 | 0.485144 | 1.000000 | 0.485144 |
| progress | 0.001971 | 0.002247 | 0.998118 | 0.001971 |
| stability_penalty | -0.003104 | 0.003104 | 0.999136 | -0.003104 |
| total_reward | 0.484011 | 0.484257 | 1.000000 | 0.484011 |
| generated_reward | 0.484011 | 0.484257 | 1.000000 | 0.484011 |
| original_env_reward | 0.033171 | 1.141502 | 1.000000 | 0.033171 |

## Distribution
- score: mean=234.795902, min=168.545600, max=284.504977
- episode_length: mean=503.600000
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
| 5 | progress + soft_landing_proxy + stability_penalty | 144.59 | 187.93 | -43.34 | 1000.00 | progress=0.003 soft_landing_proxy=0.244 stability_penalty=-0.001 | unsolved_stagnation_fresh_restart |
| 6 | dist_reward + landing_proxy + stability_penalty | -113.31 | -113.31 | 0.00 | 71.90 | dist_reward=-0.972 landing_proxy=0.002 stability_penalty=-0.145 | new_best |
| 7 | landing_quality + progress + stability_penalty | 234.80 | 234.80 | 0.00 | 503.60 | landing_quality=0.485 progress=0.002 stability_penalty=-0.003 | target_solved_new_best |
