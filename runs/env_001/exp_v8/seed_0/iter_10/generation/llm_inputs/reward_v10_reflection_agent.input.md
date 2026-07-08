# ⚠️ 上一版代码验证失败
错误信息：Reward v10 failed validation: runs\env_001\exp_v8\seed_0\iter_10\generation\validations\reward_v10.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 240.905342）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：scale 从 8.0 → 15.0，强化进度引导
    #    骨架推荐 5~20，当前 8.0 下 progress 仅占总 reward ~5%，偏弱
    progress_reward = (d_curr - d_next) * 15.0

    # 2. 软着陆近似信号：连续乘积（保持 best 路线不变）
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))

    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    soft_landing_proxy = (
        proximity_factor * speed_factor * angle_factor * contact_factor
    )

    # 3. 中心邻近奖励：保持上次改动（radius=0.5, max=0.2），不改
    center_bonus = 0.2 * max(0.0, 1.0 - d_next / 0.5)

    total_reward = progress_reward + soft_landing_proxy + center_bonus

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "center_bonus": center_bonus,
    }
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标（scale=8，骨架推荐 5~20）
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 软着陆近似信号：连续乘积（无饱和，回到 best 路线）
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))

    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    soft_landing_proxy = (
        proximity_factor * speed_factor * angle_factor * contact_factor
    )

    # 3. 中心邻近奖励：替代已死的 stability_penalty
    #    半径 0.5 内给予正向信号（最大 0.2），引导 agent 留在目标附近
    #    比 penalty 更友好——鼓励靠近而非惩罚探索
    center_bonus = 0.2 * max(0.0, 1.0 - d_next / 0.5)

    total_reward = progress_reward + soft_landing_proxy + center_bonus

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "center_bonus": center_bonus,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=240.905342, len=415.150000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| center_bonus | 0.114038 | 0.114038 | 0.740887 | 2.297248 |
| progress_reward | 0.049641 | 0.054063 | 0.998966 | 1.000000 |
| soft_landing_proxy | 0.372611 | 0.372611 | 0.519768 | 7.506118 |
| total_reward | 0.536290 | 0.538518 | 0.999998 | 10.803365 |
| generated_reward | 0.536290 | 0.538518 | 0.999998 | 10.803365 |
| original_env_reward | -0.108050 | 1.652333 | 1.000000 | -2.176627 |
| original_env_reward | -0.108050 | 1.652333 | 1.000000 | -2.176627 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=240.905342, min=202.336528, max=274.576486
- episode_length: mean=415.150000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -88.79 | 0.00 | -88.79 | 69.00 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.058 | new_best |
| 2 | progress_reward + soft_landing_proxy + stability_penalty | -108.77 | -88.79 | -19.98 | 68.45 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.006 | no_meaningful_improvement |
| 3 | progress_reward + soft_landing_proxy + stability_penalty | 169.76 | 169.76 | 0.00 | 922.35 | progress_reward=0.043 soft_landing_proxy=0.294 stability_penalty=-0.000 | new_best |
| 4 | progress_reward + soft_landing_proxy + stability_penalty | -220.92 | 169.76 | -390.68 | 74.70 | progress_reward=0.136 soft_landing_proxy=0.000 stability_penalty=-0.000 | no_meaningful_improvement |
| 5 | progress_reward + soft_landing_proxy + stability_penalty | 169.76 | 169.76 | 0.00 | 922.35 | progress_reward=0.043 soft_landing_proxy=0.294 stability_penalty=-0.000 | no_meaningful_improvement |
| 6 | progress_reward + soft_landing_proxy + stability_penalty | -251.56 | 169.76 | -421.31 | 73.30 | progress_reward=0.140 soft_landing_proxy=0.000 stability_penalty=-0.000 | unsolved_stagnation_fresh_restart |
| 7 | progress_reward + soft_landing_proxy + stability_penalty | 137.61 | 137.61 | 0.00 | 1000.00 | progress_reward=0.034 soft_landing_proxy=0.210 stability_penalty=-0.000 | new_best |
| 8 | center_bonus + progress_reward + soft_landing_proxy | 245.28 | 245.28 | 0.00 | 374.90 | center_bonus=0.115 progress_reward=0.028 soft_landing_proxy=0.391 | target_solved_new_best |
| 9 | center_bonus + progress_reward + soft_landing_proxy | 240.91 | 245.28 | -4.38 | 415.15 | center_bonus=0.114 progress_reward=0.050 soft_landing_proxy=0.373 | target_solved_no_improvement |
