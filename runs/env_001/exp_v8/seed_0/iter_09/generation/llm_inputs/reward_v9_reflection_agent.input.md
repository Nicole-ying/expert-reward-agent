# 上一轮奖励函数代码（该轮得分: 245.281525）
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
score=245.281525, len=374.900000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| center_bonus | 0.114894 | 0.114894 | 0.732319 | 4.152320 |
| progress_reward | 0.027670 | 0.029791 | 0.999319 | 1.000000 |
| soft_landing_proxy | 0.391318 | 0.391318 | 0.520510 | 14.142416 |
| total_reward | 0.533881 | 0.534920 | 0.999979 | 19.294735 |
| generated_reward | 0.533881 | 0.534920 | 0.999979 | 19.294735 |
| original_env_reward | -0.076945 | 1.606194 | 1.000000 | -2.780842 |
| original_env_reward | -0.076945 | 1.606194 | 1.000000 | -2.780842 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=245.281525, min=105.171554, max=294.904996
- episode_length: mean=374.900000
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
