# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: runs\env_001\exp_v8\seed_0\iter_03\generation\validations\reward_v3.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -108.773725）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，平台中心
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]

    # 当前位置与目标距离
    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    # 下一时刻与目标距离
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标
    progress_reward = d_curr - d_next

    # 2. 轻量稳定约束（系数降低10倍，ratio 从 -3.63 → ~-0.36）
    vx, vy = next_obs[2], next_obs[3]
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(next_obs[4])
    angular_v = abs(next_obs[5])

    stability_penalty = (
        -0.005 * speed    # was -0.05
        - 0.01 * angle    # was -0.1
        - 0.005 * angular_v  # was -0.05
    )

    # 3. 软着陆完成近似信号（保持不变）
    soft_landing_proxy = 0.0
    if (
        d_next < 0.15
        and speed < 0.2
        and angle < 0.1
        and next_obs[6] == 1.0
        and next_obs[7] == 1.0
    ):
        soft_landing_proxy = 1.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-108.773725, len=68.450000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.016043 | 0.016982 | 0.999998 | 1.000000 |
| soft_landing_proxy | 0.003080 | 0.003080 | 0.003080 | 0.191990 |
| stability_penalty | -0.006062 | 0.006062 | 1.000000 | -0.377833 |
| total_reward | 0.013062 | 0.015507 | 1.000000 | 0.814157 |
| generated_reward | 0.013062 | 0.015507 | 1.000000 | 0.814157 |
| original_env_reward | -1.575244 | 2.424714 | 1.000000 | -98.186655 |
| original_env_reward | -1.575244 | 2.424714 | 1.000000 | -98.186655 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=-108.773725, min=-124.224441, max=-91.779191
- episode_length: mean=68.450000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -88.79 | 0.00 | -88.79 | 69.00 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.058 | new_best |
| 2 | progress_reward + soft_landing_proxy + stability_penalty | -108.77 | -88.79 | -19.98 | 68.45 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.006 | no_meaningful_improvement |
