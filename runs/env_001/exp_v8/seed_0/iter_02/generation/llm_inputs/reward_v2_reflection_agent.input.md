# 上一轮奖励函数代码（该轮得分: -88.785487）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置为 (0,0)，平台中心
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]

    # 当前位置与目标距离
    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    # 下一时刻与目标距离
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标
    progress_reward = d_curr - d_next

    # 2. 轻量稳定约束，基于下一状态
    vx, vy = next_obs[2], next_obs[3]
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(next_obs[4])
    angular_v = abs(next_obs[5])

    stability_penalty = (
        -0.05 * speed
        - 0.1 * angle
        - 0.05 * angular_v
    )

    # 3. 软着陆完成近似信号
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
score=-88.785487, len=69.000000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.015989 | 0.016941 | 0.999990 | 1.000000 |
| soft_landing_proxy | 0.003192 | 0.003192 | 0.003192 | 0.199627 |
| stability_penalty | -0.058028 | 0.058028 | 1.000000 | -3.629324 |
| total_reward | -0.038848 | 0.045166 | 1.000000 | -2.429697 |
| generated_reward | -0.038848 | 0.045166 | 1.000000 | -2.429697 |
| original_env_reward | -1.456194 | 2.453977 | 1.000000 | -91.076577 |
| original_env_reward | -1.456194 | 2.453977 | 1.000000 | -91.076577 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=-88.785487, min=-119.653678, max=-63.373109
- episode_length: mean=69.000000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -88.79 | 0.00 | -88.79 | 69.00 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.058 | new_best |
