# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: runs\env_001\exp_v8\seed_0\iter_08\generation\validations\reward_v8.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 137.606613）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标（scale=8，符合 skeleton 推荐 5~20）
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 轻量稳定约束
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))
    angular_v = abs(float(next_obs[5]))

    raw_stability_penalty = (
        -0.005 * speed
        - 0.01 * angle
        - 0.005 * angular_v
    )

    # stage weighting: 早期忽略 stability，后期逐渐加入
    # t=0 → late_weight=0（无惩罚）；t=1 → late_weight=1（全惩罚）
    t = training_progress
    late_weight = max(0.0, min(1.0, 2.0 * t - 0.5))
    stability_penalty = raw_stability_penalty * late_weight

    # 3. 软着陆近似信号：连续乘积因子 + x/(1+x) 饱和
    #    相比裸乘积（ratio 6.9x），饱和形式将输出压缩到 [0, 0.5]
    #    保留低值梯度（导数在 0 处≈1），压制高值 exploit 空间
    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    raw_proxy = proximity_factor * speed_factor * angle_factor * contact_factor
    # x/(1+x) 饱和：raw_proxy ∈ [0,1] → soft_landing_proxy ∈ [0, 0.5]
    soft_landing_proxy = raw_proxy / (1.0 + raw_proxy)

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
score=137.606613, len=1000.000000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.034361 | 0.036802 | 0.999804 | 1.000000 |
| soft_landing_proxy | 0.210295 | 0.210295 | 0.481814 | 6.120186 |
| stability_penalty | -0.000000 | 0.000000 | 0.003508 | -0.000000 |
| total_reward | 0.244655 | 0.246697 | 0.999993 | 7.120186 |
| generated_reward | 0.244655 | 0.246697 | 0.999993 | 7.120186 |
| original_env_reward | -0.247575 | 1.769447 | 1.000000 | -7.205159 |
| original_env_reward | -0.247575 | 1.769447 | 1.000000 | -7.205159 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=137.606613, min=107.956019, max=171.250968
- episode_length: mean=1000.000000
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
