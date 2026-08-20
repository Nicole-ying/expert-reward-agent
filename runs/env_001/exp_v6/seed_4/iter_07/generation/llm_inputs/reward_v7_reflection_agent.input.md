# ⚠️ 上一版代码验证失败
错误信息：Reward v7 failed validation: runs\env_001\exp_v6\seed_4\iter_07\generation\validations\reward_v7.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 263.607902）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 诊断与修改理由 ===
    # 问题：上轮 shaping_reward 均值仅 0.014/step，信号强度不到环境惩罚 (-0.91/step) 的 2%。
    #       agent 主要感受环境惩罚，无法学到有效策略 → 80% crash。
    # 修改（层次2）：采用推荐 Φ=-(distance + speed + |angle|) 组合势能，
    #       单次势能差同时反映"靠近 + 减速 + 扶正"三项进度，信号更丰富。
    #       scale=20 将信号放大到与 env reward 可比量级。
    #       移除独立 stability_penalty——稳定目标已融入势能差，减少组件数量。

    # 计算当前与下一时刻的势能要素
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    speed_current = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5

    angle_current = abs(obs[4])
    angle_next = abs(next_obs[4])

    # 组合势能：Φ = -(distance + speed + |angle|)
    phi_current = -(dist_current + speed_current + angle_current)
    phi_next = -(dist_next + speed_next + angle_next)

    # γ=1.0 势能塑形：F = Φ(next) - Φ(current) = 各项减少量之和
    F = phi_next - phi_current
    scale = 20.0
    shaping_reward = scale * F

    total_reward = shaping_reward

    components = {
        "shaping_reward": shaping_reward,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=263.607902, len=350.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| shaping_reward | 0.047212 | 0.229841 | 1.000000 | 0.047212 |
| total_reward | 0.046740 | 0.229369 | 1.000000 | 0.046740 |
| generated_reward | 0.046740 | 0.229369 | 1.000000 | 0.046740 |
| original_env_reward | 0.071031 | 1.439352 | 1.000000 | 0.071031 |

## Distribution
- score: mean=263.607902, min=232.619849, max=292.890648
- episode_length: mean=350.100000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -111.44 | -111.44 | 0.00 | 71.30 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.057 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | -111.12 | -111.44 | 0.33 | 71.40 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.006 | no_meaningful_improvement |
| 4 | potential_shaping_reward + stability_penalty | -111.89 | -111.89 | 0.00 | 71.30 | potential_shaping_reward=0.026 stability_penalty=-0.068 | new_best |
| 5 | shaping_reward + stability_penalty | -81.52 | -81.52 | 0.00 | 98.10 | shaping_reward=0.014 stability_penalty=-0.006 | new_best |
| 6 | shaping_reward | 263.61 | 263.61 | 0.00 | 350.10 | shaping_reward=0.047 | target_solved_new_best |
