# 上一轮奖励函数代码（该轮得分: 466.150000）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward.
    Uses:
    - progress_reward: dense improvement towards stable upright and center.
    - stability_penalty: light penalty on linear/angular speeds.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    w_vel = 0.01
    w_angvel = 0.01

    # -- cost helper --
    def cost(o):
        return w_pos * abs(o[0]) + w_angle * abs(o[2])

    # -- main progress reward --
    progress_reward = scale_progress * (cost(obs) - cost(next_obs))

    # -- light stability penalty (only on next step) --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=466.150000, len=466.150000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | -0.001244 | 0.003207 | 1.000000 | -1.000000 |
| stability_penalty | -0.003930 | 0.003930 | 1.000000 | -3.159049 |
| total_reward | -0.005174 | 0.006307 | 1.000000 | -4.159049 |
| generated_reward | -0.005174 | 0.006307 | 1.000000 | -4.159049 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 803.846372 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 803.846372 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=466.150000, min=156.000000, max=500.000000
- episode_length: mean=466.150000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + stability_penalty | 466.15 | 466.15 | 0.00 | 466.15 | progress_reward=-0.001 stability_penalty=-0.004 | new_best |
