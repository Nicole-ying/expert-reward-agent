# 上一轮奖励函数代码（该轮得分: 500.000000）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for Env_003 (CartPole-like survival balance task).

    - progress_reward: baseline - (w_angle * angle^2 + w_pos * pos^2)
      Encourages keeping the pole near upright and the cart near the center.
    - stability_penalty: small penalty on cart velocity and pole angular velocity
      to suppress unnecessary oscillations.
    """
    # extracted constants for clarity
    w_angle = 5.0
    w_pos = 0.1
    baseline = 1.0

    lambda_vel = 0.01
    lambda_angvel = 0.01

    # next state
    pos = next_obs[0]
    vel = next_obs[1]
    angle = next_obs[2]
    angvel = next_obs[3]

    # main learning signal
    progress_reward = baseline - (w_angle * (angle ** 2) + w_pos * (pos ** 2))

    # stability penalty (light, avoid dominating)
    stability_penalty = -lambda_vel * abs(vel) - lambda_angvel * abs(angvel)

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=500.000000, len=500.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.981300 | 0.981300 | 1.000000 | 1.000000 |
| stability_penalty | -0.003662 | 0.003662 | 1.000000 | -0.003732 |
| total_reward | 0.977637 | 0.977637 | 1.000000 | 0.996268 |
| generated_reward | 0.977637 | 0.977637 | 1.000000 | 0.996268 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 1.019056 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 1.019056 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=500.000000, min=500.000000, max=500.000000
- episode_length: mean=500.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + stability_penalty | 500.00 | 500.00 | 0.00 | 500.00 | progress_reward=0.981 stability_penalty=-0.004 | target_solved_new_best |
