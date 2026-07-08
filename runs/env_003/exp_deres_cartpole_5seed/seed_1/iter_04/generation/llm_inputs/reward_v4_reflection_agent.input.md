# 上一轮奖励函数代码（该轮得分: 500.000000）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward — v3.

    Based on iter 2 (score=500, solved).  One change: added a small per-step
    survival_bonus so that the total reward stays slightly positive even when
    the agent is perfectly balanced (previously mean ~ -0.0007).

    Signals:
      - progress_reward: encourages moving towards upright + center.
      - stability_penalty: light damping on velocities to suppress jitter.
      - survival_bonus: constant small positive per step.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    # kept low from iter 2 (0.001) — ratio_to_progress ~ -0.32, healthy
    w_vel = 0.001
    w_angvel = 0.001

    # new in v3: tiny constant to keep total reward positive at steady state
    survival_bonus = 0.005

    # -- cost helper --
    def cost(o):
        return w_pos * abs(o[0]) + w_angle * abs(o[2])

    # -- main progress reward (potential-based shaping) --
    progress_reward = scale_progress * (cost(obs) - cost(next_obs))

    # -- light stability penalty --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    # -- total --
    total_reward = progress_reward + stability_penalty + survival_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward.
    Uses:
    - progress_reward: dense improvement towards stable upright and center.
    - stability_penalty: light penalty on linear/angular speeds.
    
    Iter 2 change: reduced stability_penalty weights 10x (0.01→0.001)
    because penalty ratio_to_progress was -3.16, drowning out the progress signal.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    # Reduced 10x from 0.01 → 0.001 to fix penalty dominance (ratio was -3.16)
    w_vel = 0.001
    w_angvel = 0.001

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
score=500.000000, len=500.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | -0.000312 | 0.002519 | 1.000000 | -0.311570 |
| stability_penalty | -0.000327 | 0.000327 | 1.000000 | -0.327459 |
| survival_bonus | 0.005000 | 0.005000 | 1.000000 | 5.000000 |
| total_reward | 0.004361 | 0.004805 | 1.000000 | 4.360972 |
| generated_reward | 0.004361 | 0.004805 | 1.000000 | 4.360972 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 1000.000000 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 1000.000000 |

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
| 1 | progress_reward + stability_penalty | 466.15 | 466.15 | 0.00 | 466.15 | progress_reward=-0.001 stability_penalty=-0.004 | new_best |
| 2 | progress_reward + stability_penalty | 500.00 | 500.00 | 0.00 | 500.00 | progress_reward=-0.000 stability_penalty=-0.000 | target_solved_new_best |
| 3 | progress_reward + stability_penalty + survival_bonus | 500.00 | 500.00 | 0.00 | 500.00 | progress_reward=-0.000 stability_penalty=-0.000 survival_bonus=0.005 | target_solved_no_improvement |
