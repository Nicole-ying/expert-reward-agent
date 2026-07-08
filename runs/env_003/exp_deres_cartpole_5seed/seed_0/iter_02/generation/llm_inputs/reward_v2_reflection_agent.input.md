# 上一轮奖励函数代码（该轮得分: 500.000000）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Survival balance task (Env_003).
    Use potential-based shaping as the main progress signal.
    """
    # Constants
    GAMMA = 0.99
    ANGLE_LIMIT = 0.20943951   # radians, pole falls if exceeded
    POS_LIMIT = 2.4            # base goes out of bounds if exceeded

    # Current and next states
    pos_now, _, angle_now, _ = obs[0], obs[1], obs[2], obs[3]
    pos_next, _, angle_next, _ = next_obs[0], next_obs[1], next_obs[2], next_obs[3]

    # Potential function: encourages pole upright and base centered
    # Phi = - ( (angle/limit)^2 + (position/limit)^2 )
    phi_now = -((angle_now / ANGLE_LIMIT) ** 2 + (pos_now / POS_LIMIT) ** 2)
    phi_next = -((angle_next / ANGLE_LIMIT) ** 2 + (pos_next / POS_LIMIT) ** 2)

    # Shaping reward
    progress_reward = GAMMA * phi_next - phi_now

    total_reward = progress_reward
    components = {
        "progress_reward": progress_reward
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
| progress_reward | -0.001612 | 0.003351 | 1.000000 | -1.000000 |
| total_reward | -0.001612 | 0.003351 | 1.000000 | -1.000000 |
| generated_reward | -0.001612 | 0.003351 | 1.000000 | -1.000000 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 620.229801 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 620.229801 |

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
| 1 | progress_reward | 500.00 | 500.00 | 0.00 | 500.00 | progress_reward=-0.002 | target_solved_new_best |
