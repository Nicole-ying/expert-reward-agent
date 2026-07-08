# 上一轮奖励函数代码（该轮得分: 500.000000）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v2。
    
    改动 (v1→v2)：将二次惩罚改为绝对值惩罚。
    - 二次惩罚在接近零时梯度消失，绝对值提供恒定梯度，便于精细平衡。
    - 系数重新标定以在终止阈值附近保持与 v1 相同的惩罚量级。
    """
    # 提取下一步观察
    base_pos = next_obs[0]            # 底座水平位置
    pole_angle = next_obs[2]          # 杆偏角 (rad)
    pole_angular_vel = next_obs[3]    # 杆角速度

    # 惩罚系数（已针对绝对值形态重新标定）
    w_angle = 1.0
    w_angular_vel = 0.5
    w_pos = 0.25

    # 核心学习信号：偏离直立和中心的绝对值惩罚
    progress_reward = -(
        w_angle * abs(pole_angle) +
        w_angular_vel * abs(pole_angular_vel) +
        w_pos * abs(base_pos)
    )

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v1。
    使用连续负惩罚引导摆杆保持竖直并使底座接近轨道中心。
    """
    # 提取下一步观察
    base_pos = next_obs[0]       # 底座水平位置
    pole_angle = next_obs[2]     # 杆偏角 (rad)
    pole_angular_vel = next_obs[3]  # 杆角速度

    # 惩罚系数
    w_angle = 5.0
    w_angular_vel = 0.5
    w_pos = 0.1

    # 核心学习信号：偏离直立和中心的二次惩罚
    progress_reward = -(
        w_angle * (pole_angle ** 2) +
        w_angular_vel * (pole_angular_vel ** 2) +
        w_pos * (base_pos ** 2)
    )

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

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
| progress_reward | -0.189149 | 0.189149 | 1.000000 | -1.000000 |
| total_reward | -0.189149 | 0.189149 | 1.000000 | -1.000000 |
| generated_reward | -0.189149 | 0.189149 | 1.000000 | -1.000000 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 5.286841 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 5.286841 |

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
| 1 | progress_reward | 500.00 | 500.00 | 0.00 | 500.00 | progress_reward=-0.059 | target_solved_new_best |
| 2 | progress_reward | 500.00 | 500.00 | 0.00 | 500.00 | progress_reward=-0.189 | target_solved_no_improvement |
