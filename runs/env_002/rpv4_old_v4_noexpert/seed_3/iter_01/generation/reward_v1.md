环境事实 → 任务画像：连续力矩驱动的平面双足行走，主目标是正向前进，核心约束是避免摔倒。`info` 为空，没有显式 success/failure 标志，因此不使用 terminal reward。  
奖励职责选择：`forward_progress` 和 `balance_and_fall_prevention`。  
职责-信号映射：  
- `forward_progress`：`next_obs[2]`，前向水平速度  
- `balance_and_fall_prevention`：`next_obs[0]` 躯干倾角、`next_obs[1]` 躯干角速度、`next_obs[3]` 垂直速度  
- 排除 `energy_efficiency`、`gait_symmetry/contact_pattern`、terminal_success/failure：v1 先学习稳定前进，不加入动作代价，避免 agent 不敢迈步。

公式算子：前向速度使用线性奖励；平衡约束使用二次惩罚，保留连续梯度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v1 reward: 前向速度驱动 + 躯干/身体稳定性正则。
    # info 为空，不使用 info。
    # explicit success/failure flags 均不可用，因此不包含 terminal reward。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 1.0 * next_obs[2]

    # 稳定性/防摔倒约束：惩罚躯干倾斜、剧烈角速度和过度垂直运动。
    tilt_penalty = 1.0 * next_obs[0] ** 2
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    balance_penalty = (
        tilt_penalty
        + angular_velocity_penalty
        + vertical_velocity_penalty
    )

    total_reward = forward_reward - balance_penalty

    components = {
        "forward_reward": forward_reward,
        "balance_penalty": -balance_penalty,
    }

    return float(total_reward), components
```