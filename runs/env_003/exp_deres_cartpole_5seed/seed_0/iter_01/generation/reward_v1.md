# reward_v1.py

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

# reward_v1 设计说明

## 使用了哪些奖励组件
- **`progress_reward`**：基于势能塑形的密集学习信号。定义为 `γ * Φ(s′) − Φ(s)`，其中势能函数 `Φ(s) = -[(θ/θ_limit)² + (x/x_limit)²]`。θ 是杆子角度，x 是底座位置。该信号指导智能体同时保持杆子直立和底座居中，值越接近 0 表示越稳定。塑形奖励不改变最优策略，且能为每一步提供有意义的梯度。

## 每个组件的角色
- `progress_reward` 是唯一组件，承担主学习信号的角色。它直接表达“存活并保持稳定”的目标，通过连续函数避免离散条件的可利用风险，并且每步都有可微梯度。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境明确声明 `explicit_success_flag_available = false` 且 `explicit_failure_flag_available = false`。`info` 始终为空字典，没有可用的成功或失败标志。依据设计规范，不得使用这些组件。

## 哪些组件留到后续迭代
- **stability_penalty**：如果需要显式抑制高速或高角速度（而不仅仅是位置/角度偏差），可在后续轻量加入。
- **survival_bonus**（每步正奖励）或 **time_penalty**（效率约束）：目前 v1 专注于学会活下来，不引入额外效率信号，以免干扰早期探索。
- **terminal_failure_penalty** 或 **terminal_success_reward**：待环境提供明确的终止原因字段后，可引入更强的终端信号，加速最终策略收敛。
- **gated_reward**、**energy_penalty** 等复杂组件均留待 v2+ 优化。

## 训练后应观察的 failure mode
- **边界附近震荡**：agent 可能为了最大化 `progress_reward` 而在角度或位置边界附近小幅调整，但始终不触发终止。应当监控 episode 平均存活步数和势能值，检查高频小幅调整是否导致低效但存活的行为。
- **过度依赖 shaping 导致慢收敛**：势力塑形奖励幅度较小，可能使价值函数学习变慢，需观察早期 reward 曲线是否平缓。可考虑后续加入轻量事件奖励或增大势能尺度。
- **终止时不完整 final_step 奖励估计**：部分框架在终止步可能不给 `next_obs` 或使用截断后的观测，可能导致势能差值不准确。需确认环境是否提供了合法的最终 `next_obs`。如果出现异常 spike，可能需要用 `done` 标志进行特殊处理（当前版本依赖框架提供规范的 `next_obs`）。