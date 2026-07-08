# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D bipedal locomotion on rough terrain.
    Drives forward progress while maintaining stable posture.
    No explicit success/failure flags are available in info.
    """
    # ---------- forward progress component ----------
    # Horizontal velocity in the forward direction (next_obs[2]).
    # Positive values correspond to moving forward, negative to backward.
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # ---------- stability penalty component ----------
    # Penalize large hull tilt (deviation from upright) and vertical velocity (jumping/falling).
    # Using absolute values gives smooth, dense gradients.
    hull_angle = next_obs[0]          # tilt angle
    vertical_velocity = next_obs[3]   # vertical speed

    w_angle = 0.5
    w_vertical = 0.1
    stability_penalty = -w_angle * abs(hull_angle) - w_vertical * abs(vertical_velocity)

    # ---------- total reward ----------
    total_reward = forward_reward + stability_penalty

    # components dict: only the terms that are directly summed into total_reward
    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色

1. **forward_reward**（主学习信号）
   - 形态：`w_fwd * horizontal_velocity`
   - 角色：驱动机器人持续向前移动。每一步都能获得梯度（连续的速度值），与任务目标“走得远、走得快”直接对应。
   - 选择理由：环境的观测明确提供前进线速度，无需间接推导；该信号每个 step 都提供有意义的反馈，是最核心的前进驱动力。

2. **stability_penalty**（稳定/安全约束）
   - 形态：`-w_angle * |hull_angle| - w_vertical * |vertical_velocity|`
   - 角色：抑制机器人过度倾斜和上下跳跃，促使它形成平稳的步态，降低摔倒风险。
   - 选择理由：躯干角度直接反映姿态，垂直速度能捕获跳跃或坠落的早期信号。两者都是稠密、连续的信号，可以平滑地引导稳定性。权重较小（0.5 和 0.1），不会压制前向探索。

## 为何没有使用 terminal_success_reward / terminal_failure_penalty

- 环境信息卡明确声明 `explicit_success_flag_available: false` 且 `explicit_failure_flag_available: false`。
- `info` 字典为空，无法获取任何成功或失败的标志。
- 因此必须避免直接奖励/惩罚终止事件，这类信号在 v1 不可用。

## 未引入的组件及其理由（留到后续迭代）

- **energy_penalty（动作扭矩惩罚）**：v1 阶段先让机器人学会走路，过早引入能耗约束可能导致 agent 不敢做出足够力量的动作，造成“不敢动”的问题。
- **alive_bonus（存活奖励）**：我们没有可靠的 `done` 信号传入奖励函数，无法按步区分存活与终止，因此暂不加。前向速度奖励与稳定性惩罚已经能鼓励存活（摔倒会导致较大的倾斜/垂直速度惩罚，且后续步无法获得前进奖励）。
- **action_smoothness_penalty**：需要前一时刻的动作信息，当前函数签名不提供历史动作，因此无法可靠计算。
- **gated_reward / dynamic_curriculum**：属于复杂门控和课程学习，适合后续性能优化阶段再引入。

## 训练后应观察的 failure mode

- **fast_then_fail**：如果稳定性惩罚过弱，机器人可能先学会大步幅快速前进，但缺乏足够姿态控制，最终摔倒。应检查是否出现短期高奖励后突然终止。
- **hover_or_stand_still**：如果稳定性惩罚过强或主信号不够，agent 可能学会小幅挪动甚至静止来保持低惩罚。没有存活奖励在一定程度上降低了站立收益，但仍需监控前向速度是否持续上升。
- **oscillating around target**：由于没有终点目标，不存在这种情况。主要关注是否出现高频小幅摆动、但前进效率低的步态。
- **过保守策略**：如果机器人动作幅度极小，速度缓慢但保持稳定，说明主信号不足以驱动实际前进，可能需要后续调整权重或加入速度门控。