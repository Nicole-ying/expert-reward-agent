# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 前进速度奖励（正值表示向前，直接作为主学习信号）
    forward_speed = next_obs[2]
    forward_reward = forward_speed  # 系数 1.0

    # 稳定性惩罚：惩罚大倾角、高角速度和明显垂直速度（弹跳）
    tilt_angle = next_obs[0]
    angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    tilt_penalty = -0.5 * (tilt_angle ** 2)
    angular_vel_penalty = -0.1 * (angular_vel ** 2)
    vertical_vel_penalty = -0.5 * (vertical_vel ** 2)
    stability_penalty = tilt_penalty + angular_vel_penalty + vertical_vel_penalty

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 组件角色

1. **forward_reward (主学习信号)**  
   - 直接使用下一步水平速度 `next_obs[2]`（正值前进、负值后退），稠密、每步有梯度。  
   - 提供主要的前进驱动力，与任务目标“尽可能远、尽可能快”吻合。

2. **stability_penalty (稳定约束)**  
   - 由三部分组成：躯干倾角惩罚 (`tilt_angle^2`)、躯干角速度惩罚 (`angular_vel^2`)、垂直速度惩罚 (`vertical_vel^2`)。  
   - 轻量抑制身体剧烈晃动和弹跳，避免 fast‑then‑fail，不直接压制前进动力。  
   - 系数较小，不影响探索初期的大幅度动作。

## 为何没有使用 terminal_success_reward / terminal_failure_penalty

- `info` 字典为空，**没有显式成功/失败标志**。  
- 无法在 `compute_reward` 内部可靠区分 `reached_end_of_terrain` 或 `body_fallen_over`，因此不能加入终点奖励或失败惩罚。

## 未纳入 v1 的组件（留待后续迭代）

- **energy_penalty / action_smoothness_penalty**：效率与平滑代价，v1 阶段应先学会稳定前进，后续再引入。  
- **alive_bonus**：可能会诱导原地站立（hover），当前前进速度奖励已足够提供存活激励。  
- **步态相位奖励（交替接地）**：利用 `leg1_contact`/`leg2_contact` 可以形成更自然的步态，v1 保留信号，未来迭代中作为辅助锚点加入。

## 预期 failure mode 观察点

- **fast_then_fail**：前进奖励过大而稳定性惩罚不足时，机器人可能快速冲出几步后摔倒。如果发生，应增大 `stability_penalty` 各系数。  
- **over_conservative_policy**：如果稳定性惩罚过强，机器人可能畏缩不敢前进（速度接近零）。此时应适当降低惩罚系数或引入存活激励进行平衡。  
- **原地弹跳/静止**：若出现小幅垂直跳动但仍保持倾斜度低，垂直惩罚可能不够敏感，需调整垂直速度系数。