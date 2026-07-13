# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    # 使用 next_obs[2] 作为水平前进速度，鼓励 agent 向前移动
    forward_velocity = next_obs[2]
    # 限制速度范围，避免过大速度导致不稳定
    forward_velocity_clipped = max(-2.0, min(2.0, forward_velocity))
    # 只奖励正向速度（前进），负向速度（后退）不奖励
    forward_reward = 2.0 * max(0.0, forward_velocity_clipped)
    
    # 存活奖励：每步给予小额奖励，鼓励 agent 保持不倒
    # 通过检查 next_obs[0]（主体角度）是否在合理范围内来判断是否存活
    # 角度绝对值小于 1.0 弧度（约 57 度）视为存活
    hull_angle = next_obs[0]
    is_alive = abs(hull_angle) < 1.0
    alive_bonus = 0.5 if is_alive else 0.0
    
    # 稳定性惩罚：轻量约束，抑制过大角度和角速度
    # 使用 next_obs[0]（角度）和 next_obs[1]（角速度）
    angle_penalty = -0.1 * (hull_angle ** 2)
    angular_vel_penalty = -0.05 * (next_obs[1] ** 2)
    stability_penalty = angle_penalty + angular_vel_penalty
    
    # 总奖励
    total_reward = forward_reward + alive_bonus + stability_penalty
    
    # 组件字典
    components = {
        "progress_reward": forward_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward（主学习信号）**：基于 `next_obs[2]`（水平前进速度）的线性奖励。这是任务的核心驱动力，告诉 agent "向前走就能得分"。速度被裁剪到 [-2, 2] 范围并只奖励正向速度，避免 agent 通过后退或高速震荡来 hack 奖励。

2. **alive_bonus（存活激励）**：每步给予 0.5 的小额奖励，条件是 `next_obs[0]`（主体角度）绝对值小于 1.0 弧度。这鼓励 agent 保持直立不倒，为学习稳定步态提供基础信号。

3. **stability_penalty（稳定约束）**：轻量惩罚项，对主体角度和角速度的平方进行惩罚。权重较小（-0.1 和 -0.05），主要作用是抑制过大角度和角速度，防止 agent 在快速前进时过度倾斜或旋转。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available = false`：info 始终为空字典，没有显式成功标志，无法可靠判断是否到达终点。
- `explicit_failure_flag_available = false`：info 始终为空字典，没有显式失败标志，无法区分摔倒和其他终止原因。
- 因此，这两个终端信号在 v1 阶段不可用，留待后续迭代当环境提供更丰富的终止信息时再加入。

## 留到后续迭代的组件

- **energy_penalty**：动作能耗惩罚。v1 阶段 agent 应先学会稳定行走，过早加入能耗惩罚可能导致 agent 不敢动作。
- **action_smoothness_penalty**：动作平滑约束。如果训练中出现动作抖动，后续迭代再加入。
- **time_penalty**：效率约束。v1 阶段先关注完成任务，效率优化后续迭代。
- **gated_reward**：复杂门控。v1 阶段保持简单，后续若出现安全与进度冲突再考虑。
- **soft_landing_proxy**：任务完成近似信号。当前没有显式位置信息，且 agent 应先学会基本行走，后续若出现 high_reward_without_success 再考虑。

## 训练后应该观察的 failure mode

1. **fast_then_fail**：agent 可能快速前进但频繁摔倒。如果观察到这种情况，应增大 stability_penalty 权重或加入更严格的角度约束。
2. **hover_or_stand_still**：agent 可能原地不动获取存活奖励而不前进。如果观察到这种情况，应减小 alive_bonus 权重或增大 forward_reward 权重。
3. **over_conservative_policy**：agent 可能过于保守，前进速度很慢但非常稳定。如果观察到这种情况，应调整 forward_reward 和 stability_penalty 的平衡，鼓励更积极的步态。
4. **动作抖动**：agent 可能产生高频振荡动作。如果观察到这种情况，后续迭代应加入 action_smoothness_penalty。