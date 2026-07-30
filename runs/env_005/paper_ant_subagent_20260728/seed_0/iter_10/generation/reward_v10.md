# 设计理由

过去9轮迭代中，唯一的高分点出现在 iter 3（score 1839.71，len 981.5），其骨架为 `gated_forward（前向受高度门控）+ upright_bonus（直立奖励）+ lateral_penalty + action_penalty`。此后所有改动都试图通过“更严格的门控乘法（前向 × 高度门 × 直立门）”或“增加独立惩罚（角速度、更大的侧向惩罚）”来提升稳定性，结果全部失败，长剧集得分从未恢复。根本原因是：

- **门控乘法坍塌梯度**：当 `forward_reward = v_x * height_gate * upright_gate` 时，只要一个因子偏低，整个奖励就大幅衰减，agent 在早期难以获得足够梯度来同时学习保持高度、直立和前进。而 iter 3 的成功表明，**把直立目标从门控中解放出来，做成独立的连续奖励（upright_bonus）**，允许 agent 在姿态改善的过程中仍能获得前进奖励，这是收敛的关键。
- **过多惩罚压制探索**：iter 4–7 添加了角速度惩罚、加大侧向惩罚，导致 agent 产生“不敢动”的保守策略，score 暴跌至负值。
- **当前策略（iter 9）已陷入慢速安全区**：均长 872 步、terminated 仅 15%，但 forward 和 upright 被捆绑，最终 shaped reward per-step 仅 0.576，几乎被原始环境负信号（‑0.564）抵消，无力推进。

因此，本次重建**回归 iter 3 的成功架构，但做关键简化和尺度优化**：
- 选用 **soft_health_gate（2.6）** 只对高度做门控，乘到前向速度上，避免上界过度限制；
- 选用 **dense_state_signal（2.1）** 的连续正向奖励作为 upright_bonus，不再使用 upright_gate 作为乘法门；
- 保留 **quadratic_penalty（2.5）** 对侧向速度和动作幅度，权重严格控制；
- 删除一切无用的乘法耦合与已退役的惩罚组件。

与已尝试过的所有骨架的本质不同：**前向与直立在奖励函数中解耦，通过加法合并**，而非乘法互锁。这样两个学习目标可独立贡献梯度，避免乘积塌缩，同时高度安全依然由门控保证。

# 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]
    v_y      = obs[14]

    # ---- upright measure (1.0 when perfectly upright) ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ---- height gate: 1.0 in safe zone, decays to 0 near hard limits ----
    z_low  = 0.25   # termination boundary: z <= 0.2
    z_high = 0.95   # termination boundary: z >= 1.0
    z_safe_low  = 0.35
    z_safe_high = 0.85

    low_factor = (body_z - z_low) / (z_safe_low - z_low)
    low_factor = max(0.0, min(1.0, low_factor))

    high_factor = (z_high - body_z) / (z_high - z_safe_high)
    high_factor = max(0.0, min(1.0, high_factor))

    height_gate = low_factor * high_factor   # range [0, 1]

    # ---- forward progress (main signal), gated only by height ----
    w_fwd = 1.0
    forward_reward = w_fwd * v_x * height_gate

    # ---- upright bonus: independent continuous reward for staying upright ----
    w_up = 0.2
    upright_bonus = w_up * max(0.0, up_z)

    # ---- lateral stability (quadratic penalty) ----
    w_lat = 0.3
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action smoothness (light penalty) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a ** 2 for a in action) / len(action)

    # ---- total reward (additive composition) ----
    total_reward = forward_reward + upright_bonus + lateral_penalty + action_penalty

    components = {
        "forward_reward":  forward_reward,
        "upright_bonus":   upright_bonus,
        "lateral_penalty": lateral_penalty,
        "action_penalty":  action_penalty,
        "_height_gate":    height_gate   # for logging, not a reward term
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 缺少独立直立奖励，且前向被双门控（高度×直立）过度压制；高度门控已有但缺少直立正向分离信号。
- **behavior**: agent 以保守姿态慢速前进，满足于低 reward 的安全区域，无法突破到高分。
- **signal**: 缺独立的 upright_bonus；前向与直立耦合太紧，缺解耦加法。
- **level**: Level 2（但在重建模式下执行结构变换，属实际 Level 3 骨架更换）
- **hypothesis**: 恢复 iter 3 的直立奖励独立成分，解除乘法门控对前向的压制，将同时提升前向速度和姿态质量，score 应大幅回升。
- **risk**: 如果 upright_bonus 权重偏高，agent 可能偏向“站立不动”获得直立奖励而忽略前进；但 w_up=0.2 仅为前向最大梯度的约 1/5，风险可控。侧向惩罚权重可能仍需后续微调。