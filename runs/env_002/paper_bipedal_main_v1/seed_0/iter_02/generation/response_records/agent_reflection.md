# Response Record

## 诊断输出

1. **evidence**：首轮实验，20/20 terminated（走完全程），episode长度~1110，score=270.71；forward_reward占总奖励绝对量的88.8%，stability_penalty占11.2%，两者active_rate均为100%，无明显崩溃或徘徊。

2. **behavior_diagnosis**：agent已学会稳定行走并完成地形，但当前策略可能动作幅度过大、能耗偏高——环境事实明确要求"尽量降低能量消耗"，而现有奖励完全没有对关节力矩的约束。

3. **signal_completeness**：前进引导（forward_reward）和稳定约束（stability_penalty）均完备且可达，但**能量效率信号完全缺失**——这是任务声明的核心目标之一，目前无任何组件覆盖。

4. **selected_level**：Level 2。触发条件：必要职责（能量效率）缺失，且该信号在obs/action中直接可用（action各维度的力矩值），无需猜测或代理。

5. **selected_intervention**：新增 `energy_penalty` 组件，对四个关节力矩的平方和施加轻度惩罚，引导策略用更小的力矩完成相同的前进任务。

6. **falsifiable_hypothesis**：加入小幅力矩惩罚后，策略应学会用更经济的关节发力维持步态，外部score应上升（若任务评分内含能耗因子），同时forward_reward不应显著下降（因为小力矩仍可维持前进速度）。

7. **expected_next_round**：`energy_penalty` 的 episode_sum_mean 应为小负值（约-10到-30量级，取决于力矩水平），`forward_reward` 保持相近水平，外部 score 若有能耗因子应上升；若 score 下降且 forward_reward 明显降低，说明惩罚过强抑制了移动。

8. **main_risk**：力矩惩罚可能使agent减少发力导致前进速度下降，若系数过大可能造成"为省电而不敢迈步"。选择 w_energy=0.05（约为forward_reward系数1.0的5%）作为轻量起步。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重
    w_forward = 1.0
    w_angle = 0.5
    w_vertical_vel = 0.1
    w_angular_vel = 0.05
    w_energy = 0.05

    # 主学习信号：前进速度奖励
    forward_vel = next_obs[2]
    forward_reward = w_forward * forward_vel

    # 稳定/安全约束
    angle = abs(next_obs[0])
    vertical_vel = abs(next_obs[3])
    angular_vel = abs(next_obs[1])
    stability_penalty = -(w_angle * angle + w_vertical_vel * vertical_vel + w_angular_vel * angular_vel)

    # 能量效率：惩罚过大的关节力矩
    torque_sum_sq = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -w_energy * torque_sum_sq

    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty
    }
    return float(total_reward), components
```
