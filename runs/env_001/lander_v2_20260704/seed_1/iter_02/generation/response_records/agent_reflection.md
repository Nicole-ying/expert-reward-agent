# Response Record

1. `evidence`：20/20 episodes 均在 <150 步内以 score<-50 提前终止（均值 68.3 步），soft_landing_bonus 的 active_rate 仅 0.3%，距离成本的 signed_share 达 -84.3% 且全程激活，表明完成信号几乎不可达。
2. `behavior_diagnosis`：agent 快速坠毁，未能学会着陆——稀疏 bonus 在 0.3% 的步骤中触发，无法为"接近→减速→调姿→触地"的多阶段着陆序列提供任何梯度引导，仅靠距离惩罚不足以学习受控下降。
3. `signal_completeness`：缺失可达的完成信号——binary soft_landing_bonus 要求 6 个条件同时满足，active_rate 远低于 1% 阈值，信用分配在结构上不可能。
4. `selected_level`：Level 2 — `sparse_to_dense`：完成 bonus 的结构性稀疏（0.3% << 1%）明确要求将硬性二值事件替换为连续过程证据，而非仅调系数。
5. `selected_intervention`：将 `soft_landing_bonus` 从硬性六条件联合判定，转换为由接近度、速度、姿态角和支撑腿接触的连续有界指数得分之和构成的 `approach_landing_reward`。
6. `falsifiable_hypothesis`：连续部分信用信号应使该组件的 active_rate 大幅上升，为着陆条件提供可达梯度，降低早期坠毁率并提升 score。
7. `expected_next_round`：新组件的 active_rate 应远高于 0.3%（理想 >50%），episode_length 应超过 68，score 应从 -119 改善，early_terminal 比例应下降。
8. `main_risk`：接近度和速度的连续奖励可能被悬停 exploit——但 distance_cost 仍惩罚偏离、body_not_awake_or_settled 需实际稳定在垫上才能终止，净激励应仍倾向完成着陆。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取下一步状态
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 权重和阈值（distance_cost 与 stability_cost 保持不变）
    w_dist = 1.0
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05

    # 组件1: 距离成本（保持不变）
    distance_cost = -w_dist * (nx ** 2 + ny ** 2)

    # 组件2: 稳定性成本（保持不变）
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 组件3: 连续渐进着陆奖励（替代原稀疏 soft_landing_bonus）
    dist = (nx ** 2 + ny ** 2) ** 0.5
    vel_sq = vx ** 2 + vy ** 2

    proximity = 4.0 * (2.718281828 ** (-dist ** 2 / 0.5))
    velocity_ok = 4.0 * (2.718281828 ** (-vel_sq / 0.08))
    angle_ok = 4.0 * (2.718281828 ** (-angle ** 2 / 0.02))
    contact = 4.0 * (left_contact + right_contact) / 2.0

    approach_landing_reward = proximity + velocity_ok + angle_ok + contact

    # 合成总奖励
    total_reward = distance_cost + stability_cost + approach_landing_reward

    # 组件字典
    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'approach_landing_reward': approach_landing_reward
    }

    return float(total_reward), components
```
