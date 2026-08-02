# Response Record

1. `evidence`：最终得分 216.19 超越目标；terminated=20/20 无截断；组件 `angle_penalty` 的 active_rate=0%，mean=0，signed_share=0%，完全未激活，成为无功能僵尸。
2. `behavior_diagnosis`：策略已学会稳定着陆，所有评估回合均正常终止，行为收敛，无失败或徘徊迹象。
3. `signal_completeness`：位置、速度、朝向均已通过 soft_landing_progress 和距离差分覆盖；angle_penalty 试图约束的角度已被 soft_landing_progress 的低角度因子替代，职责重复。
4. `selected_level`：Level 2 结构变换 — 触发条件为僵尸组件 active_rate=0%，其约束意图已在别处实现，应删除以保持奖励简洁。
5. `selected_intervention`：唯一目标组件为 `angle_penalty`，将其从奖励计算中完全移除，不再定义和使用权重 `w_health`。
6. `falsifiable_hypothesis`：由于 `angle_penalty` 从不激活，删除后策略接收的奖励信号不变，因此下一轮评估得分应在原值附近（±5，因种子随机性），且不会引入新失败模式。
7. `expected_next_round`：得分与当前相近（约 211–221），episode 长度和终止模式不变，angle_penalty 不再出现在组件统计中。
8. `main_risk`：无实质风险。代码删除仅清理死代码，唯一潜在风险是误删引入语法错误，但代码仍保留其余组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs / next_obs : [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]
    # goal is at origin (0,0); x,y relative to target pad
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_distance = dist_old - dist_new  # positive when approaching

    # Soft landing progress: encourage closeness, low speed, low angle
    near_goal = 1.0 / (1.0 + 5.0 * dist_new)
    speed_sq = next_obs[2]**2 + next_obs[3]**2
    low_speed = 1.0 / (1.0 + 10.0 * speed_sq)
    abs_angle = abs(next_obs[4])
    low_angle = 1.0 / (1.0 + 20.0 * abs_angle)
    soft_progress = near_goal * low_speed * low_angle

    # Engine usage penalty: penalize any thrust action (discrete actions 1,2,3)
    engine_penalty = 1.0 if action != 0 else 0.0

    # Weights
    w_dist = 10.0
    w_soft = 2.0
    w_engine = 0.01

    total = (w_dist * delta_distance +
             w_soft * soft_progress -
             w_engine * engine_penalty)

    components = {
        'distance_delta': w_dist * delta_distance,
        'soft_landing_progress': w_soft * soft_progress,
        'engine_penalty': -w_engine * engine_penalty,
    }
    return float(total), components
```
