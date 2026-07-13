# Response Record

## 诊断

1. **evidence**：第二轮得分从 150.58 升至 221.40（+47%），ep_len 从 756 升至 1199。但 score_range [11.6, 308.7] 极宽，说明策略不一致：部分 episode 走到终点（高分），部分在早期就摔倒（低分）。forward_reward/stability_cost 比值约 0.22，尺度合理；本轮未修改过其他组件。

2. **behavior_diagnosis**：策略平均走得远但不一致——部分 episode 步态失控导致提前摔倒。当前奖励只鼓励前向速度和惩罚身体倾角，但未约束动作本身的剧烈程度，可能允许高力矩、低效率的抽搐式步态，这种步态在部分 episode 中引发失衡摔倒。

3. **signal_completeness**：任务明确要求"尽量减小能量消耗"，当前奖励缺少动作效率信号。前向进度和稳定性信号已存在且尺度合理，但能量效率职责完全缺失——这同时是任务目标之一，也是提升步态一致性的潜在杠杆。已完成职责（progress + stability）基本完备且可达。

4. **selected_level**：Level 2 — 必要信号（动作能量效率）缺失。stability_cost 尺度合理（ratio≈0.22），无需 Level 1 调整。

5. **selected_intervention**：新增 `action_cost` 组件：对四个关节力矩的均方值施加轻量惩罚（w=0.05），鼓励平滑、低能耗步态。保持 forward_reward 和 stability_cost 不变。

6. **falsifiable_hypothesis**：轻量动作代价应抑制高力矩抽搐动作，使步态更平滑一致，从而减少早期摔倒（提升 score 下限），同时因 w=0.05 极轻（预期占总奖励 <5%），不应压制前进动机或导致不动。

7. **expected_next_round**：score 应上升（尤其低分 episode 应明显改善），score_range 应收窄，stability_cost 的 magnitude 可能随步态变平滑而下降，forward_reward 应持平或略升。

8. **main_risk**：若动作代价实际上过重（joint 力矩天然偏大），可能触发 locomotion_energy_domination——策略为省力而减小步幅，导致前进速度下降。验证时若 forward_reward 显著下降且 score 未刷新 best，则下一轮应降 w_action 或改为门控形态。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    forward_velocity = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_velocity

    # 稳定性约束：惩罚身体倾角、角速度和剧烈垂直振荡
    hull_angle = next_obs[0]
    hull_angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    w_angle = 1.0
    w_angvel = 0.2
    w_vvel = 0.4
    stability_cost = (
        -w_angle * abs(hull_angle)
        - w_angvel * abs(hull_angular_vel)
        - w_vvel * abs(vertical_vel)
    )

    # 能量效率：轻量动作代价，抑制高力矩抽搐，鼓励平滑高效步态
    w_action = 0.05
    action_cost = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0

    total_reward = forward_reward + stability_cost + action_cost

    components = {
        "forward_reward": forward_reward,
        "stability_cost": stability_cost,
        "action_cost": action_cost,
    }

    return float(total_reward), components
```
