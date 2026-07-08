## 诊断与干预

**evidence**：上一轮将w_forward从1.0提高到1.3后，得分从318降至305，episode长度从1269骤降至948步，20/20回合因摔倒终止；forward_reward每步均值从0.362升至0.507，且magnitude_share达81.2%。这直接命中`fast_then_fail`模式——无界线性速度奖励驱动策略以失稳高速换取短期前进。

**behavior_diagnosis**：策略为最大化瞬时前进速度而牺牲姿态稳定，跑得更快但更早摔倒，导致总行进距离下降。纯`forward_vel`是无界线性proxy，与真正目标（速度×存活时间的距离）在高速区出现结构性错位。

**signal_completeness**：当前具备前进引导（forward_vel）、姿态约束（angle+vertical/angular vel）和能耗惩罚，但缺少对"持续存活"的显式激励。无界线性速度proxy在极端值区与任务目标脱钩——这是结构缺陷，不是尺度问题。

**selected_level**：Level 2。`fast_then_fail`证据直接否定当前无界线性速度形态，匹配`unbounded_to_bounded`变换条件。仅回退w_forward到1.0（Level 1尺度修复）不足以防止策略在训练中重新发现高速-摔倒的捷径。

**selected_intervention**：将`forward_reward`从无界线性`w_forward * forward_vel`改为有界饱和形式`w_forward * forward_vel / (1 + k * abs(forward_vel))`。同步设置w_forward=1.1、k=0.5，使正常速度区奖励接近best（~0.36/步），高速区边际收益递减，其他组件不变。

**falsifiable_hypothesis**：饱和函数使高速的边际回报下降，策略不再有动机为极端速度牺牲稳定；应恢复中等可持续步态，episode长度回升至~1200+步，得分超越318。

**expected_next_round**：episode_length从948回升至1100+；forward_reward每步均值从0.507降至0.35-0.40区间；stability_penalty每步均值从-0.081收窄；score超越318.32；terminated仍为20/20但摔倒前步数显著延长。

**main_risk**：若k=0.5过强，正常速度区也会被过度压缩，策略可能退化为缓慢蹒跚（速度过低导致距离下降）；下一轮需同时监控forward_reward每步均值和episode长度，若速度降至0.2以下则k需调小。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重
    w_forward = 1.1
    k_saturate = 0.5
    w_angle = 0.5
    w_vertical_vel = 0.1
    w_angular_vel = 0.05
    w_energy = 0.05

    # 主学习信号：有界饱和前进速度奖励（防止高速-摔倒捷径）
    forward_vel = next_obs[2]
    forward_reward = w_forward * forward_vel / (1.0 + k_saturate * abs(forward_vel))

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