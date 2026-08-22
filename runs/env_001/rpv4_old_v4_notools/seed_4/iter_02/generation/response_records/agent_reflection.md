# Response Record

## 诊断报告

**evidence**
- 上一轮 score = -121.5，全训练过程中 score 在 -113 ~ -121 之间波动，没有任何改善趋势。ep_len=68.5，所有 episode 都在 150 步内 early_terminal (score<-50)：说明策略始终以失败状态提前结束（极可能是急坠撞地）。
- goal_proximity 在训练全过程中几乎恒定（≈2.15），表明 agent 学会了稳定的向下坠落模式以持续获得“接近目标”的奖励（因为 L1 距离减小主要来自 y 下降）。
- settling_stability 均值 0.44，但 magnitude_share 36%，且 active_rate 100%，可忽略的速度改进完全无法阻止急坠。
- attitude_constraint 始终为负（均值 -1.8），active 100%，表明姿态抖动或大角度机动被连续惩罚，进一步拖累总分。
- landing_contact active_rate 仅 0.2%，均值 0.5，说明成功着陆事件几乎从未发生，无法提供正向引导。

**behavior_diagnosis**
策略学到的行为是快速垂直下坠，以最大化 goal_proximity 中的距离改进，同时忽略姿态稳定和减速，导致早终止、极低分数。这是一个严重错误的代理（proxy）——主信号鼓励向下急坠而非受控着陆。

**signal_completeness**
必要职责缺失：当前奖励中没有有效信号来惩罚高速撞击或奖励安全接触，landing_contact 门控过严且奖励不足，难以成为有效学习信号。goal_proximity 对 y 下降的过度奖励是毒信号。

**selected_level: L3**
得分始终接近负值，远低于目标 200，且主信号框架将距离改进错误地导向跌落，符合“score 始终接近零或负→重写整个 reward，换完全不同的主信号框架”的规则。第一轮即发现根本错误，必须重构。

**selected_intervention**
完全重写奖励函数，采用经典着陆器势能差框架：
- **主信号**：shaping = 10 × (current_L1_distance − next_L1_distance)，保持向目标移动的引导，但系数加大以抵消负面惩罚。
- **强约束**：全局速度惩罚 −0.2×L1速度，驱策减速。
- **姿态约束**：全局角度/角速度惩罚，直接抑制翻滚。
- **着陆成功奖励**：当双接触、低速、对准中心时，给予持续较大奖励 +10，作为明确的任务完成信号。
- **燃料效率**：轻量引擎使用惩罚。

**falsifiable_hypothesis**
新的势能差 + 全局速度/姿态惩罚 + 高着陆奖励，将使快速下坠不再有利可图（速度惩罚会吃掉距离改善的奖励），必须同时减速且稳定姿态才能积累正奖励，从而使策略学会受控下降与着陆；预计下一轮 score 由负转正，landing_contact 活跃度大幅上升，episode 长度增加，early_terminal 比例下降。

**expected_next_round**
下一轮策略将学会减速并尝试保持姿态稳定，landing_contact 触发频率明显提高，总分预计回到正值（可能在 -50 ~ 50 区间），agent 会出现符合预期的接近目标、减速、接触行为。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2 — 完全重构。
    采用势能差 + 全局速度/姿态惩罚 + 高着陆奖励框架，
    引导主体受控下降并安全停靠。
    """
    # 当前状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    # 下一状态
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ========== A. 目标接近 shaping ==========
    # 势能：对 L1 距离取负，距离越小势能越高
    current_potential = -(abs(x_pos) + abs(y_pos))
    next_potential = -(abs(next_x_pos) + abs(next_y_pos))
    goal_proximity = 10.0 * (next_potential - current_potential)  # 距离减小得正奖励

    # ========== B. 速度惩罚 ==========
    current_speed = abs(x_vel) + abs(y_vel)
    next_speed = abs(next_x_vel) + abs(next_y_vel)
    # 对全局速度进行惩罚，鼓励减速；这里用均值，防止单步过大抖动
    speed_penalty = -0.2 * (current_speed + next_speed) / 2.0

    # ========== C. 姿态约束 ==========
    # 惩罚偏离竖直的绝对角度和角速度
    angle_penalty = -0.5 * abs(next_body_angle)
    ang_vel_penalty = -0.2 * abs(next_ang_vel)
    attitude_constraint = angle_penalty + ang_vel_penalty

    # ========== D. 安全着陆接触奖励 ==========
    near_center = abs(next_x_pos) < 0.2 and abs(next_y_pos) < 0.2
    low_speed_contact = next_speed < 0.5
    both_contacts = (next_left_contact > 0.5 and next_right_contact > 0.5)

    landing_contact = 0.0
    if near_center and low_speed_contact and both_contacts:
        landing_contact = 10.0  # 持续鼓励在目标上双足接触、低速

    # ========== E. 燃料效率 ==========
    # 对任何非悬停动作施以轻惩罚（0: no_engine）
    fuel_efficiency = -0.05 * float(action != 0)

    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        speed_penalty +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )

    components = {
        'goal_proximity': goal_proximity,
        'speed_penalty': speed_penalty,        # settling_stability 并入速度惩罚
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components
```
