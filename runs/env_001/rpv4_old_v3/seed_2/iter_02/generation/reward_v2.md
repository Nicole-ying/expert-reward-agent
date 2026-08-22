1. `evidence`：首轮得分为-155.975，进度整形每步+0.01但总ep_sum仅+1.58，动作代价和角度惩罚合计每步-0.019，ep_sum共-2.94，造成整体负奖励；着陆速度惩罚激活率仅5.3%，几乎不影响训练；10/20 episode 提前终止(early_terminal)，表明策略在落地前即因碰撞或飞出而失败，且无成功信号引导软着陆。
2. `behavior_diagnosis`：策略学习到向目标缓慢移动（progress_shaping为正），但姿态不稳定（角度惩罚高）、引擎使用频繁（动作代价高），缺少对成功软着陆的奖励，导致从未稳定着陆，常以高速或大角度触地/出界被终止。
3. `signal_completeness`：当前信号缺少“任务完成”职责，仅有过程引导（靠近）、安全约束（角度、速度惩罚），无法区分成功终结与失败终结。成功奖励是必须且可达的——可通过next_obs的姿态、速度、接触推断软着陆。
4. `selected_level`：Level 2——着陆速度惩罚组件形式错误（稀疏惩罚且门控过窄，形同虚设），需替换为成功着陆奖励，即从惩罚形态变为正向bonus。
5. `selected_intervention`：将`landing_velocity_penalty`组件整体替换为`success_landing_bonus`，基于next_obs的接触、位置、速度、角度给出稀疏大正奖励。
6. `falsifiable_hypothesis`：添加成功奖励后，策略在成功沾盘瞬间获得+100，远大于所有单步惩罚，应能强化接近并软着陆的行为，使最终得分由负转正、terminated中成功终止占比增加。
7. `expected_next_round`：预期下一轮评估中score显著提高（至少突破负值），terminated数量仍为20/20但early_terminal减少，progress_shaping的ep_sum略增，angle_penalty和action_cost略降，success_landing_bonus出现非零mass。
8. `main_risk`：成功奖励阈值过宽可能导致agent学会以微小的姿态/速度不完全达标但仍获得奖励的行为（如半碰半浮），但阈值设计已合理（距离<0.1、速度<0.2、角度<0.1rad、接触），最坏情况下可通过后续调整阈值解决。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Lunar Lander 风格任务奖励函数 v2。

    修改:
    - 移除无效的 landing_velocity_penalty，新增 success_landing_bonus:
      当 next_obs 满足软着陆条件时给予 +100 稀疏大奖励，解决 v1 无成功信号的问题。
    - 其余组件保持不变。
    """
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 主学习信号: 距离势能塑造 (goal_progress_guidance)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus)
    # 判定条件: 距离目标极近、速度很小、姿态竖直、至少一个支撑脚触地
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.1 and
        abs(next_x_vel) < 0.2 and
        abs(next_y_vel) < 0.2 and
        abs(next_body_angle) < 0.1 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    # 5. 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```