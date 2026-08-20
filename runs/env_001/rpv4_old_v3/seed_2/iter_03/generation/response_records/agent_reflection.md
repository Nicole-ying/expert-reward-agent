# Response Record

1. evidence：current 所有 episode 都过早终止（len=68），success_landing_bonus 几乎不触发（active_rate=0.1%），progress_shaping 是唯一正向驱动信号，但其鼓励快速接近目标，而缺少对 crash 的惩罚，导致 agent 学会全速坠毁以获取 progress shaping 后结束回合。
2. behavior_diagnosis：agent 学会了“全速冲向地面并 crash”的失败策略，因为任何接触都会终止回合，且 progress_shaping 对接近地面给正奖励，无 crash 负反馈。
3. signal_completeness：任务需要成功着陆（刚性），但当前奖励缺乏对 crash 的明确惩罚，也未提供有效的减速/稳定引导。成功信号仍为稀疏且无法触及；缺少失败负信号使 crash 成为局部最优。
4. selected_level：Level 1（系数/惩罚调整）。progress_shaping 形态本身合理，但需要补充 crash 惩罚以阻断 crash 路径，引导 agent 尝试更安全的着陆。
5. selected_intervention：新增 `crash_penalty` 组件，在接触地面但未满足软着陆条件、或 x 位置超出视区边界时给予 -100，与 success_landing_bonus 互斥，其余组件不变。
6. falsifiable_hypothesis：添加与成功对称的 crash 惩罚后，agent 将被迫避免高速接触和越界，从而逐渐探索减速并在靠近目标时精准控制，最终触发 success_landing_bonus，score 应上升，终止方式可能出现成功终止或 len 增加。
7. expected_next_round：score 改善（负分幅度缩小），可能出现少量成功着陆（terminated 中出现成功状态），len 可能增加或持平，crash_penalty 会有非零激活率，success_landing_bonus 激活率可能略有上升。
8. main_risk：-100 惩罚可能过于严厉，导致 agent 过早害怕靠近地面而悬停在高处，使 progress_shaping 优势消失，回合长度大幅增加且分数不升反降；若出现此情况需降低 crash_penalty 系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造 (progress_shaping)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus)
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.1 and
        abs(next_x_vel) < 0.2 and
        abs(next_y_vel) < 0.2 and
        abs(next_body_angle) < 0.1 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 坠毁/越界惩罚 (crash_penalty)
    out_of_bounds = abs(next_x_pos) > 1.5
    # 接触地面但未满足软着陆条件，或出界，均视为失败
    crash = (not landed) and (contact_any or out_of_bounds)
    crash_penalty_val = -100.0 if crash else 0.0

    # 4. 姿态惩罚
    angle_penalty_val = -0.5 * (body_angle ** 2)

    # 5. 动作代价
    action_cost_val = -0.02 if action != 0 else 0.0

    # 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        crash_penalty_val +
        angle_penalty_val +
        action_cost_val
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "crash_penalty": crash_penalty_val,
        "angle_penalty": angle_penalty_val,
        "action_cost": action_cost_val
    }

    return float(total_reward), components
```
