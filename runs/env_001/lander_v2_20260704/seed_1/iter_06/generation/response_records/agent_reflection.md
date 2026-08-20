# Response Record

`evidence`：本轮得分108.45（较上轮-87.91大幅提升），episode长度629（上轮~70），14/20正常终止、6/20截断；approach_landing_reward以95.5% magnitude share绝对主导（episode_sum_mean=606.66），distance_cost和stability_cost各仅占~2%；该代码为首次使用improvement型奖励并显著放大系数，策略从此前快速失败转变为缓慢接近并着陆。

`behavior_diagnosis`：策略已学会存活并接近着陆垫（14/20成功终止），但episode长达629步，说明策略正在利用improvement型proxy——通过极缓慢的渐进式接近来最大化累计改善奖励，而非快速高效完成着陆。6/20截断进一步印证部分episode因过于缓慢而耗尽时间。

`signal_completeness`：当前奖励具备过程引导（接近、减速、姿态、触地改善）和稳定约束（distance_cost、stability_cost），但缺失任务明确要求的"尽量少使用引擎推力"信号，且improvement型proxy缺乏时间压力或效率激励，导致proxy可被缓慢进展无限套利。

`selected_level`：Level 2。触发条件：improvement型proxy（proxy_to_completion_alignment模式）被策略系统性套利——外部任务完成（成功着陆）但效率极差（episode过长），且缺失引擎使用约束这一任务明确要求。

`selected_intervention`：新增`engine_penalty`组件——对任何非零动作（action 1/2/3使用引擎）施加固定惩罚，惩罚"使用引擎推力"这一行为本身。这是当前奖励完全缺失的任务维度。

`falsifiable_hypothesis`：引擎惩罚将使缓慢悬停（需持续使用引擎维持高度和控制姿态）的净收益下降，而快速滑翔+间歇引擎的着陆策略相对更具优势；episode长度应缩短，引擎使用频率应下降，同时成功着陆率（terminated比例）应保持或改善。

`expected_next_round`：engin_penalty的active_rate应接近非零动作使用率（预计60-80%），magnitude_share应可观但不压倒approach_landing_reward；episode长度应从~629下降；truncated比例应下降（更快完成不再超时）；terminated比例保持或上升；总得分可能因路径缩短而暂时下降，但不应回到负数。

`main_risk`：引擎惩罚过重可能导致策略为避免惩罚而完全不使用引擎（action=0始终），无法完成着陆必需的减速和姿态调整，导致crash终止增加；若出现此情况需降低系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一步状态
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离与稳定性成本
    w_dist = 0.08
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05

    distance_cost = -w_dist * (nx ** 2 + ny ** 2)
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 改善量型 approach_landing_reward：有界差分替代持续状态值
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    curr_dist = (nx ** 2 + ny ** 2) ** 0.5
    prev_vel = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    curr_vel = (vx ** 2 + vy ** 2) ** 0.5
    prev_contact = (obs[6] + obs[7]) / 2.0
    curr_contact = (left_contact + right_contact) / 2.0

    # 接近改善：向目标靠近即获奖
    proximity_improvement = 5.0 * max(0.0, prev_dist - curr_dist)

    # 减速改善：仅在靠近着陆垫时奖励减速（避免远距离时惩罚必要的高速接近）
    proximity_gate = max(0.0, 1.0 - curr_dist / 1.5)
    velocity_improvement = 10.0 * proximity_gate * max(0.0, prev_vel - curr_vel)

    # 姿态改善：减小倾角即获奖
    angle_improvement = 3.0 * max(0.0, abs(obs[4]) - abs(angle))

    # 触地改善：建立支撑腿接触的转移奖励（非持续占有）
    contact_improvement = 30.0 * max(0.0, curr_contact - prev_contact)

    approach_landing_reward = proximity_improvement + velocity_improvement + angle_improvement + contact_improvement

    # 引擎使用惩罚：惩罚任何非零动作（使用引擎推力），鼓励滑翔和最小推力策略
    engine_penalty = -0.3 * (1.0 if action != 0 else 0.0)

    total_reward = distance_cost + stability_cost + approach_landing_reward + engine_penalty

    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'approach_landing_reward': approach_landing_reward,
        'engine_penalty': engine_penalty
    }

    return float(total_reward), components
```
