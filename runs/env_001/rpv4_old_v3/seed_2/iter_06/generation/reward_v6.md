1. `evidence`：v5将成功条件阈值大幅放宽后，success_landing_bonus每episode平均高达1195，active_rate仅1.6%，回合长度从68激增至740，表明agent在接触后保持软着陆状态持续获得bonus而未被终止，形成了奖励劫持；score仍为-46.5，未达成目标。
2. `behavior_diagnosis`：策略当前失败在于学会了“假着陆”——在接触后维持低速、小角度、距离中心的状态，每一步都满足landed条件从而连续获取100奖励，导致episode被truncated而非成功终止，外部任务表现未改善。
3. `signal_completeness`：现有信号缺少一次性着陆事件奖励，状态值形式的success_landing_bonus可被无限利用；需要改为基于接触转变的事件奖励。
4. `selected_level`：Level 2，success_landing_bonus的数学形态传达了错误激励（状态值可连续触发），需改为事件触发并收紧阈值。
5. `selected_intervention`：将success_landing_bonus修改为仅在从无接触转为有接触时触发，且收紧软着陆条件（距离<0.3，速度绝对值<0.3，角度绝对值<0.3），奖金提高至200。
6. `falsifiable_hypothesis`：改为一次性事件奖励并收紧阈值后，agent不再有动机在接触后滞留，将趋向成功着陆后自然终止，episode长度应显著下降，terminated比例上升，success_bonus的ep_sum将下降但集中在真正的成功episode，score因此上升。
7. `expected_next_round`：平均len回到100以内，terminated数量增加，success_landing_bonus的ep_sum大幅下降（如30~60），score升至-80~0之间或更高。
8. `main_risk`：阈值收紧可能导致bonus触发极度稀疏（active_rate < 0.1%），学习困难；agent可能学会离开再接近以重复触发事件，需后续观察。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_reward = 5.0 * ((-next_dist) - (-dist))

    # 2. 成功着陆奖励 —— 基于接触转变事件，收紧阈值
    prev_contact = (left_contact > 0.5) or (right_contact > 0.5)
    curr_contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landing_event = (
        (not prev_contact) and curr_contact and
        next_dist < 0.3 and
        abs(next_x_vel) < 0.3 and
        abs(next_y_vel) < 0.3 and
        abs(next_body_angle) < 0.3
    )
    success_bonus = 200.0 if landing_event else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

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