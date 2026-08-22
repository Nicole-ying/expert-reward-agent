1. `evidence`：本轮的 success_landing_bonus 主动率为 0%，完全无法触发；历史最佳 v5 中该组件主动率 1.6%、ep_sum 均值 28.08，虽然存在滞留问题但已接近成功着陆信号；len 从 740 骤降至 68 且 terminated 全为 20/20，提示 agent 失去成功信号后仅依赖 progress_shaping 直接坠地。

2. `behavior_diagnosis`：当前策略快速坠地，没有任何减速或软着陆行为，完全依靠 progress_shaping 冲向中心，随即因高速碰撞终止，无法完成安全着陆。

3. `signal_completeness`：任务需要接近目标并软着陆的信号，现有 progress_shaping 只有位置引导，缺乏减速和姿态控制激励；success_landing_bonus 在 v5 能提供正向奖励但导致滞留，调整后可补充这一缺口，信号完备性可修复。

4. `selected_level`：Level 2——修改 success_landing_bonus 的数学形态，从每步状态值奖励改为基于接触转变的一次性事件奖励，使用 v5 的宽松阈值以避免死组件，同时解决滞留。

5. `selected_intervention`：将 success_landing_bonus 改为在支撑脚从无接触到有接触的瞬间，若满足 v5 的软着陆条件（dist<0.4, vel_abs<1.0, angle_abs<0.5）则给予 +100，其余时刻为 0。

6. `falsifiable_hypothesis`：基于转变的一次性奖励既能保证 agent 抵达并软着陆时获得明确的正信号，又能避免奖励滞留；使用 v5 的宽松阈值可确保事件有足够的触发率，从而引导 agent 学习减速和姿态控制，score 回升向 v5 并因 episode 缩短进一步改善。

7. `expected_next_round`：success_landing_bonus 的 active_rate 将从 0% 回升至约 0.5%~1.5%，episode 长度明显缩短（远低于 740），terminated 仍以 crash 为主但出现一定比例的成功终止，score 显著上升（预计 > -60）。

8. `main_risk`：若 agent 在接触瞬间速度仍略高于阈值，事件仍然可能触发率偏低，需要在下一轮根据 active_rate 和 ep_sum 调整阈值。

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

    # 1. 距离势能塑造（与 v5 一致）
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_reward = 5.0 * ((-next_dist) - (-dist))

    # 2. 成功着陆奖励 —— 基于接触转变事件，使用 v5 的宽松阈值
    prev_contact = (left_contact > 0.5) or (right_contact > 0.5)
    curr_contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landing_event = (
        (not prev_contact) and curr_contact and
        next_dist < 0.4 and
        abs(next_x_vel) < 1.0 and
        abs(next_y_vel) < 1.0 and
        abs(next_body_angle) < 0.5
    )
    success_bonus = 100.0 if landing_event else 0.0

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