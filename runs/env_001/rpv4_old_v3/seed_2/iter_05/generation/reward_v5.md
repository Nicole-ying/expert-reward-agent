1. `evidence`：v4通过放宽success_bonus阈值将score从-203提升至-114，active_rate从0.0%升至0.4%，但该信号仍然极度稀疏（magnitude_share 83.4%却仅0.4%触发），大部分episode未获正奖励，导致总分为负。
2. `behavior_diagnosis`：策略仅在少数episode成功触发软着陆+100，多数episode以高速接触或脱离终止，未能学会精确减速与姿态控制，训练仍由负奖励主导。
3. `signal_completeness`：progress_shaping提供连续接近引导，angle_penalty约束倾斜，action_cost抑制推力，success_bonus给出完成信号，职责完备但完成信号过于苛刻，需提升可达性。
4. `selected_level`：Level 2，基于v4成功经验进一步放宽success_landing_bonus的触发阈值以提升active_rate，不做框架级重建。
5. `selected_intervention`：将success_landing_bonus的判定条件中的距离阈值由0.2提至0.4，水平/垂直速度阈值由0.5提至1.0，角度阈值由0.2提至0.5，保持接触条件不变。
6. `falsifiable_hypothesis`：放宽后触发率应升至5%~15%，每episode平均bonus从30增至50以上，总score有望升至0附近或正数，terminated中成功着陆占比明显增加。
7. `expected_next_round`：success_landing_bonus active_rate >2%，ep_sum_mean >50，score > -50，平均episode长度可能因更早触发成功而继续缩短。
8. `main_risk`：角度0.5 rad（≈28°）可能容忍略微倾斜着陆，但接触条件保证支撑脚触地，不致严重翻倒；若后续出现bad landing但terminated，可重新收紧角度或速度条件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_reward = 5.0 * ((-next_dist) - (-dist))

    # 2. 成功着陆奖励 —— 进一步放宽阈值
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.4 and
        abs(next_x_vel) < 1.0 and
        abs(next_y_vel) < 1.0 and
        abs(next_body_angle) < 0.5 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

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