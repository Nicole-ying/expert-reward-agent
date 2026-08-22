# Response Record

1. `evidence`：当前得分-40，全部episode以terminated结束（无截断），且goal_proximity(ep_sum_mean=2.14)是唯一活跃信号，soft_landing_penalty仅1.5%激活，两个crash_prevention完全无触发，agent被距离差分奖励驱动但缺乏着陆成功引导。
2. `behavior_diagnosis`：策略只学到向目标垫靠近，但未学会减速与稳定着陆，导致高速度硬着陆或姿态失控而失败，整体负分。
3. `signal_completeness`：缺乏正向着陆成功信号，现有的软着陆惩罚只在极近处激活且无效，没有利用腿接触和位置/速度条件去激励精确、稳定着陆，故必要职责不完整。
4. `selected_level`：Level 2 — 将无效的soft_landing_penalty改为基于着陆条件的持续正向奖励，以提供可学习的成功引导。
5. `selected_intervention`：唯一目标组件为soft_landing_penalty，将其替换为landing_success_reward，条件为双腿接触、近中心、低速、小姿态角，持续给予正奖励。
6. `falsifiable_hypothesis`：添加着陆成功信号后，agent会在dist接近0时获得持续激励，从而学会减速并维持姿态，减少失败硬着陆。
7. `expected_next_round`：下一轮score应明显上升，landing_success_reward的active_rate升高并成为重要贡献，episode的terminated成功率提高，episode长度可能略微下降（更快稳定着陆）。
8. `main_risk`：着陆条件过于严格导致奖励稀疏，agent仍以goal_proximity为主导无改进；或条件过松导致agent在非最终稳定时获得奖励而停滞，但可通过合理阈值缓解。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    # 腿接触信息（0/1）
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励（替代原 soft_landing_penalty）
    # 当双腿接触、位置接近原点、速度小、姿态水平时给予持续正奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15   # 用于水平/垂直线速度绝对值
    landing_ang_thresh = 0.1    # 角度绝对值
    w_landing = 1.0

    contacts = left_contact + right_contact           # 0, 1, 2
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    # 只有两条腿都与地面接触且满足其他条件才算成功着陆
    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 碰撞预防（保留原有逻辑，尝试微调使之可能触发）
    x_bound = 0.9
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 4. 时间惩罚
    living_penalty = -0.002

    total_reward = (reward_prox + reward_landing +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```
