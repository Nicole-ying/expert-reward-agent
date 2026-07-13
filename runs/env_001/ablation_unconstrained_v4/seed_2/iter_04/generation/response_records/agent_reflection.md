# Response Record

从前一轮的训练反馈来看，agent在大部分episode中无法触发终止（17/20 truncated），只有3次成功触发终止（疑似成功着陆）。contact_reward活跃但只在部分步骤（26%），且主要基于任意支撑腿接触，未能有效引导双腿稳定着陆。height_near_reward在无接触的85%步骤中提供了低高度奖励，可能诱使agent低空徘徊而不完成最终着陆。progress和y_progress贡献过小，导致下降速度缓慢。水平和姿态惩罚几乎不触发，说明控制已较好，但缺少对稳定静止状态的强力激励。

因此，修改目标：
1. 强化“双腿稳定着陆”的引导，使用双腿接触与全状态稳定性的乘积奖励，并在接近完全静止时提供较高奖励。
2. 降低徘徊激励：将height_near_reward与接触绑定，仅在已有支撑腿接触时奖励低高度。
3. 增大下降进度权重，加快接近速度。
4. 保留单腿接触的微弱奖励作为平滑过渡，避免完全移除导致学习中断。
5. 移除几乎不起作用的horizontal_penalty并微调其他参数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态（用于进度计算）
    x_old, y_old = obs[0], obs[1]
    
    # 下一状态
    x_new = next_obs[0]
    y_new = next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 辅助量 ----------
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5
    any_contact = max(left_contact, right_contact)
    double_contact = left_contact * right_contact  # 0.0 或 1.0

    # ---------- 参数 ----------
    # 进度
    w_progress = 5.0
    w_y_progress = 10.0

    # 水平对准（保持不变，温和引导）
    w_alignment = 0.5

    # 单腿接触微弱引导（维持探索）
    w_any_contact = 1.0

    # 双腿稳定着陆（核心）
    w_stable = 5.0
    vx_stable_tol = 0.2
    vy_stable_tol = 0.2
    angle_stable_tol = 0.1
    angvel_stable_tol = 0.1

    # 高度奖励（仅在接触时）
    w_height_near = 1.0
    ground_threshold = 0.5

    # 安全约束（与之前类似，但略降低速度惩罚权重以避免过于保守）
    w_vel = 0.1
    w_att = 0.1
    vx_max = 0.5
    vy_max = 0.5
    angle_max = 0.2
    angvel_max = 0.2

    # ---------- 组件 ----------
    # 1. 稠密进展
    progress = w_progress * (dist_old - dist_new)
    y_progress = w_y_progress * (y_old - y_new)

    # 2. 水平对准
    alignment = w_alignment * (1.0 / (1.0 + x_new ** 2))

    # 3. 单腿接触奖励（平滑过渡，非必需但降低学习难度）
    any_contact_reward = w_any_contact * any_contact

    # 4. 双腿稳定接触奖励
    # 稳定性因子：越接近零，因子越接近1
    factor_vx = max(0.0, 1.0 - abs(vx_new) / vx_stable_tol)
    factor_vy = max(0.0, 1.0 - abs(vy_new) / vy_stable_tol)
    factor_angle = max(0.0, 1.0 - abs(angle_new) / angle_stable_tol)
    factor_angvel = max(0.0, 1.0 - abs(angvel_new) / angvel_stable_tol)
    stability = factor_vx * factor_vy * factor_angle * factor_angvel
    stable_contact_reward = w_stable * double_contact * stability

    # 5. 接近地面奖励，仅当已有支撑腿接触时（避免无接触低飞徘徊）
    height_near_reward = w_height_near * any_contact * max(0.0, ground_threshold - y_new)

    # 6. 安全速度与姿态约束（保持软边界，几乎不触发，仅作安全网）
    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * (excess_vx ** 2 + excess_vy ** 2)

    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * (excess_angle ** 2 + excess_angvel ** 2)

    # ---------- 总和 ----------
    total_reward = (progress + y_progress + alignment +
                    any_contact_reward + stable_contact_reward +
                    height_near_reward +
                    vel_penalty + att_penalty)

    components = {
        'progress': progress,
        'y_progress': y_progress,
        'alignment': alignment,
        'any_contact_reward': any_contact_reward,
        'stable_contact_reward': stable_contact_reward,
        'height_near_reward': height_near_reward,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty
    }

    return float(total_reward), components
```
