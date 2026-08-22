# Response Record

**1. 诊断报告**

- **evidence**：
  - 最终得分 -110.57，所有 20 个评估集均提前终止（terminated=20），episode 长度仅 68.45，说明主体在到达目标前坠毁或飞出边界。
  - 奖励构成中 `speed_penalty` 量级占比 51.4%（均值 -18.34），远超正信号 `goal_proximity` 的 +10.84，导致每集净奖励为负。
  - `goal_proximity` 全程几乎无变化（cp10%~cp100% 稳定在 10.8 左右），说明势能差信号未能有效引导主体向目标移动；`speed_penalty` 同样全程不变，表明 RL 无法优化它。
  - `landing_contact` 活跃率仅 0.5%，稀疏且几乎无引导作用；`fuel_efficiency` 活跃率 4%，表明主体极少使用引擎（action≠0）。
  - 训练趋势：仅 `attitude_constraint` 从 -6.87 改善至 -1.52，说明主体只学会了压低姿态误差，但速度与位置未优化。
- **behavior_diagnosis**：
  主体因全局速度惩罚过强，倾向于不使用引擎，仅靠初始随机速度和角动量滑行，同时学习保持竖直姿态以降低姿态惩罚。由于移动速度受限，无法在终止前抵达中心目标垫板，导致所有 episode 提前坠毁或超界。当前策略本质是“被动滑行 + 姿态自稳”，而非“主动导航到目标”。
- **signal_completeness**：
  缺少与任务阶段解耦的速度指导信号。当前的全局速度惩罚与目标接近需求矛盾——要接近目标就需要速度，但速度立即被惩罚。必须改为**阶段门控型速度惩罚**，允许远离目标时高速移动，仅在接近目标时严厉限制速度。
- **selected_level**：**L2**
  - 理由：仅调系数（L1）无法解决“远距离速度被错误惩罚”的结构性问题，必须改变 `speed_penalty` 的数学形态，使其依赖距离。
  - 不与历史重复：历史 iter_2 使用全局速度惩罚，本次改为距离门控速度惩罚，属于完全不同的语义方向。
- **selected_intervention**：
  目标组件：`speed_penalty`。
  修改：将 `speed_penalty = -0.2 * (|vx|+|vy|)/2` 改为：
  ```
  dist = sqrt(x_pos² + y_pos²)
  gating = 1.0 / (1.0 + 5.0 * dist)
  speed_penalty = -0.2 * avg_speed * gating
  ```
  这样，当主体远离目标 (dist 大) 时门控值很小，速度惩罚接近零，可自由高速接近；当靠近目标时门控上升至 1，惩罚有效驱动减速。同时保留姿态、燃料和着陆信号不变。
- **falsifiable_hypothesis**：
  移除对远距离速度的惩罚后，主体将学会高速向目标移动，`goal_proximity` 的正奖励会明显放大（因为每步距离减小更多），整体净奖励将由负转正或显著上升；接近目标时门控增大迫使减速，配合着陆信号，可能首次触发稳定着陆。预计下一轮得分将大幅改善（负值减小甚至接近正值）。
- **expected_next_round**：
  主体将表现出“快速下降接近目标，临近时减速，尝试保持姿态”的行为。`speed_penalty` 的全程均值会下降，`landing_contact` 活跃率可能因接近目标而提升。整体 score 预计不再被速度惩罚支配，可能提升到 -30 至 0 区间。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v4 — 引入距离门控速度惩罚。
    保留势能差目标接近，将全局速度惩罚改为阶段敏感：远离目标时弱化，靠近时强化。
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
    goal_proximity = 10.0 * (next_potential - current_potential)

    # ========== B. 速度惩罚（距离门控版） ==========
    # 计算平均速度大小
    current_speed = abs(x_vel) + abs(y_vel)
    next_speed = abs(next_x_vel) + abs(next_y_vel)
    avg_speed = (current_speed + next_speed) / 2.0

    # 当前位置到目标的欧氏距离
    dist_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # 门控因子：远距离时接近0，近距离时接近1
    gating = 1.0 / (1.0 + 5.0 * dist_to_target)
    speed_penalty = -0.2 * avg_speed * gating

    # ========== C. 姿态约束 ==========
    angle_penalty = -0.5 * abs(next_body_angle)
    ang_vel_penalty = -0.2 * abs(next_ang_vel)
    attitude_constraint = angle_penalty + ang_vel_penalty

    # ========== D. 安全着陆接触奖励 ==========
    near_center = abs(next_x_pos) < 0.2 and abs(next_y_pos) < 0.2
    low_speed_contact = (abs(next_x_vel) + abs(next_y_vel)) < 0.5
    both_contacts = (next_left_contact > 0.5 and next_right_contact > 0.5)

    landing_contact = 0.0
    if near_center and low_speed_contact and both_contacts:
        landing_contact = 10.0

    # ========== E. 燃料效率 ==========
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
        'speed_penalty': speed_penalty,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components
```
