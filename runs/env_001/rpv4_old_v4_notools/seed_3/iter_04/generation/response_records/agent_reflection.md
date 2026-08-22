# Response Record

**1. 诊断报告**

- **evidence**：  
  - 最终得分 -27.46，全部 20 个评估片段均为 truncated（长度 1000），说明智能体从未成功着陆。  
  - contact_bonus 的 magnitude_share 高达 99.2%（ep_sum_mean ≈ 972），而 goal_proximity 仅 3.87，其他惩罚信号量级极小。该组件完全主导了训练奖励。  
  - contact_bonus 本身是稠密的、基于状态的奖励：距离越近奖励越高，且每条腿接触每步给 1.0。智能体很容易在完成最终着陆前（即双腿稳定接触、低垂直速度）在平台上方盘旋，持续收取该奖励，而不触发任务终止。  
  - checkpoint 趋势显示 contact_bonus 能从 139 增长到 1213，说明强化学习确实在优化该信号，但任务得分最高仅 84（cp90%)，随后又跌回 -40，说明该信号与真实任务（安全着陆）严重脱钩——即典型的 reward misspecification（“占据好状态即可持续获奖”）。

- **behavior_diagnosis**：  
  智能体学到的是**最大限度地停留在目标区域，并让一条或两条腿频繁接触，却从不完成稳定着陆**。由于接触奖励每步持续发放，即使没有真正完成任务，总奖励也能很高。这解释了满分时的接触奖励巨大而任务得分仍为负的现象。当前策略是“徘徊骗取稠密奖励”，必须纠正。

- **signal_completeness**：  
  必要职责：①驱近（已有 goal_proximity，但信号偏小）②减速（velocity_penalty 被距离门控，合理）③姿态稳定（orientation_penalty 合理）④**最终成功着陆事件**（缺失）⑤**接触进展的事件性奖励**以替代状态奖励。当前缺失一次性成功奖励，且接触奖励是持续性的，必须替换为事件型奖励。

- **selected_level**：**L2**  
  理由：需要改变 contact_bonus 的数学形态——从稠密状态奖励改为“接触事件 + 一次性成功着陆事件”，属于 L2 语义修改。权重会适当微调，但核心是公式形态变化。

- **selected_intervention**：  
  目标组件 `contact_bonus`（重命名为 `landing_progress`）。  
  具体修改：  
  1. 移除原有的距离衰减稠密奖励 `landing_bonus`（避免状态奖励）。  
  2. 将 `leg_bonus` 改为对每条腿从 0→1 转变的事件奖励（每股奖励 5.0），鼓励探索接触而非持续驻留。  
  3. 新增一次性成功着陆事件奖励：当两条腿同时接触且垂直速度低于阈值（如 0.5）且前一状态不满足此条件时，发放 100.0 奖励。  
  4. goal_proximity 的权重略微提升（w_x 1→2, w_y 2→4），加强接近目标的驱动力。

- **falsifiable_hypothesis**：  
  将接触奖励从事件‑持续状态转为事件‑进展，并引入一次性成功奖励，会迫使智能体**必须完成真正着陆**才能获得大部分正向信号；盘旋不再频繁受益，因此策略会转向最终安全着陆。预计外部任务得分将显著正增长。

- **expected_next_round**：  
  策略在初期可能因失去持续奖励而短暂下降，但随训练进展，将学会快速接近、双足接触并稳定降落到平台，episode 长度明显缩短，终止率上升（成功着陆）。外部任务得分有望跃升至正值区间（>50 分）。

---

**2. 完整 Python 代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3 reward: v2 improvements + turn contact signal from dense state reward
                into event-based progress (leg touch events) and a one-time
                success bonus when both legs touch at low vertical speed.
    
    obs / next_obs layout: [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    """
    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_cur = obs[6]
    right_cont_cur = obs[7]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement delta on squared distance) ---
    #     increased weights to give stronger pull toward the pad
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity damping (gate by distance, same as before) ---
    w_v = 0.05
    k_gate = 1.0
    dist_nxt = abs(x_nxt) + abs(y_nxt)
    gate = 1.0 / (1.0 + k_gate * dist_nxt)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation penalty ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. landing progress (event-based, replaces dense contact_bonus) ---
    # a) leg touch events – one bonus per leg transition 0→1
    w_leg_event = 5.0
    left_touch  = w_leg_event if (left_cont_nxt > left_cont_cur) else 0.0
    right_touch = w_leg_event if (right_cont_nxt > right_cont_cur) else 0.0

    # b) success event – both legs on pad AND low vertical speed (one‑time)
    w_success = 100.0
    success_cur = (left_cont_cur == 1 and right_cont_cur == 1 and abs(obs[3]) < 0.5)
    success_nxt = (left_cont_nxt == 1 and right_cont_nxt == 1 and abs(vy_nxt) < 0.5)
    success_event = w_success if (success_nxt and not success_cur) else 0.0

    landing_progress = left_touch + right_touch + success_event

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + landing_progress

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'landing_progress': landing_progress   # event‑based now
    }

    return float(total_reward), components
```
