# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]           # relative horizontal distance to platform center
    y_pos = next_obs[1]           # relative vertical distance to platform support surface
    x_vel = next_obs[2]           # horizontal velocity
    y_vel = next_obs[3]           # vertical velocity
    body_angle = next_obs[4]      # tilt angle
    angular_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]    # left support leg contact (0 or 1)
    right_contact = next_obs[7]   # right support leg contact (0 or 1)

    # 1. Primary learning signal: continuous negative Euclidean distance to goal
    #    Guides the agent to move towards the landing platform center.
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Stability penalty: light constraint to suppress excessive speed, tilt, and spin
    #    Helps smooth approach and safe landing, but weights are low enough not to freeze exploration.
    w_vx = 0.15
    w_vy = 0.05   # vertical velocity penalty smaller to allow descent
    w_angle = 0.2
    w_angvel = 0.2
    stability_penalty = -(
        w_vx * abs(x_vel) +
        w_vy * abs(y_vel) +
        w_angle * abs(body_angle) +
        w_angvel * abs(angular_vel)
    )

    # 3. Soft landing proxy: multi‑condition bonus that signals near‑perfect landing posture
    #    Only triggers when the agent is close to the platform, has low speed, is nearly upright,
    #    and both legs are in contact. This gives a clear hint for the desired final state.
    prox_thresh = 0.5
    speed_thresh_x = 0.2
    speed_thresh_y = 0.3
    angle_thresh = 0.1
    soft_landing_bonus = 0.0
    if (distance_to_target < prox_thresh and
        abs(x_vel) < speed_thresh_x and
        abs(y_vel) < speed_thresh_y and
        abs(body_angle) < angle_thresh and
        left_contact == 1.0 and right_contact == 1.0):
        soft_landing_bonus = 0.5

    total_reward = distance_reward + stability_penalty + soft_landing_bonus

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **distance_reward**（主学习信号）  
   - 角色：连续负欧几里得距离，引导飞行器靠近目标平台中心。  
   - 数学形态：`- (x² + y²)^0.5`，距离越近奖励越接近 0，远离时负值增大。  
   - 提供每步稠密梯度，是学习逼近和着陆的核心驱动力。

2. **stability_penalty**（轻量稳定约束）  
   - 角色：抑制过大的水平/垂直速度、机身倾斜和角速度，使轨迹平滑、姿态稳定。  
   - 数学形态：对速度、角度、角速度的绝对值加权惩罚，水平速度与姿态权重稍大，垂直速度权重极小（避免阻碍必要下降）。  
   - 属于“方向盘”而非刹车，数值量级明显小于主信号。

3. **soft_landing_bonus**（任务完成近似信号）  
   - 角色：多条件组合的 proxy，在 agent 满足“接近平台 + 低速 + 姿态水平 + 双腿接触”时给予正向激励，向策略传递最终理想状态的特征。  
   - 数学形态：恒为 0，除非所有条件同时成立才给固定小值（0.5）。  
   - 避免 agent 将单一接触信号 exploit 为成功；需同时满足位置、速度、姿态多项，降低误触发风险。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

环境在 `info` 中不提供任何字段，且 `explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false。终止条件 `body_not_awake_or_settled` 无法可靠区分成功与失败，因此不能依赖 terminal 信号。奖励完全由观测驱动的连续和条件信号构成。

## 后续迭代可考虑加入的组件

- **energy_penalty / efficiency_term**：对主发动机（action=2）或姿态发动机施加微小惩罚，优化推力使用。当前 v1 优先确保 agent 能完成可靠着陆，避免过早限制探索。  
- **time_penalty**：若着陆后策略不愿稳定导致步数过长，可加入极小时间惩罚。  
- **gated_reward 或动态阈值**：若 soft_landing_bonus 被 exploit（例如在目标附近反复弹跳刷分），可收紧条件或改为首次触发后固定奖励。  
- **terminal_success_reward**：若 future wrapper 提供显式 success flag，可加入高额终局奖励取代或补充 soft_landing_bonus。

## 训练后应重点观察的 failure mode

- **high_reward_without_success**：虽然距离越来越小、稳定性良好，但未能稳定着陆（双腿未同时接触），导致长时间滞空但不终止。需监控距离与接触信号的同步关系。  
- **fast_crash_near_goal**：靠近目标时速度过大直接撞击地面或导致翻滚，说明 `stability_penalty` 的垂直速度权重可能仍过大，或需要更智能的速度控制片段。  
- **agent afraid to descend**：若垂直速度惩罚过重，agent 可能长期保持高空悬停不下降，应适当降低 `w_vy` 或采用基于加速平滑的方案（后续可改进）。  
- **soft_landing exploit**：若 agent 学会在平台边缘反复点触（一腿接触，另一腿未接触）难以获得 bonus，但偶尔满足条件时可能刷分，需观察触发频率并决定是否加设“连续若干步满足”才给 bonus。