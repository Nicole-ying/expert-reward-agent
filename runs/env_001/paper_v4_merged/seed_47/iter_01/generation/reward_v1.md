# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, n_angle, n_ang_vel, n_left, n_right = next_obs

    # ---------- 1. Main progress: improvement in Euclidean distance to landing pad ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = dist - next_dist                # positive when getting closer
    progress_reward = 2.0 * delta_dist

    # ---------- 2. Attitude safety constraint: penalise large body angle and angular velocity ----------
    angle_err = abs(n_angle)
    ang_vel_abs = abs(n_ang_vel)
    attitude_penalty = -0.5 * (angle_err**2 + (0.5 * ang_vel_abs)**2)

    # ---------- 3. Soft success proxy: combination of near‑target, upright, stationary and two‑leg contact ----------
    prox = max(0.0, 1.0 - next_dist / 5.0)           # close to pad centre and ground
    upright = max(0.0, 1.0 - angle_err / 0.5)        # nearly vertical
    speed = (nvx**2 + nvy**2) ** 0.5
    stationary = max(0.0, 1.0 - speed / 1.0)         # low linear velocity
    contact = (n_left + n_right) / 2.0                # 1.0 when both legs touch
    success_proxy = prox * upright * stationary * contact
    success_reward = 5.0 * success_proxy

    # ---------- Aggregate ----------
    total_reward = progress_reward + attitude_penalty + success_reward

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": attitude_penalty,
        "success_reward": success_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`（2D 飞行器精确着陆）。

- **selected reward roles**  
  1. `distance_improvement` (mandatory) – 核心进展信号。  
  2. `attitude_safety_constraint` (conditional) – 轻量惩罚，抑制侧倾和剧烈旋转。  
  3. `soft_success_proxy` (conditional) – 连续多条件组合的任务完成近似信号，弥补缺失的显式 success flag。

- **role‑to‑signal mapping**  
  | role | used signals |
  |------|--------------|
  | distance_improvement | `x_position`, `y_position` |
  | attitude_safety_constraint | `body_angle`, `angular_velocity` |
  | soft_success_proxy | `x_position`, `y_position`, `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity`, `left_support_contact`, `right_support_contact` |

- **formula operators chosen**  
  - `improvement_delta`（3.2）→ 用 `dist - next_dist` 驱动 agent 向目标中心靠近，可直接对抗悬停。  
  - `dense_state_signal` (quadratic penalty)（3.1）→ 姿态误差的二次惩罚，在不终止训练的前提下持续鼓励竖直和低角速度。  
  - `joint_condition_proxy`（3.8）→ 用连续 bounded 因子（proximity、upright、stationary、contact）的乘积构造“着陆完成度”软信号，奖励一旦双腿触地且稳定即给予正反馈，推动 agent 完成最后的下落与接触。

- **excluded roles 及原因**  
  - `terminal_success_reward` / `terminal_failure_penalty`：info 为空，缺少显式标志；观测也没有可靠的 crash 或 episode 结束信号，无法安全使用硬覆盖终止奖励。  
  - `action_efficiency`：离散动作空间（4 动作），v1 阶段优先学会安全着陆，能耗优化留到后续迭代。  
  - `soft_health_gate`：已有姿态惩罚，暂时不需要额外门控；后续若发现 agent “先冲后死”且进度奖励在 crash 回合仍过大，再考虑引入基于 body_angle 的 gate。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**  
  environment_card 明确 `explicit_success_flag_available = false` 且 `info` 为空，无法可靠检测 episode 终止及其原因。强行用观测推断 crash 或成功会在 v1 引入不可靠的信号，故弃用。

- **哪些职责留到后续迭代**  
  - 燃料/动作代价惩罚（action_efficiency）。  
  - crash 或出界的硬终止惩罚（terminal_event）。  
  - 速度门控（接近地面时对垂直速度的附加约束）。  
  - 动态课程或自适应权重。

- **训练后应观察的 failure modes**  
  1. **hover above pad**：progress 信号因距离不再减小而趋近零，agent 在高处悬停不落地 → 说明 contact‑aware 的 success_proxy 不足或 prox 阈值过大。  
  2. **crash 后仍得到正向 episode sum**：agent 快速下降获得大量 progress_reward 然后坠毁 → 后续需引入 soft_health_gate 或 crash 推断惩罚。  
  3. **单腿着地**：contact 因子低于 1 但 success_proxy 仍可能激活（若其他因子高），导致不安全着陆 → 后续可改为 `contact = n_left * n_right`（强制双接触）。  
  4. **成功信号过弱，学习停滞**：若 success_proxy 触发率极低 (<1%)，可增大阈值或改用几何平均缓解乘积塌缩。