# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Potential = -(x^2 + y^2), reward = potential_curr - potential_next
    #   Positive when distance to pad decreases, zero when stationary.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    #   Penalise speed more strongly when near the pad.
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)          # alpha = 2.0
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Keep body upright and angular velocity small.
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (joint_condition_proxy)
    #   Give positive reward when both legs touch the pad,
    #   vehicle is at pad height, and vertical velocity is low.
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    # y position close to zero (pad height)
    proximity_to_pad = max(0.0, 1.0 - abs(next_obs[1]) / 0.2)   # threshold_y = 0.2
    # vertical velocity small
    soft_vel = max(0.0, 1.0 - abs(next_obs[3]) / 0.2)           # threshold_vy = 0.2
    safe_contact_reward = 0.5 * both_contacts * proximity_to_pad * soft_vel

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return total_reward, components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- `navigation_goal_reaching` / `goal_approach_and_soft_contact`

## selected reward roles
1. **goal_proximity** – 主学习信号，使用 `improvement_delta`（距离²的减少量）驱动 agent 向目标平台中心移动。
2. **velocity_damping** – 速度约束，使用 `proximity_gated quadratic_penalty`：靠近 pad 时加重速度惩罚，远离时放松，实现软着陆。
3. **orientation_stability** – 姿态稳定约束，使用 `quadratic_penalty` 同时抑制机体倾角和角速度。
4. **safe_contact** – 任务完成近似信号，使用 `joint_condition_proxy`：双脚接触 + 高度接近 pad + 垂向速度小时给予正奖励，鼓励稳定静止。

## role_to_signal_mapping
| role | signals used | formula operator |
|---|---|---|
| `goal_proximity` | `obs[0], obs[1]` (x,y) vs `next_obs[0], next_obs[1]` | `improvement_delta` on potential = -(x²+y²) |
| `velocity_damping` | `next_obs[2], next_obs[3]` (vx,vy), distance next | `proximity_gated quadratic_penalty` |
| `orientation_stability` | `next_obs[4], next_obs[5]` (angle, ω) | `quadratic_penalty` |
| `safe_contact` | `next_obs[6], next_obs[7]` (contacts), `next_obs[1]` (y), `next_obs[3]` (vy) | `joint_condition_proxy` (product form) |

## excluded roles 及原因
- **fuel_efficiency** – 动作效率惩罚在 v1 会压制早期探索，留待 agent 先学会安全着陆后再加入。  
- **fast_progress_encouragement** – 可能加剧高速冲撞风险，当前由 progress 主信号和速度阻尼即可提供基本时间驱动力。  
- **boundary_penalty** – 环境通过终止条件处理越界，无可用边界坐标，额外惩罚易干扰学习。  
- **explicit_success_reward / terminal_failure_penalty** – 环境无显式成功/失败标志，info 为空，无法安全使用。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
因为 `explicit_success_flag_available=false`、`explicit_failure_flag_available=false` 且 `info` 为空字典，无法可靠推断终止原因。改用连续近似信号 `safe_contact_reward` 提供软着陆的正反馈。

## 哪些职责留到后续迭代
- **fuel_efficiency**：待 agent 稳定完成着陆后加入轻微动作代价。  
- **fast_progress**：必要时以 shaping 形式微调早期下降速度。  
- **更精确的边界/坠落惩罚**：当观测支持时可用 hinge 或 soft gate。

## 训练后应该观察哪些 failure modes
- 高速砸地 → 检查 episode 长度、crash 率，若高则加大 `velocity_damping` 权重或 alpha。  
- 飘出水平边界 → 若无额外惩罚则依赖姿态稳定和速度阻尼，观察越界终止频率。  
- 长时间悬停不下降 → `progress_reward` 的 potential 差会小，观察 episode 距离是否持续减小；必要时微调 progress 权重。  
- 单脚着陆倾斜失败 → 观察 `body_angle` 和接触模式，可提高 `safe_contact` 的条件或后续增加双接触奖励。  
- 姿态剧烈振荡 → 提升 `orientation_stability` 中的 `w_ang` 或 `w_angvel`。  
- 反复弹跳 → 当在 pad 上且速度较大时 `safe_contact_reward` 小，会自然抑制，但可考虑增大 `velocity_damping` 的门控强度。