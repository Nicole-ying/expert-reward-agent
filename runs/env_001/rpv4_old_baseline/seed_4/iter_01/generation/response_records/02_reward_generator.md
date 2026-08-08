# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations for current and next state
    # obs indices:
    # 0: x_position, 1: y_position, 2: x_velocity, 3: y_velocity,
    # 4: body_angle, 5: angular_velocity, 6: left_contact, 7: right_contact
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Compute distances from target (0,0)
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # Weights (heuristically reasonable starting points for this environment)
    w_progress = 1.0           # scale for distance improvement
    w_speed   = 0.5            # penalty for residual kinetic energy
    w_angle   = 0.5            # penalty for tilt
    w_angvel  = 0.5            # penalty for rotation
    w_contact = 10.0           # terminal soft‑landing incentive
    alpha     = 1.0            # sharpness of vertical‑speed gate
    beta      = 1.0            # sharpness of tilt gate

    # 1. Main progress signal: approach + velocity damping
    #    Use improvement_delta for distance (reward getting closer)
    #    and penalize high speed in next state.
    progress_reward = w_progress * (dist_curr - dist_next) - w_speed * speed_next

    # 2. Orientation stabilization (safety constraint)
    orientation_penalty = -w_angle * (angle_next ** 2) - w_angvel * (angvel_next ** 2)

    # 3. Soft‑landing contact proxy (approximate task completion)
    both_legs_on_platform = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    # Use exponential decay to favour landing with low vertical speed and near‑vertical attitude.
    # exp(-alpha*|vy|) is computed as e ** (‑alpha * abs(vy))
    # exp(‑beta * angle²) is computed similarly.
    smooth_vy_gate = 2.718281828 ** (-alpha * abs(vy_next))
    smooth_angle_gate = 2.718281828 ** (-beta * (angle_next ** 2))
    soft_contact_reward = w_contact * both_legs_on_platform * smooth_vy_gate * smooth_angle_gate

    total_reward = progress_reward + orientation_penalty + soft_contact_reward

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_contact_reward": soft_contact_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像
- **task_family**: `navigation_goal_reaching`  
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
- **控制类型**: 离散动作（主发动机 + 左右姿态发动机）  
- **主要目标**: 精准、节能地软着陆在中央平台上（位置≈0、速度≈0、姿态竖直、双腿同时接触）  
- **次要目标**: 节省燃料（本版本暂且忽略，留到后续迭代）

## 选中的奖励职责与组件对应
| 职责角色 | 对应组件 | 使用的公式算子 | 关键信号 |
|----------|----------|----------------|----------|
| `goal_proximity_and_settling` (主学习信号) | `progress_reward` | `improvement_delta` (距离) + `dense_state_signal` 惩罚 (速度) | `next_obs[0,1,2,3]` |
| `orientation_stabilization` (稳定/安全约束) | `orientation_penalty` | `quadratic_penalty` | `next_obs[4,5]` |
| `soft_landing_and_contact` (任务完成近似信号) | `soft_contact_reward` | `joint_condition_proxy` (乘积形式, 用指数衰减门控) | `next_obs[6,7]` (双腿接触), `next_obs[3,4]` (vy, angle) |

## 公式细节
- **progress_reward**：`w_progress * (dist_prev - dist_next) - w_speed * speed_next`  
  鼓励每步向目标靠近，同时抑制高速移动，让 agent 学会渐进减速而不是冲刺。`improvement_delta` 天然包含了“远离即罚”的语义。
- **orientation_penalty**：`-w_angle * angle² - w_angvel * (angvel)²`  
  持续压制大幅度倾斜和旋转，避免侧翻等灾难性失败。
- **soft_contact_reward**：`w_contact * both_legs * exp(-α·|vy|) * exp(-β·angle²)`  
  仅在双腿都接触平台时激活，且速度越小、姿态越竖直时奖励越高。使用指数衰减避免过早、过猛的正奖励，促使 agent 学习真正的软着陆。

## 未使用的职责及原因
- **explicit_success_bonus / crash_penalty_from_termination** – `info` 为空，不存在可用的完成或失败标志；观测也无法可靠推断精确的终止原因。因此完全排除硬性终端事件信号。
- **thrust_penalty** – 属于条件职责（燃料效率）。v1 阶段优先让 agent 学会安全着陆，暂时不加入推力成本，避免抑制必要的发动机使用。可在后续迭代中以微小权重逐步引入。
- **survival_bonus_or_time_penalty** – 容易诱发 agent 过早终止或悬停 “拖延”，在完成信号尚未牢固时不使用。

## 没有使用 terminal_success_reward / terminal_failure_penalty 的原因
环境未提供任何与终止结果直接相关的观测或 info 键（`explicit_success_flag_available = false`，`explicit_failure_flag_available = false`）。强行从观测猜测成功或失败（例如根据双腿接触和低速度判断）可能将“硬着陆后昏迷但仍满足接触”误判为成功。因此改用连续、可微的 soft contact 奖励代替离散终端事件。

## 后续迭代建议
- **燃料效率**：一旦基本着陆行为稳定，可加入极小权重的 `thrust_penalty`（例如 -0.001 per non-zero action）。
- **多条件接触门控**：若观察到 agent 在低高度反复微调而不着陆，可考虑加入 y 接近零的显式门控，让接触奖励更紧密地与目标区域耦合。
- **动态权重或课程**：如果训练早期坠毁率过高，可临时降低 speed 惩罚权重，待 agent 学会接近平台后再逐步提高。

## 训练后应重点观察的失败模式
| 失败模式 | 症状 | 可能的干预方向 |
|----------|------|----------------|
| 高速垂直撞击 | 终止时 vy 很大，双腿接触但倾角大 | 增大 `w_speed` 或在接触附近加强速度门控 |
| 始终悬停高空 | y 远离 0，主发动机几乎不用 | 提高 `w_progress` 或为高度远离零的状态加入额外的负向引导 |
| 过度校正导致侧翻 | 频繁触发大角度、大角速度 | 适当提高 `w_angle` 或引入角速度变化惩罚 |
| 只碰一条腿 | 左右接触不均衡 | 可考虑在接触奖励中加入左右对称性要求（后续迭代） |
