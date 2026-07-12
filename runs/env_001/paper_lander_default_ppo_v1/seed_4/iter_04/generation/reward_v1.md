# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: goal-directed velocity alignment (dense, primary)
    # ------------------------------------------------------------
    pos_x, pos_y = obs[0], obs[1]
    vel_x, vel_y = obs[2], obs[3]
    # Dot product of position and velocity: positive = moving outward,
    # negative = moving toward origin. We reward moving toward origin.
    dot = pos_x * vel_x + pos_y * vel_y
    # Bounded form to keep scale in [-1, 1]
    direction_reward = (-dot) / (1.0 + abs(dot))

    # ------------------------------------------------------------
    # Component B: soft landing proxy (continuous, bounded)
    # ------------------------------------------------------------
    body_angle = obs[4]
    # Contact factor: average contact, encourages both feet
    contact_factor = (obs[6] + obs[7]) / 2.0
    # Angle factor: 1 when upright (angle 0), 0 when |angle| > 0.5 rad
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.5)
    # Speed factor: 1 when near zero speed, 0 when speed^2 > 0.5
    speed2 = vel_x**2 + vel_y**2
    speed_factor = max(0.0, 1.0 - speed2 / 0.5)
    # Joint product proxy for settled landing
    soft_landing_proxy = contact_factor * angle_factor * speed_factor

    # ------------------------------------------------------------
    # Component C: contact event bonus (one-shot settle detection)
    # ------------------------------------------------------------
    event_bonus = 0.0
    # Both feet on ground in next_obs, but not in current obs
    if next_obs[6] == 1.0 and next_obs[7] == 1.0 and (obs[6] == 0.0 or obs[7] == 0.0):
        n_vel_x, n_vel_y = next_obs[2], next_obs[3]
        n_angle = next_obs[4]
        # Require low speed and near-upright to count as a genuine settle event
        if (n_vel_x**2 + n_vel_y**2) < 0.5 and abs(n_angle) < 0.5:
            event_bonus = 10.0

    # ------------------------------------------------------------
    # Component D: attitude stability penalty
    # ------------------------------------------------------------
    ang_vel = obs[5]
    attitude_penalty = -0.1 * (abs(body_angle) + abs(ang_vel))

    # ------------------------------------------------------------
    # Component E: light fuel penalty (mandatory role)
    # ------------------------------------------------------------
    fuel_penalty = -0.01 if action != 0 else 0.0

    # ------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------
    total_reward = (1.0 * direction_reward +
                    0.5 * soft_landing_proxy +
                    event_bonus +
                    attitude_penalty +
                    fuel_penalty)

    components = {
        "direction_reward": direction_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "event_bonus": event_bonus,
        "attitude_penalty": attitude_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`（2D 着陆器）

## selected reward roles（与 environment_card 的 `reward_role_decomposition` 对应）
1. **goal_proximity**（主学习信号） – 使用 `goal-directed velocity alignment` 重写为速度方向对齐奖励，代替常规距离或势能差分。  
2. **safe_landing**（任务完成近似） – 通过两个子组件实现：  
   - 连续 `soft_landing_proxy`（乘积形式，joint condition proxy）  
   - 一次性 `event_bonus`（基于接触状态迁移+低速+低倾角的 settle 检测）  
3. **attitude_stability** – 对 body_angle 和 angular_velocity 的轻量线性惩罚  
4. **fuel_efficiency** – 控制引擎使用的微小惩罚（mandatory role）

## role_to_signal_mapping（与使用的信号对齐）
| role | signals | formula operator |
|---|---|---|
| goal_proximity (directional alignment) | `pos = (obs[0], obs[1])`, `vel = (obs[2], obs[3])` | bounded ratio: `(-dot)/(1+abs(dot))` |
| safe_landing (proxy) | contacts, body_angle, velocities | joint condition proxy (product of bounded factors) |
| safe_landing (event) | contacts (obs & next_obs), next velocities/angle | one-shot condition trigger |
| attitude_stability | `obs[4]`, `obs[5]` | linear penalty |
| fuel_efficiency | `action` | discrete penalty |

## 每个 role 的 formula operator 选择理由
- **方向对齐**：`(-dot)/(1+abs(dot))` 将奖励压缩在 [-1,1]，避免速度幅值支配总奖励，且直接反映“向目标移动”这一核心意图，比距离变化更直接衡量导航效率。  
- **soft_landing_proxy**：三个连续因子的乘积在着陆稳定时产生强信号，同时避免二值触发的稀疏性。  
- **event_bonus**：仅在双脚从非全额接触变为全额接触且速度/姿态都满足阈值时触发，提供一次性 10.0 奖励，明确标记真正的 settle 事件，防止 agent 用“轻触一下就跑”的捷径。  
- **attitude_penalty**：线性惩罚，轻量，避免过度抑制必要的倾转机动。  
- **fuel_penalty**：极小的动作成本（-0.01），在不压制主任务驱动的前提下暗示节约燃料。

## excluded roles 及原因
- **terminal_success_reward / terminal_failure_penalty**：环境未提供显式 success/failure flag，无法使用。  
- **approach_velocity_adaptation**（条件职责）：留到后续迭代；过早引入可能造成非最小相位动态复杂化。  
- **soft_landing_surge_penalty**：留到出现高速触地问题后再加入。  
- **time_pressure**：环境未提供步数或时间信号，不设计。  
- **contact_only_reward**：已被更好的联合条件 proxy 取代。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境卡片明确声明 `explicit_success_flag_available = false` 且 `info` 恒为 `{}`，因此无法获得 episode 级别的成功/失败信号。

## 哪些职责留到后续迭代
- 期望速度随距离自适应调整（approach_velocity_adaptation）  
- 靠近地面时的垂直速度额外惩罚（soft_landing_surge_penalty）  
- 更精细的燃料消耗限制（当前仅为最小动作成本）  
- 任何基于 training_progress 的动态权重课程

## 训练后应观察的 failure modes
- **hovering and wasting fuel**：方向奖励可被原地 hover 获得微小奖励（dot≈0），如果 agent 学成不动而不靠近目标，可能需要增强距离相关项。  
- **crash due to insufficient deceleration**：方向对齐奖励可能鼓励高速冲向目标，soft_landing_proxy 和 event_bonus 需足够强以引导减速；若仍高速触地，需后续加入 surge 惩罚。  
- **foot contact gaming**：agent 可能试图用单脚轻触平台获取部分 proxy 奖励而不彻底 settle；通过要求双脚同时触地的事件阈值和 proxy 中的 contact_factor 可缓解。  
- **inefficient attitude oscillation**：attitude_penalty 若调得过重，可能使策略过度僵硬；若发现不敢使用姿态引擎进行减速/调整，可适当降低权重。