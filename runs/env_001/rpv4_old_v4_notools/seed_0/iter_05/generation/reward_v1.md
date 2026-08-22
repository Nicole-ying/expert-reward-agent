# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0
    c_angle = 2.0      # health gate sensitivity for body angle
    c_vel = 1.0        # health gate sensitivity for total speed
    w_land = 5.0
    angle_thresh = 0.5   # rad, quality for landing
    vel_thresh = 1.0     # m/s (or units), total speed quality for landing
    w_action = 0.01

    # -------------------------------
    # 1. Goal‑reaching progress (improvement_delta)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_unscaled = dist_cur - dist_next  # positive when getting closer

    # -------------------------------
    # 2. Soft health gate (on current body state)
    # -------------------------------
    # Use body angle and total speed to scale progress
    health_penalty = c_angle * abs(obs[4]) + c_vel * (abs(obs[2]) + abs(obs[3]))
    health_gate = 1.0 / (1.0 + health_penalty)

    progress_reward = w_progress * progress_unscaled * health_gate

    # -------------------------------
    # 3. Landing quality (joint condition proxy)
    # -------------------------------
    both_legs_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    if both_legs_contact > 0.0:
        angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / angle_thresh)
        speed_total = abs(next_obs[2]) + abs(next_obs[3])
        speed_factor = max(0.0, 1.0 - speed_total / vel_thresh)
        landing_quality = angle_factor * speed_factor
    else:
        landing_quality = 0.0

    landing_reward = w_land * landing_quality

    # -------------------------------
    # 4. Action sparsity
    # -------------------------------
    action_penalty = -w_action * (1.0 if action != 0 else 0.0)

    # -------------------------------
    # Total reward
    # -------------------------------
    total_reward = progress_reward + landing_reward + action_penalty
    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与职责选择

- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **selected reward roles**:
  - `goal_reaching_progress` (mandatory)
  - `soft_landing_on_both_legs` (mandatory, 通过条件组合实现)
  - `action_sparsity` (mandatory)
  - `orientation_stabilization_during_flight` (通过健康门控间接实现)

## Role‑to‑Signal Mapping 与 Formula Operators

| Role | Signals | Formula Operator | 说明 |
|------|---------|------------------|------|
| goal_reaching_progress | `obs[0], obs[1], next_obs[0], next_obs[1]` | `improvement_delta` (距离减小量) 乘以 `soft_health_gate` | 鼓励向目标原点接近，但在身体状态恶化时衰减，防止“先冲后死” |
| soft_landing_on_both_legs | `next_obs[6], next_obs[7], next_obs[4], next_obs[2], next_obs[3]` | `joint_condition_proxy` (乘积组合连续条件) | 仅在双腿触地时启用，着陆质量由小角度、低总速度共同决定 |
| action_sparsity | `action` | 离散动作固定惩罚 (`step_cost`) | 每步对非零动作施加极小负奖励，鼓励滑翔 |
| soft_health_gate (辅助) | `obs[4], obs[2], obs[3]` | 倒数门 `1/(1+penalty)` | 用当前机体角度和速度构造衰减因子，乘在主进度信号上，作为隐式稳定约束 |

## Excluded Roles 及原因

- `terminal_success_bonus`：环境 `info` 为空，无法获得 episode 终止与成功标志；即使基于观测猜测成功也无法可靠触发，排除。
- `binary_crash_penalty`：没有可靠的 crash 信号，且硬负惩罚容易在未真实 crash 时误伤学习，改用软着陆引导和健康门控替代。
- `time_penalty`：无时间计数器信号，显式时间惩罚不可行；动作稀疏惩罚间接促进快速着陆。

## 为何未使用 terminal_success_reward / terminal_failure_penalty

环境卡片明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，`info` 为空字典。根据设计原则，没有可用标志时不可写入此类组件。

## 与先前失败结构的关键区别

- **不再使用独立的 orientation_penalty 和 vel_damping**，而是通过 `soft_health_gate` 乘到进度奖励上，使姿态与速度惩罚不再抑制整个奖励空间，而是在不安全时衰减任务驱动力，迫使 agent 在保持稳定的前提下追求目标。
- **着陆奖励不再只是 contact_reward**，而是基于双腿接触 + 低速小角度的连续质量因子，提供“如何着陆”的梯度，而不是简单的接触计数。
- **核心进展信号使用 delta distance 而非 fixed rate**，配合门控使 agent 无法通过单步高速俯冲长时间获得大奖励。

## 后续迭代可加入的职责

- **空中角速度硬惩罚**：若门控仍不足以消除高频振荡，可加入未接触时的角速度二次惩罚。
- **动态课程门控**：根据 `training_progress` 逐渐收紧健康门控阈值，使早期探索更宽松。
- **效率项权重自适应**：若动作惩罚阻碍必要机动，可引入基于剩余距离的 action cost scaling。

## 训练后应观察的 failure modes

- **高速坠毁**：episode 过短，最后一帧速度/角度极大。若出现，增大 `c_vel` 或 `c_angle`，或提高 `w_land`。
- **悬停盘旋**：大量使用姿态引擎且高度不降，总奖励长期为负。可适度提高 `w_progress` 或降低 `w_action`。
- **单腿翻倒**：终止时仅一腿触地、角度大。强化着陆奖励中的双腿乘积因子，或引入仅接触惩罚。
- **动作振荡**：action 在 1/3 之间频繁切换。增大 `w_action` 或后续加入动作变化平滑项。