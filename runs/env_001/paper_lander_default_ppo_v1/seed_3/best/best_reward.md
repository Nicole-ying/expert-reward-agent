# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next state signals
    x = next_obs[0]
    y = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to target platform center (goal is (0,0))
    distance = (x**2 + y**2) ** 0.5

    # ---------- main learning signal: goal proximity ----------
    w_prox = 2.0
    # Bounded positive reward: maximum 2.0 at distance 0, decays smoothly
    prox_reward = w_prox / (1.0 + distance)

    # ---------- near‑factor for soft‑landing constraints ----------
    near = 1.0 / (1.0 + distance)   # 1 when close, ~0 when far

    # ---------- velocity penalty ----------
    w_vel = 0.1
    vel_penalty = -w_vel * near * (x_vel**2 + y_vel**2)

    # ---------- body angle penalty ----------
    w_angle = 0.2
    angle_penalty = -w_angle * near * (angle**2)

    # ---------- angular velocity damping ----------
    w_angvel = 0.1
    angvel_penalty = -w_angvel * near * (ang_vel**2)

    # ---------- contact bonus (both feet on ground) ----------
    w_contact = 1.0
    both_contacts = left_contact * right_contact   # 0 or 1
    contact_reward = w_contact * near * both_contacts

    # ---------- total reward ----------
    total_reward = prox_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        "proximity_reward": prox_reward,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angvel_penalty,
        "contact_bonus": contact_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
  飞行器必须在二维空间中接近目标平台、减速并实现双脚同时接触的软着陆。

## selected reward roles

| role | purpose | formula operator(s) | signals used |
|------|---------|---------------------|--------------|
| `goal_proximity_and_arrival` (mandatory) | 推动飞行器向平台中心 (0,0) 移动 | `bounded_signal`：`w / (1 + distance)` | `x_position` (obs[0]), `y_position` (obs[1]) |
| `soft_landing_and_stabilization` (mandatory) | 靠近目标时强制减速、水平姿态与双脚接触 | `quadratic_penalty` × `near` gate、`soft_health_gate` 风格的门控正奖励 | `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity`, `left_contact`, `right_contact` (obs[2:8]) |
| `energy_efficiency` (conditional) | **暂未引入**（留到后续迭代） | – | `action` |

## role_to_signal_mapping 说明
- `goal_proximity_and_arrival` → 使用距离 `(x^2 + y^2)^0.5`，通过 `2.0 / (1 + distance)` 形成**连续正奖励**，最大值为 2（在目标处），远处衰减平滑，避免远距离数值过大。
- `soft_landing_and_stabilization` → 利用 **门控因子 `near = 1/(1+distance)`**，将速度惩罚、姿态角惩罚、角速度阻尼和双脚接触奖励都限制在近目标区域，防止早期学习阶段被过度约束。
  - 速度惩罚：`-0.1 * near * (x_vel^2 + y_vel^2)`  
  - 倾角惩罚：`-0.2 * near * (angle^2)`  
  - 角速度阻尼：`-0.1 * near * (ang_vel^2)`  
  - 接触奖励：`+1.0 * near * (left * right)`（仅在双脚同时接地时给予）
- 因为 `info` 恒为空且无 `done` 参数，**没有实现 terminal_settlement_bonus**，完全用逐步信号替代。

## excluded roles 及原因

| excluded role | 原因 |
|---------------|------|
| `terminal_settlement_bonus` | 需要 `done` 标志或 `info.success`，但在当前 `compute_reward` 接口中不可用，强行猜测会引入错误奖励。 |
| `energy_efficiency` (v1) | v1 阶段优先学习安全着陆任务，过早加入动作成本可能抑制必要推力使用，留到后续迭代。 |
| `angular_velocity_penalty_direct` | 单独角速度惩罚不考虑绝对倾角，可能误罚旋转但不罚静倾；当前已将角速度作为阻尼项与倾角配合，更安全。 |
| `pure_time_penalty` | 环境无最大步数截断，每步负奖励可能诱导策略通过失败终止来逃避惩罚，安全性下降。 |
| `alive_bonus` | 容易产生悬停或拖延行为，与“尽快着陆”目标冲突。 |

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available = false`（`info` 为空）
- `explicit_failure_flag_available = false`
- 这两种信号不可用，因此 reward 完全基于每步的 `next_obs` 构建稠密或条件化奖励。

## 哪些职责留到后续迭代
- **动作效率**（`energy_efficiency`）：当策略已稳定着陆后，可加入小权重动作使用惩罚，鼓励减少不必要的发动机点火。
- **双脚接触门控加强**：未来可引入双脚接触作为任务完成前的更精细条件（例如 `joint_condition_proxy`，但需要小心的门控避免单脚悬停 exploit）。
- **复杂课程/动态权重**：当前任务在 v1 阶段无明显需要，后期可考虑 `curriculum_weighting` 调整接近奖励与稳定惩罚的比例。

## 训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击平台导致 crash（未减速但位置靠近） | 终止前 `y` 已接近 0 但 `y_vel` 负值很大，双脚未同时接触 | 增加 `near` 门控内的速度惩罚权重，或针对垂直速度添加额外约束 | 
| 长时间无推力（自由落体失败或不敢点火） | 动作长时间为 0，距离减少依赖重力，速度失控 | 检查接近奖励是否足够，确认 `energy_efficiency` 未过早压制动作 |
| 过度使用主引擎导致飞出视口 | 水平位置超出边界，大量 `action=2` | 未来可通过位置安全边界惩罚或限制主引擎使用频率，v1 可先观察 |
| 姿态振荡、无法稳定接触 | `body_angle` 和 `ang_vel` 高频摆动，双脚接触短暂 | 适当提高角度/角速度惩罚权重，或增加平滑处理 |
| 仅单脚接触即停止 | 终止时左右接触不同时为 1 | 提升 `contact_bonus` 权重或在接近时进一步惩罚单脚悬空（v2 可加入） |
| 悬停于目标附近但永不接地 | `y` 接近 0 但 `y_vel` 未归零，双脚未触 | 强化接触奖励或后期加入时间衰减因子推动着陆决策 |