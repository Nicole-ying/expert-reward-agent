# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables from next_obs (post-action state)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Proximity reward (main learning signal)
    # Drive the agent toward the target platform (x=0, y=0)
    dist_sq = x * x + y * y
    proximity_reward = -0.5 * dist_sq

    # 2. Velocity penalty (soft landing constraint)
    # Penalise excessive horizontal and vertical velocity
    v_thresh = 0.2
    v_penalty = 0.0
    if abs(vx) > v_thresh:
        v_penalty += abs(vx) - v_thresh
    if abs(vy) > v_thresh:
        v_penalty += abs(vy) - v_thresh
    velocity_penalty = -1.0 * v_penalty

    # 3. Body angle penalty (stability constraint)
    # Penalise tilting away from upright
    angle_thresh = 0.1  # radians
    if abs(body_angle) > angle_thresh:
        angle_penalty = -1.0 * (abs(body_angle) - angle_thresh)
    else:
        angle_penalty = 0.0

    # 4. Landing bonus (task completion proxy)
    # Strong bonus when both legs touch with low speed, angle, and angular velocity
    k_v = 5.0
    k_angle = 10.0
    k_angvel = 5.0

    vx_factor = 1.0 / (1.0 + k_v * abs(vx))
    vy_factor = 1.0 / (1.0 + k_v * abs(vy))
    angle_factor = 1.0 / (1.0 + k_angle * abs(body_angle))
    angvel_factor = 1.0 / (1.0 + k_angvel * abs(angvel))
    contact_factor = left_contact * right_contact  # only if both feet touch

    landing_bonus = 20.0 * contact_factor * vx_factor * vy_factor * angle_factor * angvel_factor

    total = proximity_reward + velocity_penalty + angle_penalty + landing_bonus
    components = {
        "proximity_reward": proximity_reward,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "landing_bonus": landing_bonus
    }
    return total, components
```

# reward_v1 设计说明

**任务家族与动力学子类型**  
- `task_family`: `navigation_goal_reaching`  
- `dynamics_subtype`: `goal_approach_and_soft_contact` (带支脚的垂直起降飞行器，离散推力控制)

**选中的奖励职责 (selected reward roles)**  
1. **proximity_to_target** (mandatory) – 核心驱动力，引导飞行器向目标平台中心移动。  
2. **soft_landing_conditions** (mandatory) – 通过速度和姿态约束确保低速、直立着陆，避免撞击和侧翻。  
3. **terminal_landing_bonus** (conditional) – 当双支脚平稳接触且状态良好时给予强正向信号，作为隐式成功完成的近似。

**职责‑信号映射**  
| 职责 | 信号来源 (next_obs) | 使用的公式算子 |
|------|---------------------|----------------|
| proximity_to_target | `x_position`, `y_position` | `dense_state_signal` (二次惩罚: `-0.5 * distance²`) |
| soft_landing (速度) | `x_velocity`, `y_velocity` | `dense_state_signal` 的 hinge 变体: `-1.0 * max(0, |v| - threshold)` |
| soft_landing (姿态) | `body_angle` | 同上 hinge: `-1.0 * max(0, |angle| - threshold)` |
| terminal_landing_bonus | `left_contact`, `right_contact`, `vx`, `vy`, `angle`, `angvel` | `joint_condition_proxy` (因子乘积) + `bounded_signal` (`1/(1+k*|error|)`) |

**排除的职责及原因**  
- `terminal_success_reward` / `terminal_failure_penalty`: 环境 `info` 为空，无显式成功/失败标志，无法安全实现。  
- `fuel_efficiency`: 离散动作的效率惩罚在 v1 暂不加入，避免阻碍飞行器学习必要的上升和姿态控制；留待后续迭代。  
- `time_bonus_or_penalty` (avoid role): 与安全着陆冲突，禁用。  
- `exact_position_shape_reward`: 无子目标分段需求，禁用。

**为什么未使用 terminal_success/failure 奖励**  
因为 `explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false，`info` 字典不提供任何终止原因。强行推断会引入不可靠信号，因此 v1 完全依赖连续状态信号驱动学习，并用 `landing_bonus` 作为软完成信号。

**推迟到后续迭代的职责**  
- 动作代价/燃料惩罚 (`fuel_efficiency`)：待着陆行为稳定后加入轻量惩罚以优化能耗。  
- 基于观测推断的 terminal failure penalty：如果能可靠地从 `obs` 推断 crash 或飞出边界，可加入硬惩罚。  
- `soft_health_gate`：若观察到 agent “高速冲向目标后 crash” 的模式，可以在速度/角度恶化时逐渐衰减主奖励，而不是简单 hinge。

**训练后应重点观察的 failure modes**  
- **悬停不降落**：`y_position` 长期远离零，末端未触发着陆奖励；需检查 `proximity` 权重是否过大或 `landing_bonus` 吸引力不足。  
- **高速撞击**：`vy` 过大且终止时未双接触；可通过增大速度 hinge 阈值斜率或在后续迭代引入速度上限惩罚缓解。  
- **着陆倾斜侧翻**：`body_angle` 大且仅单脚接触；可能需要加大 `angle_penalty` 系数或结合 `angular_velocity` 惩罚。  
- **利用着陆奖励刷分**（理论上较难）：若 agent 重复接触又脱离，`landing_bonus` 乘积会剧烈波动，但不会持续获得高分；若出现，可降低 `w_terminal` 或加入腿接触的持续性要求。
