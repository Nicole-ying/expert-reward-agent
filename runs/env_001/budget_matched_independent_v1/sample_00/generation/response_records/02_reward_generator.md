# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander approach‑and‑soft‑contact task.
    Components:
      - goal_proximity_reward   (main learning signal)
      - safe_landing_reward     (task‑completion soft proxy)
      - energy_penalty          (mandatory secondary goal)
      - orientation_penalty     (safety / stability constraint)
    """
    # Extract signals from next_obs (post‑transition state)
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. Goal proximity (dense state signal) ----------
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    w_dist = 1.0
    goal_proximity = -w_dist * distance

    # ---------- 2. Safe landing proxy (joint condition on contact + stability) ----------
    both_contact = float(left_contact > 0.5 and right_contact > 0.5)
    # stability: high reward when velocities and angles are near zero
    vel_sq = x_vel ** 2 + y_vel ** 2
    angle_sq = body_angle ** 2
    omega_sq = angular_vel ** 2
    stability = (1.0 / (1.0 + 10.0 * vel_sq)) * \
                (1.0 / (1.0 + 10.0 * angle_sq)) * \
                (1.0 / (1.0 + 10.0 * omega_sq))
    w_land = 10.0
    safe_landing = w_land * both_contact * stability

    # ---------- 3. Energy penalty ----------
    # penalise any engine activation; action 0 = no engine, 1/2/3 = engine on
    w_energy = 0.02
    if action == 0:
        energy_penalty = 0.0
    else:
        energy_penalty = -w_energy

    # ---------- 4. Orientation penalty ----------
    # light penalty to keep the lander upright
    w_angle = 0.1
    orientation_penalty = -w_angle * (body_angle ** 2)

    # ---------- Assemble total reward ----------
    total_reward = goal_proximity + safe_landing + energy_penalty + orientation_penalty

    components = {
        'goal_proximity': goal_proximity,
        'safe_landing': safe_landing,
        'energy_penalty': energy_penalty,
        'orientation_penalty': orientation_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

### 任务画像
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **控制类型**: 离散 (4 个动作)
- **可用信号**: 8 维观测（相对位置、速度、角度、角速度、双支撑接触）以及动作索引。

### 选中的角色与信号映射
| role | mandatory/conditional | 使用的信号 | 公式算子 |
|---|---|---|---|
| `goal_proximity_reward` | mandatory (主学习信号) | `x_pos, y_pos` | `dense_state_signal` 中的线性惩罚 `‑distance` |
| `safe_landing_reward` | mandatory (任务完成近似) | left/right_contact, x_vel, y_vel, body_angle, angular_vel | `joint_condition_proxy` 的连续乘积形式 |
| `energy_penalty` | mandatory (次目标) | `action` | `discrete_action_penalty` |
| `orientation_penalty` | conditional (稳定/安全) | `body_angle` | `quadratic_penalty` |

### 每个组件的设计思路
1. **goal_proximity**  
   使用下一个状态到原点的欧氏距离，直接给予 `‑w_dist * dist`。距离越远惩罚越大，迫使 agent 不断靠近目标着陆点。这是整个任务的**主驱动力**。

2. **safe_landing**  
   当双侧支撑点均接触着陆垫时（二值条件），乘以一个**连续稳定性因子**。稳定性因子由速度、姿态、角速度的有理函数乘积构成，在静止且竖直时接近 1.0，赋予较大正奖励；状态越差乘积越小。这种设计避免了完全二值稀疏奖励，同时在接触‑非接触步之间提供了可微的 gap（接触时才给奖励）。

3. **energy_penalty**  
   只要发动机点火（action ≠ 0）就施加一个微小的固定惩罚。权重选择极小（0.02），在不压制必要动作的前提下鼓励 agent 尽可能滑行。

4. **orientation_penalty**  
   对整个轨迹施加轻量姿态惩罚，形式为 `‑w_angle * angle²`。这防止 agent 养成大幅偏航的习惯，同时不干扰初始必要的姿态调整（因权重小且二次函数在零附近梯度温和）。

### 排除的角色及原因
- **explicit_success_bonus**：环境中根本没有 `info["success"]`，无法构造真实成功信号。
- **completion_time_penalty**：步数/时间信号不可用，且纯密度距离奖励已隐式鼓励快速接近，无需显式时间惩罚。
- **velocity_smoothing_penalty, angular_damping_penalty**：这两个属于接近目标或接触后的动态约束，需要引入距离阈值或接触状态的复杂门控，留到后续迭代（v2+）。

### 未直接使用 termination‑specific 奖励
环境未提供显式 `success` 或 `failure` 标志，因此无法直接构造 `terminal_success_reward` 或 `terminal_failure_penalty`。唯一可用的成功近似是 `safe_landing` 组件，它利用接触标志间接反映“可能成功”的时刻。

### 留到后续迭代的职责
- `velocity_smoothing_penalty`（根据距离动态缩放的速度惩罚）
- `angular_damping_penalty`（仅在接触或接近时启用）
- 更精细的燃料效率惩罚（区分主引擎与姿态引擎）

### 训练后应观察的故障模式
1. **距离停滞但未坠毁** – 检查 `goal_proximity` 是否被 `orientation_penalty` 或 `energy_penalty` 压制，导致 agent 不敢下降。
2. **高速撞击着陆垫** – 说明 `safe_landing` 的稳定性因子不够强，需提高其对速度的敏感度或前置减速引导。
3. **超出水平视口** – 距离组件已包含水平误差，但如果水平偏移增长过快，可能需要调整权重或引入单独的横向惩罚。
4. **停在目标外但触发休眠** – 若 agent 在接触垫边缘以外停下且姿态稳定，将得不到 `safe_landing` 奖励，训练应自行修正；若大量出现，需考虑加大横向引导或严格接触条件。
