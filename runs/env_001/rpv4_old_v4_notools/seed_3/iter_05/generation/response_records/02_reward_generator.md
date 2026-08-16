# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_body_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left_contact, n_right_contact = next_obs[6], next_obs[7]

    # Distance to target platform (center at 0,0)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    n_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # ---------- 1. Main progress signal: improvement delta ----------
    progress_reward = 2.0 * (dist - n_dist)  # positive when approaching

    # ---------- 2. Soft landing proxy: joint condition average ----------
    # Proximity factor
    prox = 1.0 / (1.0 + 5.0 * n_dist)
    # Contact factor
    contact = (n_left_contact + n_right_contact) / 2.0
    # Angle factor
    angle_factor = max(0.0, 1.0 - abs(n_body_angle) / 0.8)
    # Angular velocity factor
    angvel_factor = max(0.0, 1.0 - abs(n_ang_vel) / 2.0)
    # Speed factor
    speed = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    speed_factor = max(0.0, 1.0 - speed / 2.0)

    landing_proxy = (prox + contact + angle_factor + angvel_factor + speed_factor) / 5.0
    landing_reward = 10.0 * landing_proxy

    # ---------- 3. Crash / boundary constraint: hinge penalties ----------
    # Horizontal drift penalty
    x_violation = max(0.0, abs(nx_pos) - 1.5)
    # Excessive downward velocity penalty
    v_violation = max(0.0, -ny_vel - 3.0)
    crash_penalty = -1.0 * x_violation - 1.0 * v_violation

    total_reward = progress_reward + landing_reward + crash_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "crash_penalty": crash_penalty,
    }
    return total_reward, components
```

# reward_v1 设计说明

## 任务画像与职责选择

- **task_family**: `navigation_goal_reaching`  
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
- **control_type**: discrete

从 `environment_card.md` 的奖励职责拆解出发，v1 版本选择了三个职责（2 个主职责 + 1 个安全约束）：

- **`progress_toward_target`（主学习信号）**：advancement_delta 算子 (`old_distance - new_distance`)，每步提供稠密梯度，指示智能体是否在接近目标平台。  
- **`soft_landing_and_contact`（任务完成近似信号）**：联合条件算术平均（`joint_condition_proxy`），当智能体同时满足接近、双脚接触、小姿态、低速条件时给予较大正奖励，替代缺失的 terminal success signal。  
- **`crash_and_boundary_penalty`（安全约束）**：hinge 算子，对横向越界和过快的下降速度施加连续惩罚，缓解无显式失败标志导致的学习盲区。

## 角色到信号映射与算子选择

| 奖励角色 | 可用信号 | 公式算子 | 理由 |
|----------|----------|----------|------|
| progress_toward_target | `x_position`, `y_position` (相对) | `improvement_delta` (dist - n_dist) | 基于距离的改进比单纯状态值更能避免停滞；距离减少即为正奖励。 |
| soft_landing_and_contact | 速度、角度、角速度、双脚接触、相对位置 | `joint_condition_proxy`（5 因子算术平均） | 无显式 success flag，用多个连续条件构造软着陆近似；算术平均避免乘积塌缩，允许部分信用。 |
| crash_and_boundary_penalty | 水平位置、垂直速度 | `hinge` (`max(0, violation)`) | 只在危险越界时生效，不会在安全范围内惩罚正常飞行。 |

## 排除的职责及原因

- **`fuel_efficiency`**：v1 优先让智能体学会到达和着陆，燃料惩罚可能抑制必要的主引擎使用，留给后续迭代。  
- **`orientation_stability`（单独惩罚）**：已在 landing proxy 中通过角度/角速度因子覆盖，不再重复惩罚，避免过度约束机动。  
- **`velocity_maximization`**：与环境目标“最快但安全”冲突，禁用。  
- **`direct_success_flag_reward`**：环境不存在 `info["success"]` 字段。

## 为什么没有使用 terminal success / failure reward

`environment_card.md` 明确声明 `explicit_success_flag_available = false` 和 `explicit_failure_flag_available = false`，所有终止原因均需从观测推测。因此本设计选择了替代方案：
- **成功**：用 `landing_reward` 作为密集的 soft proxy，避免在 episode 结束时单点发放不可靠的奖励。  
- **失败**：用 `crash_penalty` 的密集 hinge 惩罚提前引导智能体远离危险区域，防止终止后的延迟反馈问题。

## 与之前尝试的关键差异

此前尝试的结构包含 `velocity_penalty`（对速度的直接惩罚）以及 `contact_bonus + orientation_penalty`，可能过度压制了飞行阶段的机动，导致分数极负且无进展。  
本版设计**移除了所有全局速度/姿态惩罚**，改为：
- 通过 `landing_proxy` 将姿态、速度、接触等要求聚合为**正向引导**（奖励而非惩罚）。  
- 仅对**真实的危险状态**（出界、坠毁速度）使用轻量 hinge 惩罚。  
- `progress_reward` 全程鼓励靠近，不受 gate 衰减，确保智能体获得明确的移动信号。

这种“正向着陆引导 + 进步驱动 + 轻量安全提醒”的组合在概念上与之前尝试有本质不同。

## 待后续迭代的职责

- **燃料效率**：在 agent 稳定着陆后加入 `action_efficiency` 轻量惩罚。  
- **更复杂的姿态课程**：接近目标时可动态加强姿态/速度约束，目前由 `landing_proxy` 承担。  
- **防“单脚欺骗”**：如果智能体出现仅一脚接触便静止的静止现象，可在 `landing_proxy` 中提高双脚接触的权重或加入单脚惩罚。

## 训练后应重点观察的 failure modes

- **盘旋不下降**：若 `progress_reward` 对高度的梯度不足，智能体可能长时间停留在高空。观察平均 `y_position` 变化。  
- **高速着陆冲撞**：即便没有 crash 惩罚，若 `landing_reward` 在高速时仍能获得部分接近奖励，智能体可能学会“冲刺 + 接触瞬间得分”。需监控最终 `speed` 分布。  
- **姿态控制失稳**：靠近目标时角速度过大导致翻滚出界。`crash_penalty` 应能覆盖越界情况，但若翻滚不超界可能依然得分。需监控 `body_angle` 极值。  
- **单脚接触即静止**：环境可能因 `body_not_awake` 提前终止，而此时只有一侧支撑。当前 `landing_proxy` 鼓励双脚接触，但若单脚接触 + 低速 + 靠近目标仍能获得部分奖励，可能导致策略停滞。后续可提升双脚接触的权重或改为必要条件。
