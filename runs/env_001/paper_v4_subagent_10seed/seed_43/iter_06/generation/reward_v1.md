# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    # left_c = obs[6], right_c = obs[7] not used for current state

    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Soft landing proxy reward (main learning signal)
    landing_reward = 0.0
    if next_left > 0.5 and next_right > 0.5:
        # Position factor: prefer x close to 0
        pos_factor = 2.718281828 ** (-(next_x ** 2) / (2 * 0.0025))  # sigma = 0.05
        # Speed factor: penalise high total speed
        speed_n = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
        spd_factor = 2.718281828 ** (-(speed_n ** 2) / (2 * 0.04))   # sigma = 0.2
        # Attitude factor: prefer upright
        ang_n = abs(next_angle)
        ang_factor = 2.718281828 ** (-(ang_n ** 2) / (2 * 0.01))     # sigma = 0.1
        landing_reward = 10.0 * pos_factor * spd_factor * ang_factor

    # 2. Progress reward: reduction in distance to target (auxiliary)
    dist_now = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist_now - dist_next

    # Safety gate for progress: when near target, suppress reward if speed/angle are high
    near_target = dist_now < 0.5
    gate = 1.0
    if near_target:
        # Use current vertical speed and body angle to form a soft gate
        gate = 1.0 / (1.0 + 10.0 * (y_vel ** 2) + 5.0 * (angle ** 2))
    progress_reward = delta_dist * gate

    # 3. Action efficiency penalty (very small)
    action_cost = -0.01 if action != 0 else 0.0

    # 4. Boundary penalty: discourage moving outside viewport horizontally
    boundary_penalty = 0.0
    if abs(x_pos) > 1.0:
        boundary_penalty = -5.0 * (abs(x_pos) - 1.0)

    # 5. Light angle/angular-velocity penalty (global, to stabilise attitude)
    angle_penalty = -0.01 * abs(angle) - 0.001 * abs(ang_vel)

    total_reward = (landing_reward + progress_reward +
                    action_cost + boundary_penalty + angle_penalty)

    components = {
        "landing_soft_reward": landing_reward,
        "progress": progress_reward,
        "action_cost": action_cost,
        "boundary_penalty": boundary_penalty,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 核心设计假设（与之前失败方案的本质区别）

之前的尝试大多使用 `progress`（距离改进）+ `soft_landing_penalty` 的组合，或者稀疏的 `contact_success_reward`。结果最佳分数仅为 -18.8，说明这些结构未能让智能体学会安全着陆。我的假设是：

- **稀疏或二值的着陆奖励**无法在早期提供足够学习信号；  
- **纯距离驱动**会使智能体冲向目标，即使配合减速惩罚也可能因惩罚时机/权重不佳导致撞击或崩坏；  
- **缺少一个连续、可梯度的“软着陆状态奖励”**，导致智能体没有明确的动机去同时满足双接触、低速度、小倾角。

本设计引入一个全新的**软着陆代理奖励（soft landing proxy）**作为**主学习信号**，并在接近阶段用**安全门控**抑制距离奖励的过快接近，从而引导智能体在靠近目标时主动减速、摆正姿态并轻柔接触。

## 选定的奖励角色与职责映射

| 角色 | 使用信号 | 公式算子 | 说明 |
|------|----------|----------|------|
| `soft_landing_proxy` (main) | `next_obs[0,2,3,4,6,7]` | joint_condition_proxy (高斯核乘积) | 当双支撑腿均接触时，根据位置偏差、合速度、倾角计算出连续奖励值。最大 10 分，远离理想状态时平滑衰减。提供每步可用的梯度，取代稀疏成功信号。 |
| `delta_distance_to_target` (auxiliary) | `obs[0:2], next_obs[0:2]` | improvement_delta | 距离减少量作为辅助驱动，避免智能体停滞。权重仅 1.0，不会压倒着陆奖励。 |
| `safety_gate` (condition modifier) | `obs[3], obs[4], dist` | soft_health_gate (倒数门) | 当 `dist<0.5` 时，用当前垂直速度和倾角抑制 `progress` 奖励，迫使智能体在接近时减速、摆正。门控形式为 `1/(1+10*vy²+5*θ²)`。 |
| `action_efficiency` | `action` | 常量惩罚 | 使用引擎时 -0.01，轻微鼓励节省燃料。 |
| `boundary_penalty` | `obs[0]` | hinge 惩罚 | `abs(x)>1.0` 时给予 `-5*(|x|-1.0)`，预先遏制飞出视口的倾向。 |
| `angle_penalty` | `obs[4], obs[5]` | 二次惩罚 | 全局微小的姿态和角速度惩罚，辅助稳定。 |

## 排除的角色及原因

- **`terminal_touchdown_bonus`**：环境未提供终止标志，且 `info` 为空，无法可靠实现。  
- **`dense_speed_penalty_global`**：与“尽快到达”目标冲突，会抑制正常接近。改为仅在门控内局部生效。  
- **`approach_speed_bonus_with_safety_gate`**：原版角色依赖朝向目标的点积，但本任务速度方向可直接从 `obs[2:4]` 获得；当前设计中 `progress` 奖励已经鼓励接近，加上门控后无需额外速度奖励。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`。  
- `info` 返回空字典，无法获取任何终止原因。  
- 因此所有信号均以稠密奖励形式嵌入每一步，不依赖稀疏终止事件。

## 后续迭代预留的职责

- **更精细的下降速度管理**：如果出现悬停不降，可在 `dist<0.3` 时增加一个带限幅的下降鼓励项（`-min(y_vel, 0) * small_weight`）。  
- **动态权重或课程**：可在训练后期逐步降低 `progress` 权重，强化 `soft_landing` 作为唯一目标。  
- **接触瞬间专门奖励**：若能够从外部包装器识别 episode 结束时的状态，可添加一次性成功加分。

## 训练后应观察的 failure modes

| 失败模式 | 证据 | 应对思路 |
|---|---|---|
| 悬停在目标上方不下降 | `y_pos` 稳定小值，`vy≈0`，接触为 0，步数耗尽 | 当前门控只在 `dist<0.5` 起作用，但减速后可能停止下降。后续可加入轻微下降奖励。 |
| 降落后弹跳或翻倒 | 接触标志交替闪烁，`angle` 或 `angular_velocity` 突变 | 若出现，可增大 `landing_soft_reward` 中速度/倾角的高斯宽度（减小 sigma）或加强 `angle_penalty`。 |
| 过度使用姿态引擎失控 | 角速度持续高，动作 1/3 频繁，距离下降慢 | 当前 `angle_penalty` 较轻；可增大 `angular_velocity` 惩罚或加入姿态引擎单独惩罚。 |
| 直接侧向飞出视口 | `abs(x)` 快速增大超越边界 | `boundary_penalty` 已设计；若无效，可增大其系数或采用指数形式。 |
| 高速撞击目标垫 | 接触时 `speed_n` 极大，`landing_soft_reward` 近乎零 | 门控会抑制接近奖励，但撞击可能仍发生；`soft_landing` 指数惩罚会迅速压低总分，促使下一次更慢。 |