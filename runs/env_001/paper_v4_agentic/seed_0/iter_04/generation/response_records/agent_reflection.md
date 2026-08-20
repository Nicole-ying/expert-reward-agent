# Response Record

# 设计理由

## 第 0 步：信号覆盖审计

### 终止模式分析
训练反馈显示 terminated=0/20, truncated=20/20，即 20 条评估 episode **全部因超时而截断**，没有任何 episode 触发成功或失败的终止条件。episode 长度均为 1000 步（最大步数）。这说明 agent 学会了一种**安全徘徊策略**——它既不飞出视口、也不严重碰撞、也不会因静止被丢弃，但也没有完成精确着陆与稳定停靠。

### 观测使用扫描
当前代码中 8 个观测维度全部被使用：
- obs[0], obs[1] (x, y) → `progress_reward` 与所有门控中的距离计算
- obs[2], obs[3] (vx, vy) → `landing_safety_penalty` 的速度惩罚
- obs[4] (angle) → 倾角惩罚
- obs[5] (ang_v) → 角速度惩罚
- obs[6], obs[7] (left/right contact) → `landing_contact_bonus`

未使用的观测维度：无。看似“无信号遗漏”。

### 组件审查与僵尸组件
- `x_boundary_penalty`: active_rate = 0%，从未被触发 → **僵尸组件**，可删除但不作为本次主修改（已无用）。
- `landing_contact_bonus`: active_rate=75.1%，episode_sum_mean=138.2，占 signed_share 98.3% → **实际主信号**，且被严重 exploit。
- `landing_safety_penalty`: active_rate=100%，但 magnitude 极小 (0.92/1000步≈0.0009/步)，对高速/大角度毫无约束力。
- `progress_reward`: 线性距离改善奖励，episode_sum_mean 仅 1.4，被 contact bonus 完全淹没。

### 信号缺口判断
所有观测维度都已使用，但 **landing_contact_bonus 的数学形式和权重使其成为压倒性主信号**，而这一信号并不与“在原点精确着陆停稳”对齐——只要双腿接触任何地面/平台（无论水平位置、速度、角度如何），就能获得持续奖励。于是 agent 学会了在目标附近低空徘徊并用腿蹭地，靠 exploit 该奖励存活 1000 步，却从未触发真正的成功终止。

**结论：信号齐全，但 landing_contact_bonus 的校准严重错误，必须进行结构变换，将其从“接触即奖”改为“精确着陆联合条件奖”。**

## 行为诊断
- **agent 在做什么**：徘徊在低空、双腿频繁接触地面（但位置偏离原点、速度或角度不够低），靠 landing_contact_bonus 获得高分，始终没有成功着陆。
- **需要干预的目标**：landing_contact_bonus，必须把它改造成只奖励“接近成功”的联合条件，切断在中途的 exploit 源头。
- **当前方向评估**：累积记录连续 3 轮分数上升，但实质是 exploit 越来越强，真正的 task solving 能力未建立。若继续沿着“调 contact bonus 系数/门控”的方向修补，可能只是改变 exploit 阈值，而非迫使其完成完整任务。需对 contact bonus 做**结构性重定义**。

## 干预层级：Level 2 结构变换
选择将 `landing_contact_bonus` 从 “接触 × 距离门控” 替换为 **联合条件乘积（joint condition product）**，使得该奖励只在同时满足以下三个连续条件时出现：
- 水平与垂直位置接近原点（距离小）
- 线速度总和很低
- 机体角度接近 0
并且再乘以接触因子。所有因子都使用 **hinge（线性截断）** 形式，使奖励在“较好但不完美”的区域光滑上升，在安全区域仍能给予部分信号，避免完全稀疏。

因子设计：
- 距离因子：`f_prox = max(0, 1 - dist_next / 0.5)` （距离 0.5 为阈值，0 处为 1）
- 速度因子：`f_vel = max(0, 1 - (|nvx|+|nvy|) / 0.5)` （总速度 0.5 以下有分数）
- 角度因子：`f_ang = max(0, 1 - |n_angle| / 0.3)` （0.3 rad ≈17° 为截断）
- 接触因子：`f_contact = (n_left + n_right) / 2.0` （双腿触碰为 1）

乘积 `f_prox * f_vel * f_ang * f_contact` 只有在位置、速度、角度、接触四个条件都接近成功时才显著。系数设为 20.0，使成功步骤的即时奖励足够大（可达 20），引导 agent 迈出最后一步。

这一变换从 “接触二值 × 距离软化” 变为 **多维度联合条件乘积**，既保留了连续梯度，又严格限定了触发区域，从根本上堵塞 exploit 路径。

## 设计校准
- 新奖励的瞬时最大值约 20.0（距离=0, 速度=0, 角度=0, 双腿接触时），远大于 `progress_reward` 的每步 ~0.001–0.002。但由于触发条件严格，episode 中触发步数极少，总 episode sum 预期从 138 大幅下降，不会再次成为压倒性主信号。
- 所有 hinge 阈值设在任务成功条件的 60–80% 附近（距离 0.5 相对于原点要求宽松，速度 0.5 也宽松，角度 0.3 中等），保证在接近成功的状态下 agent 能收到渐增的奖励。
- 无独立惩罚组件，故总负担不成问题。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation variables
    x, y = obs[0], obs[1]
    x_v, y_v = obs[2], obs[3]
    angle = obs[4]
    ang_v = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ---------- 1. Progress reward: moving toward the landing pad (0,0) ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. Horizontal boundary penalty (dead component, kept for backward compat) ----------
    x_limit = 1.2
    x_boundary_penalty = 0.5 * max(0.0, abs(nx) - x_limit)

    # ---------- 3. Landing softness / safety penalty (unchanged) ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.1 * vel_pen + 0.05 * ang_pen + 0.1 * tilt_pen) * gate

    # ---------- 4. Precise landing bonus: product of proximity, velocity, angle, and contact ----------
    # Proximity factor: 1 when distance=0, 0 when distance >= 0.5
    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)
    # Velocity factor: 1 when total speed=0, 0 when >= 0.5
    total_speed = abs(nx_v) + abs(ny_v)
    velocity_factor = max(0.0, 1.0 - total_speed / 0.5)
    # Angle factor: 1 when angle=0, 0 when |angle| >= 0.3
    angle_factor = max(0.0, 1.0 - abs(n_angle) / 0.3)
    # Contact factor: average of both legs
    contact_factor = (n_left + n_right) / 2.0

    precise_landing_bonus = 20.0 * proximity_factor * velocity_factor * angle_factor * contact_factor

    # ---------- Total reward ----------
    total_reward = progress_reward - x_boundary_penalty - landing_safety_penalty + precise_landing_bonus

    components = {
        "progress_reward": float(progress_reward),
        "x_boundary_penalty": float(x_boundary_penalty),
        "landing_safety_penalty": float(landing_safety_penalty),
        "precise_landing_bonus": float(precise_landing_bonus)
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: landing_contact_bonus 成为唯一主信号，被 agent 在非成功区域 exploit；其他组件要么致死要么微弱；需将该奖励重构为只对“接近成功着陆”的状态响应。
- **behavior**: agent 存活满 1000 步，频繁靠双腿接触地面获得高分，但从未触发真正的成功终止（位置分散、速度或角度未达标）。
- **signal**: contact 信号被滥用，亟需与位置、速度、角度联合，只奖励精确着陆前期准备状态。
- **level**: Level 2
- **hypothesis**:  用 proximity、speed、angle、contact 四因子乘积取代原有的接触×距离门控，将奖励范围压缩至成功着陆的邻域内，能够迫使 agent 放弃徘徊策略，学习完成最后精确稳定着陆以获取高额 reward。
- **risk**:  新奖励触发条件严格，可能使得初期探索极难获得该奖励，导致学习停滞或 episode 长度下降；必要时后续可适度放宽阈值或加入 curriculum 温度项。
