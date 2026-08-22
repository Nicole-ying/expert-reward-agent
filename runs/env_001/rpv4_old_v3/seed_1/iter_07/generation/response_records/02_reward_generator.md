# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测
    x, y = obs[0], obs[1]
    # 下一观测
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = abs(next_obs[4])          # body_angle
    nangvel = abs(next_obs[5])         # angular_velocity
    nleft = next_obs[6]
    nright = next_obs[7]

    # 主进展信号：与目标着陆垫的距离变化
    d_current = (x**2 + y**2)**0.5
    d_next = (nx**2 + ny**2)**0.5 + 1e-8   # 避免除零
    progress = d_current - d_next          # >0 表示接近目标

    # soft health gate —— 在姿态、速度、角速度恶化时衰减正向进展
    # 使用指数衰减，温度系数根据任务特性设定
    angle_gate   = 2.718281828**(-2.0 * nangle**2)                       # 身体倾角过大时衰减
    v_sq = nvx**2 + nvy**2
    vel_gate     = 2.718281828**(-1.0 * v_sq / (1.0 + d_next))           # 高速时衰减，距离越近越严格
    angvel_gate  = 2.718281828**(-3.0 * nangvel)                         # 角速度过大时衰减

    gate = angle_gate * vel_gate * angvel_gate

    # 仅对正向进展施加 gate，负进展（远离）保留完整惩罚
    if progress > 0.0:
        progress *= gate

    # 任务完成近似信号：双腿同时接触着陆垫时给予持续性小奖励
    contact_reward = 0.2 if (nleft > 0.5 and nright > 0.5) else 0.0

    total_reward = progress + contact_reward
    components = {
        "progress": progress,
        "contact": contact_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 选定的任务家族与动力学子类型
- task_family: `navigation_goal_reaching`
- dynamics_subtype: `goal_approach_and_soft_contact`

## 选定的奖励职责 (reward roles)

| role | 实际组件 | 信号来源 |
|------|---------|---------|
| `goal_distance_reduction` (主学习信号) | `progress = d_current - d_next`，正向乘以 health gate | `x_position, y_position` → 距离改善 |
| `velocity_attenuation_near_goal` | 速度门控 `vel_gate` 在 `gate` 中，随距离自适应 | `x_velocity, y_velocity` + 距离 |
| `orientation_stabilization` | 角度门控 `angle_gate` 和角速度门控 `angvel_gate` | `body_angle, angular_velocity` |
| `task_completion_proxy` (条件职责) | `contact_reward` 每步双腿接触给予 +0.2 | `left_support_contact, right_support_contact` |

## 每个 role 选择的 formula operator
- `goal_distance_reduction` → **improvement_delta**：使用 `d_current - d_next`，鼓励每步向目标靠近，提供稠密梯度，且防止在好状态中停滞。
- 速度、姿态、角速度约束 → **soft_health_gate**：不采用传统的加性惩罚，而是通过乘法门控衰减正向进展奖励。门控体使用指数衰减函数，保证在安全状态下几乎不衰减（gate≈1），当身体状态恶化时主奖励被削弱，从而引导 agent 主动保持稳定。速度门控中分母 `(1.0 + d_next)` 使得“接近目标时对速度更为严苛”，天然实现 `velocity_attenuation_near_goal` 的职责。
- `contact_reward` → **dense_state_signal**（离散条件类型）：双腿接触时每步给予固定正奖励，作为着陆成功的软近似，提供接触阶段的持续梯度，避免完全稀疏的 terminal 奖励。

## 排除的职责及原因

- `terminal_success_bonus`：**excluded** — 环境无显式 success flag (`explicit_success_flag_available=false`)，且 reward 函数无 `done` 参数，无法安全判定终止时刻；由 `contact_reward` 的稠密形式替代。
- `terminal_failure_penalty`：**excluded** — 同样缺乏显式 failure flag，且终止时的灾难性状态无法在步级奖励中区别处理；通过姿态/速度门控间接抑制危险行为。
- `thrust_efficiency`：**excluded in v1** — 节省燃料为次要目标，v1 优先教会 agent 安全接近与着陆，避免节能惩罚过早压制探索。后续迭代若出现长时间滞空或无效动作可加入。
- 任何基于 `info` 的职责均不可用（info 为空）。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，且 `info` 字典固定为空，无法从环境接口获得 episode 终止时的成功/失败信息。因此所有 terminal 事件类的直接奖励均无法实现，改用稠密的进展信号与健康门控替代。

## 留待后续迭代的职责
- 燃料效率 (`thrust_efficiency`)：等 agent 能稳定着陆后再引入轻微的动作代价。
- 精细的水平速度对称约束或更高级的角速度阻尼：当前 gate 已提供基础约束，后续可根据回放中的振荡情况进行强化。
- 动态课程或门控权重调节：当前已是 adaptive gate，但门控参数固定；若出现学习停滞可调节温度系数。

## 训练后应观察的 failure modes
1. **主体悬停在目标正上方不下降**：`progress` 在近零距离时已无多少改善空间，且 `contact_reward` 权重较小，可能缺乏下降动力。观察器中若 y 保持较高且 `contact=0`，可考虑阶段性增加 `contact_reward` 权重或加入“高度过高的小惩罚”。
2. **高速撞击着陆垫**：`vel_gate` 可能仍不足以完全阻止，检查 fate 中 `vel_gate` 在撞击前的值，若整体 gate 太小导致主奖励消失过快，可能引发“放弃控制”现象；此时可提高温度系数或改为线性衰减门。
3. **单腿着陆后倾倒**：`contact_reward` 只在双腿接触时发放，姿态门控会强烈惩罚大角度，整体应有助于避免此模式；若仍出现，可考虑在单腿接触时给予轻微负奖励。
4. **长时间振荡无法稳定**：`angvel_gate` 和 `vel_gate` 将抑制高频运动，若持续振荡说明门控衰减过弱或 agent 在无进展区域反复机动；可能需增大角速度衰减系数。
5. **主体选择极保守策略导致进展缓慢**：`progress` 始终为小正值，未出现负进展但距离长时间不减少；此时可能 gate 过严，需在安全性与进展之间再平衡。
