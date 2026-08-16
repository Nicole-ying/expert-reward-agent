# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. Approach target (main progress) ──
    distance = (x * x + y * y) ** 0.5
    approach_reward = -1.0 * distance

    # ── 2. Soft landing gate ──
    gate = 1.0 / (1.0 + 3.0 * distance)   # close to 1 when near pad

    vel_penalty = -0.3 * gate * (vx * vx + vy * vy + 0.1 * angular_vel * angular_vel)
    orientation_penalty = -0.2 * gate * (angle * angle)

    # ── 3. Fuel efficiency (light penalty) ──
    if action == 2:               # main engine
        fuel_penalty = -0.02
    elif action == 1 or action == 3:  # orientation engines
        fuel_penalty = -0.005
    else:                         # no engine
        fuel_penalty = 0.0

    total_reward = approach_reward + vel_penalty + orientation_penalty + fuel_penalty

    components = {
        'approach_reward': approach_reward,
        'vel_penalty': vel_penalty,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components
```

# reward_v1 设计说明

**任务家族与子类型**  
`task_family`: `navigation_goal_reaching`  
`dynamics_subtype`: `goal_approach_and_soft_contact`  
控制模式：离散推力器，严格基于观测的着陆引导。

**选择的奖励职责（roles）**  
根据环境卡片的职责拆解，v1 选择了以下角色：

| 角色 | 角色ID | 数学形式 | 使用的信号 |
|------|--------|-----------|------------|
| 主学习信号 | `approach_target` | `dense_state_signal` (负 L2 距离) | `next_obs[0]` (x), `next_obs[1]` (y) → 距离 |
| 软着陆约束 | `soft_landing` | `gated_penalty` + `quadratic_penalty` | `next_obs[2]` (vx), `next_obs[3]` (vy), `next_obs[5]` (角速度)，使用距离门控 |
| 姿态稳定（条件） | `orientation_stability` | `gated_penalty` + `quadratic_penalty` | `next_obs[4]` (body_angle)，使用距离门控 |
| 燃料效率 | `fuel_efficiency` | 离散动作条件惩罚（轻量） | `action` (0-3) |

**职责-信号映射与公式选择**  
- **approach_target**：直接使用 L2 距离的负值，为每一步提供稠密梯度，驱动 agent 向目标垫中心（x=0, y=0）移动。  
- **soft_landing + orientation_stability**：使用 `soft_health_gate` 思想，通过 `gate = 1/(1+3*distance)` 在接近目标时激活速度与姿态惩罚。远距离时 gate ≈ 0，速度不受约束，允许快速机动；接近时 gate 增大，迫使减速和扶正。惩罚采用二次型以提供平滑梯度。  
- **fuel_efficiency**：轻量离散动作惩罚（主引擎 -0.02，姿态引擎 -0.005），抑制不必要的推力，避免无限盘旋。

**排除的职责（及原因）**  
- `two_leg_contact_reward`：虽为条件职责，但 v1 先不引入硬接触奖励，避免误触发导致的奖励泄露。依靠距离和速度引导自然着陆。  
- `explicit_success_bonus`：无显式 `info` 成功标志，不可用。  
- `time_penalty`：无可用步数计数器，且与燃料惩罚语义重叠。  
- `terminal_success_reward` / `terminal_failure_penalty`：均因环境不提供终止标志而不可用。v1 通过连续信号引导避免依赖事件信号。

**设计假设（与先前尝试的区别）**  
此设计跳出了之前的高斯乘积或进度加权框架，改用 **距离线性奖励 + 距离自适应门控速度惩罚**。核心假设：agent 远处需要自由机动，近处需要强制减速和姿态控制。通过同一个 `gate` 同时调节速度和角度约束，数学结构简单、梯度连续，且避免了因全时速度惩罚或过早姿态约束导致的探索困难。燃料惩罚保持极轻，防止压制主信号。

**后续迭代方向**  
- 可加入基于双腿接触的条件正奖励，结合位置/速度阈值，进一步强化最终软着陆。  
- 若出现悬停不落，可增加时间衰减项或动态缩放 `approach_reward` 对 y 的敏感性。  
- 燃料惩罚可根据训练阶段动态调整，实现更精细的能耗优化。

**训练后应观察的失效模式**  
- 悬停：y 不再下降，距离停滞。检查 gate 是否过强抑制了低速下降。  
- 高速撞地：距离快速减小但速度未收敛，检查 `gate` 激活窗口是否过窄（即需更早增大惩罚）。  
- 侧翻误终止：单边接触且角度大，但奖励函数正面反馈不足，需观察 `orientation_penalty` 是否足够。  
- 漂出边界：x 绝对值持续增大，可能需要额外水平位置惩罚或水平速度 damping。  
- 引擎滥用：动作 2 频率过高，可微调 `fuel_penalty` 权重。