# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测 (环境卡声明的 8 维)
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 计算当前与下一步的距离和速度幅值
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 主学习信号：potential shaping 鼓励靠近目标垫并减速 ----
    w_dist = 2.0
    w_speed = 1.0
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping_reward = potential_next - potential_cur  # 正值 = 靠近/减速

    # ---- 稳定约束 1：姿态惩罚 ----
    w_angle = 0.5
    angle_penalty = -w_angle * (angle**2)

    # ---- 稳定约束 2：速度惩罚 ----
    w_vel = 0.1
    vel_penalty = -w_vel * (speed**2)

    # ---- 效率：离散动作燃料惩罚 ----
    if action == 2:           # 主发动机
        fuel_penalty = -0.15
    elif action in (1, 3):    # 左/右姿态引擎
        fuel_penalty = -0.02
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 任务完成近似信号：双腿接触 + 位置/速度近零的软乘积奖励 ----
    w_contact = 2.0
    pos_thresh = 1.0
    speed_thresh = 0.5
    proximity = max(0.0, 1.0 - (abs(nx) + abs(ny)) / pos_thresh)
    velocity_factor = max(0.0, 1.0 - next_speed / speed_thresh)
    contact_score = nleft * nright * proximity * velocity_factor
    contact_reward = w_contact * contact_score

    total_reward = (
        shaping_reward
        + angle_penalty
        + vel_penalty
        + fuel_penalty
        + contact_reward
    )

    components = {
        "shaping_reward": shaping_reward,
        "angle_penalty": angle_penalty,
        "vel_penalty": vel_penalty,
        "fuel_penalty": fuel_penalty,
        "contact_reward": contact_reward,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务家族与动力学子类
- `task_family`: navigation_goal_reaching
- `dynamics_subtype`: goal_approach_and_soft_contact
- 控制类型：离散（4 种动作）

## 2. 选定的奖励角色
根据 `reward_role_decomposition`，本版本覆盖：
- **主职责**：`goal_approach_and_soft_landing`（通过 potential shaping 同时引导位置、速度、姿态和最终接触）
- **条件职责**：`fuel_efficiency`（轻量离散动作惩罚）
- **稳定/安全约束**：姿态和速度连续惩罚

## 3. 角色-信号映射与公式算子
| 角色 | 选用信号 | 公式算子 | 设计意图 |
|------|----------|----------|----------|
| 核心进展 | 位置 `(x,y)`，速度 `(vx,vy)` | **potential_based_shaping** (3.3) + **improvement_delta** 形式 | 每一步都提供稠密梯度，鼓励同时靠近目标并减速 |
| 姿态稳定 | `body_angle` | **quadratic_penalty** (3.4) | 抑制大幅倾斜，保持直立 |
| 速度过冲 | `speed` (幅值) | **quadratic_penalty** | 防止撞击，但不过度惩罚低速段 |
| 燃料效率 | 动作 `action` | **离散动作惩罚** (变体) | 主发动机惩罚显著高于姿态引擎，鼓励省油 |
| 成功软代理 | 双腿接触 + 下一步位置/速度 | **joint_condition_proxy** (3.8) 乘积形式 | 仅当所有条件同时满足时才给予显著奖励，避免单腿接触或悬停骗分 |

## 4. 排除的角色及原因
- **`terminal_success_reward`**：环境无显式成功标志 (`explicit_success_flag_available=false`)，无法区分 `done` 原因，硬加终端奖励极可能奖励失败。
- **`terminal_failure_penalty`**：同理，无失败标志，且终止惩罚已由速度/位置偏离覆盖。
- **`shaping_based_on_original_reward`**：被明确禁止。
- **`time_pressure`**：v1 由环境截断自然施压，避免额外负奖励放大恐惧。

## 5. 未包含的职责 (留待后续迭代)
- 复杂门控 / 动态权重 / 高度相关速度上限
- 单独的单腿接触惩罚
- 动作平滑性代价（无历史动作可用）

## 6. 为何没有使用终端事件
`terminal_event` 算子需要可靠推断终止原因，但环境仅给出 `done` 布尔值，无法区分成功着陆（body_not_awake）与坠毁/越界。因此所有硬终端信号均不可靠，改用 **接触乘积奖励作为成功的软近似**。

## 7. 训练后应观察的失效模式
- **高速撞击**：训练曲线不增且 episode 很短 → 需增大速度惩罚或引入高度相关 hingep 惩罚。
- **悬停不降**：位置接近但始终不接触 → 可增加高度奖励 (在低 y 时给予小奖励) 或上调接触奖励权重。
- **越界漂移**：x 绝对值剧烈增大 → 加强水平偏差的 penalty 或引入更大权重的 shaping。
- **过早起火后坠落**：燃料惩罚过重导致不敢点火反致坠落 → 调低 fuel_penalty，或增大 shaping 权重以激励推进。
- **单腿着地即停**：接触奖励乘积中的双腿因子自动阻止单腿奖励，但仍需确认 body_not_awake 需要双腿同时接触。
- **不必要振荡**：角速度持续非零 → 可额外加入角速度惩罚（下一次迭代）。

该设计从机制上区别于之前记录中叠加大量子组件的尝试（如分开 approach + progress + landing_bonus 等），而是通过单一 potential shaping 统一驱动力，辅以轻量约束和乘积型软成功代理，构成一条不同的学习路径。