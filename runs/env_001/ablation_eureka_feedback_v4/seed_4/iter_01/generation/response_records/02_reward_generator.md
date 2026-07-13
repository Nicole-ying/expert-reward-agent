# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next positions relative to landing pad center
    dx_curr, dy_curr = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_curr = (dx_curr**2 + dy_curr**2) ** 0.5
    dist_next = (dx_next**2 + dy_next**2) ** 0.5

    # Approach progress: positive when getting closer to the pad
    approach_delta = dist_curr - dist_next

    # Next‑step velocity and safety threshold
    vx_next, vy_next = next_obs[2], next_obs[3]
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    safe_speed = 0.2

    # Next‑step orientation and angular velocity
    angle_next = next_obs[4]
    angvel_next = next_obs[5]

    # Contact signals after the step
    contactL = next_obs[6]
    contactR = next_obs[7]
    both_contact = contactL * contactR  # 1 only when both legs touch

    # --- Reward weights ---
    w_approach = 2.0
    w_vel_penalty = 5.0           # per unit speed above safe_speed
    w_angle = 0.5                 # penalty on squared body angle
    w_angvel = 0.1                # penalty on squared angular velocity
    w_landing = 5.0               # bonus for a clean two‑leg landing

    # Component 1: dense progress towards the pad
    approach_reward = w_approach * approach_delta

    # Component 2: speed constraint (hinge – only penalise when exceeding
    # safe threshold)
    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    # Component 3: angular stability (quadratic penalties)
    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # Component 4: soft landing‑quality proxy
    # Continuous factors make the reward smooth when the agent is close and slow,
    # but it only triggers after both legs have contacted the pad.
    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor * both_contact

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
`navigation_goal_reaching` / `goal_approach_and_soft_contact`  
主体是一个带两条支撑腿的2D飞行器，使用离散脉冲引擎。任务要求到达中央着陆垫并稳定停靠，同时避免高速撞击、翻倒或飞出边界。

## selected reward roles
按照环境卡片的任务画像和职责拆解（主目标 > 次要目标），v1 选择了四个组件，覆盖主学习信号、安全约束和软着陆引导：

| 组件 (role) | 职责 | 公式算子 |
|---|---|---|
| `approach_progress` | **主学习信号**：每步驱动 agent 向目标垫靠近 | `improvement_delta` |
| `velocity_penalty` | 安全约束：抑制过快速度，防止高速撞击 | `dense_state_signal (hinge)` |
| `angle_stability` | 安全约束：保持竖直姿态和低角速度，防止翻倒 | `quadratic_penalty` |
| `landing_quality` | 任务完成近似：在双腿稳定着陆后提供持续正向激励 | `joint_condition_proxy` |

## role→signal mapping
- **approach_progress** ← `obs[0]` (x), `obs[1]` (y), `next_obs[0]`, `next_obs[1]`  
  计算当前与下一步到垫子中心的欧氏距离差。
- **velocity_penalty** ← `next_obs[2]` (x速度), `next_obs[3]` (y速度)  
  合成速度幅值，超过 `safe_speed = 0.2` 时线性惩罚。
- **angle_stability** ← `next_obs[4]` (机体角度), `next_obs[5]` (角速度)  
  二次惩罚偏离竖直的角度和转动。
- **landing_quality** ← `next_obs[6]` (左腿接触), `next_obs[7]` (右腿接触), `dist_next`, `speed_next`  
  只有当双腿均接触（乘积为1）时，才用连续因子（距离衰减、速度衰减）计算奖励；乘积为零时奖励为0，防止空中误发。

所有信号均在 `environment_card.md` 中声明为 `reward_usable: true`，且未使用 `info` 或 `original_reward`。

## excluded roles 及原因
- **terminal_success_reward / terminal_failure_penalty**：环境 `explicit_success_flag_available = false`，`info` 为空，无法可靠区分终止原因。
- **action_cost / fuel_penalty**：v1 阶段优先学会安全着陆；效率优化留到后续迭代。
- **time‑to‑land reward**：同样属于次要目标，v1 不引入，避免过早压缩探索。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境卡片明确说明 info 字段不可用，且唯一的终止标志是模糊的，无法安全区分成功与失败。因此所有奖励都必须从观测状态构建，不使用稀疏终端信号。

## 哪些职责留到后续迭代
- **最小化燃料 / 推力使用**（动作代价）：当 agent 已能可靠着陆后，引入轻微的动作惩罚以优化效率。
- **时间优化**：可通过 curriculum 或步数奖励激励更快着陆。
- **更精细的接触质量评估**：例如区分单腿着陆的危险状态，或附加脚底冲击力的估计。

## 训练后应观察的 failure modes
- 高速撞击：agent 为了快速接近 pad 而不减速。
- 翻倒：角速度或倾斜过大导致着陆后向一侧倾倒。
- 垂直弹跳：agent 可能通过反复高频点火产生垂直振荡来赚取 `approach_progress` 或 `landing_quality`（虽然后者受接触限制，但仍需监控）。
- 停滞不前：如果惩罚项过强，agent 可能原地漂浮而不敢前进。
- 边界飞出：虽无专门惩罚，但终止会自然截断 episode；需确保 agent 不会利用飞出前几步获取正奖励（`approach_progress` 可能因冲向边界而短暂为正）。
