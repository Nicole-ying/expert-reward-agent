# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y = obs[0], obs[1]
    xv, yv = obs[2], obs[3]
    angle = obs[4]
    # angv = obs[5]  # not used directly, could be added later if needed
    lc, rc = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    # nangv = next_obs[5]
    nlc, nrc = next_obs[6], next_obs[7]

    # Euclidean distances
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ================= A. Progress: delta distance =================
    # Dense reward for moving toward the target platform (0,0)
    w_progress = 10.0
    progress_reward = w_progress * (dist - next_dist)

    # ================= B. Velocity penalty near target =================
    # Apply only when close to the target (dist < d_threshold)
    d_threshold = 0.5  # meters, safe distance to start slowing down
    proximity_factor = max(0.0, 1.0 - dist / d_threshold)  # 1 at target, 0 beyond threshold

    safe_vx = 0.2
    safe_vy = 0.3
    excess_vx = max(0.0, abs(xv) - safe_vx)
    excess_vy = max(0.0, abs(yv) - safe_vy)
    w_vel_penalty = 2.0
    velocity_penalty = -w_vel_penalty * proximity_factor * (excess_vx + excess_vy)

    # ================= C. Touchdown event reward =================
    touchdown_reward = 0.0
    prev_double_contact = (lc > 0.5 and rc > 0.5)
    next_double_contact = (nlc > 0.5 and nrc > 0.5)
    if not prev_double_contact and next_double_contact:
        # Check soft landing requirements on the touchdown step
        speed_ok = abs(nxv) < safe_vx and abs(nyv) < safe_vy
        angle_ok = abs(nangle) < 0.2  # radians, near upright
        if speed_ok and angle_ok:
            touchdown_reward = 50.0

    total_reward = progress_reward + velocity_penalty + touchdown_reward

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "touchdown_reward": touchdown_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact` (2D lander)

## selected reward roles
1. **progress_to_target** (mandatory) – 引导智能体向目标 (0,0) 靠近。  
2. **soft_landing** (mandatory) – 在接近目标时抑制过高速度，并奖励符合着陆条件的接触事件。  
3. **fuel_efficiency** (conditional) – 本次 **未使用**，理由见排除说明。

## role-to-signal mapping & formula operators
- **progress_to_target**  
  - 信号：`obs[0] (x)`, `obs[1] (y)`, `next_obs[0], next_obs[1]`  
  - 算子：**improvement_delta** (`reward = w * (old_distance - new_distance)`)。  
  - 理由：每步都提供梯度，激励真正靠近而非停留在好位置。避免纯绝对距离奖励在远距离平台期或终点过强的问题。

- **soft_landing**  
  - 子组件1：**velocity_penalty near ground**  
    - 信号：`obs[2] (xv)`, `obs[3] (yv)`, 距离 `dist`  
    - 算子：**hinge** (`-w * max(0, |v| - safe_v)`) 乘以 **proximity_factor**（线性衰减 `1 - dist/threshold`）。  
    - 理由：只在接近目标时生效，防止远距离错误减速。hinge 仅在超速时惩罚，允许安全范围内的移动。  
  - 子组件2：**touchdown event reward**  
    - 信号：接触标志 `obs[6]/[7]` 和 `next_obs[6]/[7]`，以及 `next_obs` 中的速度、角度。  
    - 算子：**terminal_event** + **joint_condition_proxy**（双接触 + 低速 + 小角度的逻辑“与”）。  
    - 理由：没有显式 success flag，只能通过观测推断着陆瞬间并给予一次性奖励，避免持续刷分；需同时满足多条软着陆条件才给奖，防止“硬撞”也得酬。

## excluded roles and reasons
- **fuel_efficiency** – v1 核心目标是学会到达并软着陆，引入任何动作代价都可能压制发动机探索，导致 agent 不敢点火而坠毁（已从历史失败设计中吸取教训）。后续迭代开启。  
- **explicit_success_bonus_from_info** – `info` 为空，没有 success flag。  
- **trajectory_smoothing** – 离散动作空间下动作平滑性难以定义且可能与最优 bang‑bang 控制冲突。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available = false`，`explicit_failure_flag_available = false`，`info` 字典为空，因此不能也不应该依赖这类硬编码的终端信号。本设计使用可观测状态重构了软着陆事件（touchdown）作为积极反馈，并将失败信号留到后续迭代通过速度/姿态惩罚间接解决。

## 哪些职责留到后续迭代
- **fuel_efficiency**：当 agent 已稳定着陆后，通过极小权重或基于着陆成功后的节能回溯奖励加入。  
- **posture/angular velocity explicit penalty**：当前 body_angle 和角速度仅通过 touchdown 条件间接约束，未来若出现旋转失控可增加接近区域的角度/角速度惩罚。  
- **failure‑mode‑specific terminal penalties**：例如飞出视野、碰撞地面等，目前由 delta‑distance 和速度惩罚间接压制，后续可加入明确的失败检测与大型负奖励（基于观测推断）。

## 训练后应该观察哪些 failure modes
| failure mode | 观察指标 | 可能干预 |
|---|---|---|
| 持续自由落体不点火（action=0 主导） | `progress_reward` 为负且 episode 迅速终止 | 增加 `progress_reward` 权重，或临时加入极小动作多样性奖励 |
| 高速冲撞目标垫板 | 着陆时速度远超 `safe_v`，但可能仍获得 touchdown 奖励？实际上 touchdown 条件要求低速，所以 agent 不会获得奖励；但仍需检查 episode 是否因 crash 终止 | 若 agent 始终无法低速着陆，可降低 `safe_v` 阈值或增大 `velocity_penalty` 权重 |
| 接近目标后旋转失控 | `angle` / `angular_velocity` 高频振荡 | 在接近区域添加角速度惩罚（如 `soft_health_gate` 或 hinge） |
| 单腿着地或倾斜不稳 | 只有单接触且 angle 较大时 episode 结束 | touchdown 条件已要求双接触，但若 agent 侥幸满足低速低角但单腿接触便获得奖励，可调整 touchdown 条件严格要求双接触且速度/角度 |
| 过度依赖主发动机冲高再坠落（刷 progress） | `progress_reward` 持续为正但终端状态失败 | 本设计的 progress 只奖励靠近，不会反向奖励“先远离再靠近”的循环，因为每步都是增量；但若 agent 蓄意远离再冲刺，仍可能获利，后续可通过添加生存性约束缓解 |

## 与历史尝试的结构性差异
- 彻底摒弃了燃料效率、过量方向/速度全局惩罚以及复杂的多组件叠加。  
- 将 **progress** 从常见的“距离倒数”或“二次惩罚”改为 **delta‑distance**，提供更直观的移动奖赏，使 agent 每一步靠近都有正向激励。  
- 将速度约束与 **接近度** 解耦，只在目标附近生效，避免“远处不敢加速”。  
- 首次引入 **touchdown 事件** 作为软着陆的唯一大额正奖，用可观测状态模拟 success 信号，意图让 agent 学会在接触前主动减速、摆正姿态。