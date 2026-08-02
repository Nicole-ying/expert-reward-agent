# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation variables
    x, y = obs[0], obs[1]
    x_v, y_v = obs[2], obs[3]
    angle = obs[4]
    ang_v = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ---------- 1. Progress reward: moving toward the landing pad (0,0) ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next          # positive when getting closer
    progress_reward = 1.0 * progress          # weight = 1.0

    # ---------- 2. Horizontal boundary penalty (crash prevention) ----------
    x_limit = 1.2
    x_boundary_penalty = 0.5 * max(0.0, abs(nx) - x_limit)

    # ---------- 3. Landing softness / safety penalty ----------
    # Velocity and angular velocity limits
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)                  # ideal angle is 0

    # Distance‑based activation gate: only enforce strict softness near the pad
    gate = 1.0 / (1.0 + 5.0 * dist_next)     # increases when close to target

    landing_safety_penalty = (0.1 * vel_pen + 0.05 * ang_pen + 0.1 * tilt_pen) * gate

    # ---------- Total reward ----------
    total_reward = progress_reward - x_boundary_penalty - landing_safety_penalty

    components = {
        "progress_reward": float(progress_reward),
        "x_boundary_penalty": float(x_boundary_penalty),
        "landing_safety_penalty": float(landing_safety_penalty)
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与职责选择

- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **selected reward roles / components**:
  1. `progress_reward` – 主学习信号，对应 `distance_to_target`（mandatory）
  2. `landing_safety_penalty` – 软着陆约束，对应 `landing_softness`（mandatory），仅靠近目标时激活
  3. `x_boundary_penalty` – 水平出界预防，对应 `crash_prevention`（mandatory，简化为边界惩罚）

- **excluded roles 及原因**:
  - `fuel_efficiency`：条件职责，v1 不引入，避免早期压制推力，留待后续迭代加入。
  - `terminal_success_reward / terminal_failure_penalty`：环境无显式成功/失败标志，info 为空，无法可靠判断终止步，因此不使用 terminal_event。

## 信号映射与使用的 formula operators

| 组件 | 使用的信号 | formula operator | 理由 |
|---|---|---|---|
| `progress_reward` | `x`, `y`, `nx`, `ny` | `improvement_delta`（`dist_curr - dist_next`） | 鼓励持续缩小到目标点的欧氏距离，每步有梯度，避免静止刷分 |
| `landing_safety_penalty` | `nx_v`, `ny_v`, `n_ang_v`, `n_angle` | `dense_state_signal (hinge)`（速度、角速度）+ `quadratic` 风格`tilt_pen`，乘以 `soft_health_gate`（距离倒数门） | 过高速、过大角速度或倾斜都会在接近目标时被惩罚；门控使早期探索不受过多压制 |
| `x_boundary_penalty` | `nx` | `dense_state_signal (hinge)` | 当 x 坐标超出视口边界时给予硬惩罚，防止飞出 |

- `improvement_delta` 被选择而非简单负距离奖励，因为它直接奖励“靠近”的动作，避免 agent 在好位置上停滞。
- landing safety 中的 `gate = 1/(1+5·dist_next)` 实现了“远近有别”：远距离时 gate→0，惩罚极小；靠近时 gate→1，强迫低速度与小姿态，符合着陆软性要求。
- 所有组件仅使用声明过的 `obs` / `next_obs` 维度，未访问 `info`、`original_reward`，也未依赖 `training_progress`。

## 未使用的能力与后续迭代方向

- **精确的成功/失败信号**：当前缺少显式标志，未来可尝试通过接触+位置+速度组合构造 `joint_condition_proxy` 作为任务完成近似，或通过 episode 结束时的最后一步模式识别给予 terminal bonus。
- **燃料效率**：当 agent 能稳定完成着陆后，再通过小系数的 `action_efficiency`（惩罚非零动作次数）优化能耗。
- **动作平滑性**：无历史动作记录，当前无法实现。
- **动态课程**：暂不需要。

## 训练后应关注的 failure modes

- **过早着陆/虚接触**：若 agent 出现一碰地就停止推力但不满足稳定条件的虚假成功，说明 `landing_safety_penalty` 的引导不足，可能需要增加靠近目标时的倾斜和速度惩罚强度，或引入接触条件联合奖励。
- **速度爆发后坠落**：如果 agent 在学习初期以极高速度冲向目标再弹开，说明 `progress_reward` 权重相对过大，`landing_safety_penalty` 门控激活太晚，可考虑降低 `v_limit` 或减小 gate 的分母系数（使 gate 更早激活）。
- **水平边界附近的振荡**：`x_boundary_penalty` 的阈值需与实际视口匹配，若阈值过小会过度约束机动，过大则会漏判。训练中应观察 `x_boundary_penalty` 的触发频率和位置分布。
- **角度/角速度惩罚压制探索**：若 agent 完全不敢使用姿态引擎，需降低惩罚系数，确保必要的旋转动作不被扼杀。