# Response Record

## 诊断报告

### Evidence

| 信号 | 数据 |
|---|---|
| **Score** | -159.49（目标 200，差距极大） |
| **Episode length** | 303.35（接近 max，说明 agent 在挣扎而非高效完成任务） |
| **terminated** | 20/20（全部提前终止，即 crash / out-of-bounds / settled） |
| **contact_reward** | ep_sum=0.000, active_rate=0%, **所有 checkpoint 均为 0** |
| **vel_damping** | ep_sum=0.000, active_rate=0%, **所有 checkpoint 均为 0** |
| **progress_reward** | magnitude_share=93.2%，完全主导 |
| **Checkpoint score trend** | -758 → -155，改善但 plateau，从未接近正值 |

### Behavior Diagnosis

**当前策略学到的是"冲向目标然后撞毁"**。证据链：

1. `progress_reward`（delta-distance，w=1.0）占据 93.2% 的 magnitude share，agent 被这个信号完全支配——它的唯一目标就是缩小与原点距离。
2. `contact_reward` 的条件 `left_contact * right_contact`（双足同时触地）在 100% 训练中从未被触发过一次——agent 根本发现不了这个状态。即使偶尔碰到，`sigma_angle=0.1` 和 `sigma_vel=0.2` 的指数衰减过于严苛，稍有倾斜或速度就归零。
3. `vel_damping` 的硬阈值 `dist_thresh=1.0` 意味着只有距离 < 1.0 才激活——agent 从未稳定地进入这个区域，该组件全程死寂。
4. **没有成功着陆的终局奖励**：agent 不知道"好着陆"长什么样。唯一正信号是 progress_reward，它奖励"靠近"但不奖励"停稳"。所以 agent 学会的最优策略就是：冲向原点 → 高速撞击 → crash。

**这不是渐进优化问题，而是两个核心组件完全失效 + 缺少成功信号的结构性问题。**

### 干预方案

1. **新增 `landing_bonus`**：当双足触地 + 近原点 + 竖直 + 低速时，给予每步大额奖励（+10/step）。这是最重要的改变——给 agent 一个明确的"成功"定义。
2. **重写 `contact_reward`**：从 `left * right`（二元 AND，极稀疏）改为 `left + right`（0/1/2，渐进式）。任何触地都有奖励，双足更多——提供可发现的梯度。
3. **重写 `vel_damping`**：从硬阈值 gate 改为指数衰减 gate `exp(-dist²/σ²)`，始终激活，近垫强远垫弱。
4. **新增 `action_reward`**：惩罚发动机使用，鼓励滑翔/惯性。
5. **替换 `progress_reward`**：从 delta-distance 改为负距离（`-w * dist`），信号更稳定、量级更可控。
6. **全局重标定**：确保没有单一组件 magnitude_share > 50%。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Complete rewrite.
    - Negative distance to pad (smooth, always active goal proximity)
    - Orientation penalty (quadratic on angle + angvel)
    - Velocity damping via exponential proximity gate (always active)
    - Progressive contact reward (any foot → bonus, both → more)
    - Landing success bonus (large per-step reward when settled)
    - Action efficiency (reward idle, penalize engine use)
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ──── 1. Goal proximity: negative distance to origin ────
    dist = (x ** 2 + y ** 2) ** 0.5
    w_proximity = 0.3
    proximity_reward = -w_proximity * dist

    # ──── 2. Orientation penalty ────
    w_angle = 0.08
    w_angvel = 0.03
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ──── 3. Velocity damping (exponential gate, always active) ────
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # gate ≈ 1 when dist ≈ 0, gate → 0 when dist ≫ σ
    sigma_dist = 1.5
    proximity_gate = 2.718281828 ** (- (dist ** 2) / (sigma_dist ** 2))
    w_damp = 0.4
    vel_damping = -w_damp * proximity_gate * speed

    # ──── 4. Progressive contact reward ────
    # 0, 1, or 2 feet on ground — any contact is progress
    contact_sum = left_contact + right_contact
    w_contact = 0.5
    contact_reward = w_contact * contact_sum

    # ──── 5. Landing success bonus ────
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist < 0.3)
    upright = (abs(angle) < 0.15)
    slow = (speed < 0.3)
    is_good_landing = both_on_ground and near_pad and upright and slow
    w_landing = 10.0
    landing_bonus = w_landing * (1.0 if is_good_landing else 0.0)

    # ──── 6. Action efficiency ────
    if action == 0:          # no engine — reward gliding
        action_reward = 0.03
    elif action == 2:        # main engine — highest cost
        action_reward = -0.08
    else:                    # orientation engines (1 or 3) — low cost
        action_reward = -0.03

    # ──── Aggregate ────
    total_reward = (
        proximity_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components
```
