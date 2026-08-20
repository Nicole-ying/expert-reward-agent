# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for goal_approach_and_soft_contact (discrete thrusters, 2D lander).
    """
    # ------------------------------------------------------------------
    # unpack observations (current)
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    # unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ------------------------------------------------------------------
    # 1. goal_approach (potential‑based shaping using L2 distance to pad)
    #    target is (0,0) on the pad.
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # ------------------------------------------------------------------
    # 2. soft_landing – conditional velocity/angular‑velocity penalty near ground
    #    gate rises as altitude drops below threshold
    LANDING_THRESHOLD = 0.5           # meters above pad
    land_weight = max(0.0, 1.0 - (y_pos_next / LANDING_THRESHOLD))
    # quadratic penalty on linear + angular speed in next state
    soft_landing = -land_weight * (
        x_vel_next ** 2 + y_vel_next ** 2 + 0.1 * ang_vel_next ** 2
    )

    # ------------------------------------------------------------------
    # 3. crash_avoidance – harsh penalty when a crash‑like state is reached
    #    crash = below pad (negative height) AND not both legs on contact
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    CRASH_PENALTY = -10.0
    if y_pos_next < -0.2 and not BOTH_LEGS:
        crash_avoidance = CRASH_PENALTY
    else:
        crash_avoidance = 0.0

    # ------------------------------------------------------------------
    # Combine components
    total_reward = goal_approach + soft_landing + crash_avoidance

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "crash_avoidance": crash_avoidance,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与动态子类型
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- 核心目标：双足着陆器在目标垫上平稳降落，要求两腿同时接触、姿态竖直、极低速度。

## 选用的奖励职责（mandatory roles）
依据 environment_card.md 的 `reward_role_decomposition`，v1 只包含三个必选职责：
1. **goal_approach** – 促进飞行器靠近目标垫。
2. **soft_landing** – 在接近地面时压制过快线速度和角速度，保障柔顺着陆。
3. **crash_avoidance** – 对坠地 / 撞垫等灾难性状态施加强惩罚。

条件职责（燃料效率、姿态惩罚、时间奖励）以及所有被禁用的职责均不在 v1 中使用，避免过早引入效率/能量约束干扰任务学习。

## 信号映射与公式算子选择

| 职责 | 使用信号 | 公式算子 | 选择理由 |
|------|----------|----------|----------|
| goal_approach | `obs[0:2]` (x,y) | potential‑based shaping (`dist_current - dist_next`) | 使用 L2 距离的减小量作为每步正反馈，且每步都有梯度；比单纯的 `-distance` 更能激励持续靠近。 |
| soft_landing | `next_obs[1]` (y) 作为门控；`next_obs[2:3]` (vx,vy) + `next_obs[5]` (ω) 作为惩罚项 | conditional quadratic penalty with linear gate | 只在高度低于阈值（0.5 m）时激活，用线性衰减门避免突变；惩罚本步产生的速度平方，鼓励着陆前减速。 |
| crash_avoidance | `next_obs[1]` (y) + `next_obs[6:8]` (接触标志) | terminal‑event‑like 常数惩罚 | 在检测到`y < -0.2`且非双腿接触时给予固定大负值（-10.0）；仅依赖安全/可观测信号，不尝试推测 x 边界（边界值未暴露）。 |

## 排除的职责及原因
- **terminal_success_reward / terminal_failure_penalty**：环境中没有显式成功/失败标志（`explicit_success_flag_available=false`），`info` 为空，无法可靠推断。
- **orientation_penalty, fuel_efficiency, progress_time_bonus**：均为条件职责，v1 优先建立安全到达趋势，姿态与燃油优化留待后续迭代（过早加入会限制必要机动）。
- **硬边界的 x‑越界惩罚**：视口边界值未在环境卡片中给出，硬编码易误判或失效；goal_approach 对水平位置偏差的惩罚已能抑制严重漂移。

## 待后续迭代的职责
- `orientation_penalty` – 接近地面时明确要求姿态竖直。
- `fuel_efficiency` – 限制主引擎使用，优化能耗。
- `progress_time_bonus` – 在可靠安全约束下给予微弱时间奖励，加速收敛。
- 可选的 `soft_health_gate` – 如果出现“先冲后死”模式，用姿态误差衰减主奖励。

## 训练后应重点观察的失效模式
1. **悬停不下降** — y 位置不减小，goal_approach 停滞；可能需要加大下降过程中的 shaping 强度或放宽速度约束。
2. **高速撞击导致 crash** — 终止时 y 接近 0 但两腿未完全接触；需调强 `soft_landing` 权重或降低 `LANDING_THRESHOLD`。
3. **单腿着陆侧翻** — 一侧接触、另一侧未接触，同时 body_angle 变大；需后续 `orientation_penalty` 或更细粒度的接触一致性奖励。
4. **水平漂移出界** — x 位置单调增大；若 goal_approach 不足以遏制，可在 v2 引入横向位移代价或潜在奖励修正。
5. **着陆附近震荡** — 位置误差小但速度控制不稳定，episode 过长；可通过调整 soft_landing 门曲线或加入基于 y 的进度塑造改善。

本版奖励函数优先保证安全到达任务目标，并为后续效率与精细姿态控制留下清晰的扩展点。
