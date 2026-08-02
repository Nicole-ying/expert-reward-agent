# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander: approach target pad with safe attitude.
    Main progress: reduction in Euclidean distance to goal (0,0).
    Safety constraints: hinge penalty on large body angle and angular velocity.
    """
    # Unpack observations
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    angle = obs[4]
    angular_vel = obs[5]

    # ---------- 1) Main progress: distance reduction ----------
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist - next_dist           # positive when moving toward target
    progress = 1.0 * delta_dist             # main dense signal

    # ---------- 2) Safety constraints (hinge) ----------
    # Body angle – only penalise when beyond safe limit
    safe_angle = 0.5                        # radians, tuned for early training safety
    angle_error = max(0.0, abs(angle) - safe_angle)
    angle_penalty = -0.5 * angle_error

    # Angular velocity – discourage violent spinning
    safe_angvel = 1.0
    angvel_error = max(0.0, abs(angular_vel) - safe_angvel)
    angvel_penalty = -0.1 * angvel_error

    # ---------- Total reward ----------
    total_reward = progress + angle_penalty + angvel_penalty

    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动态子类型
- **task_family**: `navigation_goal_reaching`  
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
  飞行/着陆器从顶部出发，需到达并平稳停靠在中央目标垫 (0,0)，同时保持安全姿态。主目标为接近目标并软着陆，次目标为减少不必要的引擎使用。

## 2. 选中奖励角色与信号映射

| 角色 (role) | 职责 | 信号来源 | 公式算子 | 说明 |
|---|---|---|---|---|
| `delta_distance_to_target` (mandatory) | 每一步鼓励向目标垫靠近，提供持续稠密梯度 | `obs[0], obs[1]` 与 `next_obs[0], next_obs[1]` 构造的欧氏距离 | `improvement_delta` | 用 `dist - next_dist` 作为正向进展，避免悬停收割；即使停顿距离不减少，奖励为0（而非负） |
| `angle_constraint` (conditional/health) | 防止机体过度倾斜导致侧翻或失控 | `obs[4]` (body_angle) | `dense_state_signal` (hinge penalty) | 仅当 `|angle| > 0.5 rad` 时激活，不惩罚正常姿态调整 |
| `angular_velocity_constraint` (conditional/health) | 抑制剧烈自旋，维持可控角速度 | `obs[5]` (angular_velocity) | `dense_state_signal` (hinge penalty) | 仅当 `|angvel| > 1.0` 时激活，防止旋转失控 |

## 3. 排除的角色与原因

| 排除角色 | 理由 |
|---|---|
| `terminal_success_bonus` (derived_possible) | `compute_reward` 未提供 `done` 或 `terminated` 标志，无法可靠实现“仅最后一步触发”的一次性奖励；强行每步检测成功状态则变成连续 proximity‑style 奖励，有悬停利用风险。 |
| `terminal_failure_penalty` | 同无 `done` 信号，且 `explicit_failure_flag_available=false`，不能从 `info` 获得终止原因。 |
| `energy_penalty` (action‑based) | v1 阶段优先学习**到达并安全着陆**的基本行为，暂不引入能耗约束；离散动作空间下的推力惩罚容易抑制必要的主引擎点火，留待后续迭代。 |
| `survival_health_gate` | 已用更直接的 `angle_penalty` 和 `angvel_penalty` 显式约束姿态与角速度，无需再用乘性门控衰减主奖励。 |
| `proximity_reward` (如 `-distance`) | 会鼓励滞留在接近目标但不着陆的位置，违背安全着陆目标。 |
| 硬着路/冲击力惩罚 | 观测中无接触力或速度冲击信号，无法可靠区分软着陆与硬冲击。 |

## 4. 公式算子选择依据
- **主进展**：采用 `improvement_delta`，因为它只奖励向目标移动的**变化量**，即使智能体已经靠近目标，如果停止不动则得不到奖励，迫使它继续下降并着陆。  
- **姿态与角速度约束**：使用 **hinge penalty** (`dense_state_signal` 的 hinge 形式)，只对超过安全阈值的状态施加惩罚，避免在安全范围内抑制正常的姿态调整和转向。阈值 `safe_angle=0.5 rad` 和 `safe_angvel=1.0` 基于着陆器可行经验范围设定，后续可根据实际翻滚频率调整。

## 5. 未使用 terminal_success / terminal_failure 的原因
环境卡片明确 `explicit_success_flag_available=false`，且 `allowed_info_fields` 为空。`compute_reward` 接口不包含 `done` 参数，无法仅在对终止步施予一次性奖励/惩罚。因此 v1 依赖稠密的进展信号与安全约束驱动行为，不依赖稀疏事件。

## 6. 后续迭代预留职责
- **能量效率惩罚**：当智能体已能稳定着陆后，加入对主引擎 (`action=2`) 的微小惩罚（`penalty_on_action`），鼓励惯性滑行。  
- **软着陆代理**：在接近地面（`y` 很小）时对垂直速度施加条件惩罚，或引入 `joint_condition_proxy`（双腿接触+低速+角度小）作为连续完成信号，缓解硬着陆风险。  
- **边界漂移惩罚**：当 `|x|` 靠近视口边界时，加入递增的 `hinge penalty`，防止水平出界。  
- **角度门控的自适应**：根据训练曲线动态调整 `safe_angle` 阈值，或使用 `soft_health_gate` 在姿态恶化时乘性缩减主奖励，避免硬惩罚压制探索。  
- **课程权重**：若后期出现“快速冲撞→ reward 仍为正”的问题，可在接近目标时逐渐提高角度 / 速度约束权重。

## 7. 训练后应观察的 failure modes
| 失败模式 | 监测指标 | 可能原因 |
|---|---|---|
| **硬着陆撞击** | 终端前几步 `|y_velocity|` 很大，`contact flags` 刚变 1 即结束 | `progress` 信号仍鼓励快速下降，缺少终端减速约束；后续需加入接近地面时的速度惩罚或 soft landing proxy |
| **漂移出界** | `|x|` 逐步增大直至 episode 突然终止 | 缺少水平位置边界惩罚；智能体未学会主动纠正水平偏移 |
| **悬停不下降** | episode 长度极高，`contact flags` 长期为 0，`delta_dist` 近于 0 | `progress` 奖励太少（但 delta 不会鼓励悬停，应为 0）；若出现悬停，可能是随机策略或探索不足，需检查角度惩罚是否过严导致不敢动 |
| **单侧触地 / 翻倒** | `body_angle` 大，仅一侧 `contact=1`，reward 骤降后终止 | `angle_penalty` 未生效或阈值过高；可降低 `safe_angle` 或增大惩罚系数 |
| **高角速度旋转** | `angular_velocity` 长期超出 `safe_angvel`，`angvel_penalty` 持续触发但仍终止 | 惩罚权重不足或智能体已失控；可略微提高 `angvel_penalty` 系数，同时检查 `angle_penalty` 是否对快速翻滚的早期阶段反应不够快 |
