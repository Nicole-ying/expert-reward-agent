# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Lunar Lander 风格任务奖励函数 v1。

    设计理念:
    - 主学习信号: 基于距离势能的 shaping 奖励，持续引导 agent 靠近目标平台。
    - 软着陆约束: 在下降末段对垂直速度施加 hinge 惩罚，防止高速撞击。
    - 姿态稳定: 轻量二次型角度惩罚，鼓励竖直姿态。
    - 能量效率: 轻量动作代价，抑制不必要的引擎使用。
    未使用稀疏成功/失败奖励，因为没有显式标志且稀疏信号难以在 v1 有效学习。
    """
    # ------------------------------------------------------------
    # 0. 解析观测
    # ------------------------------------------------------------
    # 从 obs 提取状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    # 从 next_obs 提取状态（用于势能塑造）
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ------------------------------------------------------------
    # 1. 主学习信号: 距离势能塑造 (goal_progress_guidance)
    # ------------------------------------------------------------
    # 用于持续引导智能体靠近目标 (x=0, y=0)
    # 势能函数: -sqrt(x^2 + y^2)，距离越近势能越高
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    potential_curr = -dist
    potential_next = -next_dist
    
    # 势能塑造: 向目标靠近一步的奖励增量
    progress_reward = 5.0 * (potential_next - potential_curr)
    
    # ------------------------------------------------------------
    # 2. 稳定/安全约束: 软着陆速度惩罚 (soft_landing_safety, 针对垂直速度)
    # ------------------------------------------------------------
    # 在高度较低时（y_pos < 0.5），严格惩罚过大的向下速度
    # 使用 hinge 惩罚：只惩罚超过安全阈值的 y_vel (向下为负，越大越负越危险)
    # 安全阈值：着陆时 y_vel 应 > -0.2 (即下降速度不超过 0.2)
    safe_y_vel_threshold = -0.2
    proximity_gate = max(0.0, 1.0 - y_pos / 0.5)  # y<0.5 时逐渐激活 (0->1)
    proximity_gate = min(1.0, proximity_gate)
    
    # 危险速度部分: 如果 y_vel < -0.2 (下降过快)，则产生惩罚
    dangerous_y_vel = max(0.0, -y_vel + safe_y_vel_threshold)  # 例如 y_vel=-0.8 -> max(0, 0.8-0.2)=0.6
    landing_velocity_penalty = -2.0 * proximity_gate * dangerous_y_vel
    
    # ------------------------------------------------------------
    # 3. 稳定/安全约束: 姿态惩罚 (soft_landing_safety, 针对角度)
    # ------------------------------------------------------------
    # 始终鼓励竖直姿态，但权重适中，不阻碍必要的机动
    # 使用二次型：轻微偏离时惩罚很小，大幅偏离时惩罚变大
    angle_penalty = -0.5 * (body_angle ** 2)
    
    # ------------------------------------------------------------
    # 4. 能量效率: 引擎动作代价 (energy_efficiency)
    # ------------------------------------------------------------
    # 对任何引擎点火（action != 0）施加小惩罚，鼓励利用惯性飞行
    # 注意：权重必须很小，不能压制主任务
    action_cost = -0.02 if action != 0 else 0.0
    
    # ------------------------------------------------------------
    # 5. 总奖励组装
    # ------------------------------------------------------------
    total_reward = (
        progress_reward +
        landing_velocity_penalty +
        angle_penalty +
        action_cost
    )
    
    # 记录各个组件（只记录被加进 total 的）
    components = {
        "progress_shaping": progress_reward,
        "landing_velocity_penalty": landing_velocity_penalty,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像匹配
- **task_family**: navigation_goal_reaching
- **dynamics_subtype**: goal_approach_and_soft_contact
- **control_type**: discrete actions over continuous dynamics
- 匹配理由：主目标是到达指定位置并安全接触，符合 goal_reaching；核心挑战在于下降末段的减速与姿态稳定，对应 soft_contact。

## 2. 奖励职责选择与信号映射

| Role ID | Selected? | Formula Operator | Usable Signals | Reason |
|---|---|---|---|---|
| `goal_progress_guidance` | **Yes (核心)** | `potential_based_shaping` | `x_position`, `y_position` | 必须持续引导至目标。势能塑造比单纯距离惩罚提供每步差分信号，鼓励进展而非停滞。 |
| `soft_landing_safety` | **Yes (约束)** | `hinge` (垂直速度) + `quadratic_penalty` (角度) | `y_velocity`, `body_angle`, `x_velocity` (备用) | 防止高速坠毁是核心安全要求。对垂直速度使用 hinge 只在过界时惩罚，避免压制正常下降；对角度使用二次型提供平滑姿态约束。 |
| `energy_efficiency` | **Yes (轻量)** | `action_cost` (离散条件) | `action` | 任务要求“as little engine thrust as possible”。极小权重(-0.02)表达意图，不压制主任务进步。 |
| `horizontal_alignment_and_stabilization` | **No (v1 隐式)** | — | `x_position`, `x_velocity` | 暂不独立实现。原因：(1) 主进展信号已包含水平位置通过距离势能引导；(2) 在接近目标时，距离势能会自动加强水平对齐需求；(3) 横向速度稳定留到 v2 根据实际摆动情况决定是否加专项惩罚。 |
| `sparse_mission_success_reward` | **No (v1 排除)** | — | `next_obs` (推断) | 排除原因：(1) 环境无显式 success flag，需多条件乘积组合构造 proxy，易塌缩为 0 或误触发；(2) 稀疏信号在无课程引导的 v1 中几乎不可学习；(3) v2 可在 agent 已学会基础着陆后，加入严格门控的稀疏成功奖励以微调着陆品质。 |
| `terminal_failure_penalty` | **No (v1 排除)** | — | `next_obs` (推断) | 排除原因：无显式 failure flag，需从 `next_obs` 推断坠毁边界。v1 优先用密集速度惩罚防止这些事件发生；失败终止惩罚留到 v2 中在 agent 已有基本安全行为后精确标记灾难性失败。 |
| `survival_balance` | **No (禁止)** | — | — | 任务不是持续平衡；无生存时间信号。 |
| `exploration_bonus` | **No (禁止)** | — | — | 状态空间简单，目标明确，无需刻意鼓励探索。 |

## 3. 排除的职责与阶段考量
- **terminal_success_reward**: 无显式 `info["success"]`，且环境卡片禁止使用未声明的 info 字段。v1 完全依赖密集 shaping。
- **terminal_failure_penalty**: 无显式 `info["failure"]` 或 `termination_reason`。v1 完全依赖密集速度/角度约束防止失败状态出现。
- **horizontal_alignment_and_stabilization**: 作为独立组件留到 v2。当前通过距离势能间接引导水平位置。
- **效率强化项** (如随时间递增的动作代价、hover 惩罚): v2 可加入，解决“成功悬停但迟迟不着陆”等问题。
- **contact-conditional 着陆奖励**: v2 可加入，在支撑脚触地且满足安全条件时给予一次性正奖励。

## 4. 组件权重与尺度考量
- **progress_shaping** (系数 5.0): 提供主导梯度。在初始高度约 1.0-2.0 时，下降一步的 delta 约 0.1-0.3，奖励 0.5-1.5/步。
- **landing_velocity_penalty** (系数 -2.0): 仅在接近地面且下降过快时激活。例如 y_pos=0.2, y_vel=-0.8: 惩罚 = -2.0 × 0.6 × 0.6 ≈ -0.72。权重适度，不会压倒 progress 但足以让 agent 学会减速。
- **angle_penalty** (系数 -0.5): body_angle 通常以弧度计，小幅倾斜(<0.3)惩罚<0.045，大幅倾斜(>1.0)惩罚>0.5。合理平衡机动自由与姿态要求。
- **action_cost** (系数 -0.02): 仅约 progress 步奖励的 1-5%。不会迫使 agent 完全不用引擎，但提供稀疏的节约激励。

## 5. 训练后应观察的 failure modes
- **Hover 过度使用主引擎**: 动作序列中 main_engine 占比极高但 y_vel 接近 0，说明 agent 在悬停而非高效下降 → v2 可加 hover 惩罚或增加 progress 敏感度。
- **冲刺式坠毁**: 早期 progress 很高但突然 terminated。检查 y_vel 是否持续为负且未被 velocity penalty 充分抑制 → 可能需要增加 velocity penalty 权重或扩展 proximity_gate 作用范围。
- **边界探索不足**: x_pos 缓慢漂移至边界导致截断 → v2 可加 x 偏移的软边界惩罚。
- **完全放弃引擎控制**: action 全为 0 导致无限上升或缓慢漂移 → progress 足够强时不应出现；若出现说明 action_cost 过高，需降低。
- **成功着陆后仍点火**: contact=1 且速度安全时仍选择引擎动作 → v2 可加“着陆后引擎惩罚”或引入成功条件判断截断。

## 6. v2 迭代方向
1.  加入 `sparse_success_proxy`: 基于多条件乘积 (`proximity * low_speed * upright * contact`)，检测到成功着陆时给予较大奖励 (`+10`)。
2.  加入 `horizontal_stabilization`: 在低高度时强化横向位置和速度的对准。
3.  加入 `hover_penalty`: 如果高度很低但长时间不接触，施加递增动作代价。
4.  调整 `progress_shaping` 为自适应：着陆阶段自动提高速度/角度约束权重。
5.  引入 `soft_health_gate`: 如果 terminated 率仍高且主进展信号在失败回合中仍显著为正，用姿态健康门衰减 progress_reward 而非独立加罚。