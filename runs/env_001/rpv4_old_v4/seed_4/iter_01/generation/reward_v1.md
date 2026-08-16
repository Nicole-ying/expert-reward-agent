# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v1。
    核心目标：尽快到达中心目标垫板并稳定停靠，同时鼓励双支撑腿安全着陆。
    
    Args:
        obs: 当前观测 (8,) [x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact]
        action: 当前动作 (0-3: no_engine, left_orient, main_engine, right_orient)
        next_obs: 下一观测 (8,)
        original_reward: 原始奖励（不使用）
        info: 信息字典（为空，不使用）
        training_progress: 训练进度（v1不使用）
    
    Returns:
        float: 总奖励
        dict: 各奖励组件
    """
    
    # ========== 信号提取 ==========
    # 当前状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    # 下一状态
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    # ========== Component A: 主学习信号 - 目标接近与稳定 ==========
    # role: goal_proximity + settling_stability
    # 策略：使用 improvement_delta 鼓励朝目标方向移动并减速
    # 水平方向：减少 |x_pos|，理想为 0（对准垫板中心）
    # 垂直方向：减少 |y_pos|，理想为 0（贴齐垫板表面）
    # 同时鼓励速度归零（settling），权重随接近目标而升高
    
    # 当前位置距离（L1范数，简单有效）
    current_distance = abs(x_pos) + abs(y_pos)
    next_distance = abs(next_x_pos) + abs(next_y_pos)
    
    # 位置改进奖励：距离减少得正奖励（improvement_delta）
    distance_improvement = current_distance - next_distance
    
    # 速度惩罚：鼓励减速，尤其在接近目标时
    current_speed = abs(x_vel) + abs(y_vel)
    next_speed = abs(next_x_vel) + abs(next_y_vel)
    
    # 速度改进：速度降低得奖励（越接近目标，权重越高）
    # 使用 gating: 距离越近，速度惩罚越重要
    proximity_gate = 1.0 / (1.0 + 5.0 * current_distance)  # 距离远时≈0，距离0时≈1
    speed_improvement = current_speed - next_speed
    
    # 组合：位置改进为主驱动，速度改进为辅约束
    goal_proximity_reward = 2.0 * distance_improvement  # 主导航信号
    settling_reward = 0.5 * speed_improvement * (1.0 + proximity_gate * 3.0)  # 接近时加强速度控制
    
    # ========== Component B: 姿态稳定约束 ==========
    # role: upright_orientation (条件激活) + 部分 settling_stability
    # 仅当接近目标时激活角度惩罚，避免干扰大角度机动
    
    # 当前和下一姿态误差
    current_angle_error = abs(body_angle)
    next_angle_error = abs(next_body_angle)
    current_ang_vel_mag = abs(ang_vel)
    next_ang_vel_mag = abs(next_ang_vel)
    
    # 姿态改进：角度减小、角速度减小得奖励
    angle_improvement = current_angle_error - next_angle_error
    ang_vel_improvement = current_ang_vel_mag - next_ang_vel_mag
    
    # 仅当接近目标时施加姿态约束（y_pos < 0.5 且 distance < 1.0）
    attitude_gate = 1.0 if (abs(y_pos) < 0.5 and current_distance < 1.0) else 0.1
    attitude_reward = attitude_gate * (1.0 * angle_improvement + 0.5 * ang_vel_improvement)
    
    # ========== Component C: 安全着陆接触奖励 ==========
    # role: safe_landing_contact
    # 当接近目标、速度低、且双支撑腿接触时给予正向奖励
    # 使用 joint_condition_proxy 构造软接触成功信号
    
    # 接近目标条件
    near_target = abs(next_x_pos) < 0.2 and abs(next_y_pos) < 0.2
    
    # 低速条件（L1速度小于阈值）
    low_speed = (abs(next_x_vel) + abs(next_y_vel)) < 0.5
    
    # 双接触条件
    both_contacts = (next_left_contact > 0.5 and next_right_contact > 0.5)
    
    # 接触改进：从单边或无边到双接触
    contact_improvement = 0.0
    if both_contacts and not (left_contact > 0.5 and right_contact > 0.5):
        contact_improvement = 0.5  # 新获得双接触
    
    # 组合门控：仅当接近且低速时，接触才有意义
    if near_target and low_speed:
        landing_reward = 3.0 * float(both_contacts)  # 鼓励维持双接触
        landing_reward += 2.0 * contact_improvement  # 额外奖励首次双接触
    else:
        landing_reward = 0.0
    
    # ========== Component D: 燃料效率惩罚（轻量） ==========
    # role: fuel_efficiency
    # 对使用引擎施加轻微惩罚，避免无意义动作
    # action != 0 表示使用了引擎（离散动作空间）
    fuel_penalty = -0.05 * float(action != 0)
    
    # ========== 汇总奖励 ==========
    total_reward = (
        goal_proximity_reward +
        settling_reward +
        attitude_reward +
        landing_reward +
        fuel_penalty
    )
    
    components = {
        'goal_proximity': goal_proximity_reward,
        'settling_stability': settling_reward,
        'attitude_constraint': attitude_reward,
        'landing_contact': landing_reward,
        'fuel_efficiency': fuel_penalty
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像确认
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **control_type**: discrete (4 个离散动作)
- **核心目标**: 到达目标垫板并稳定停靠（位置对齐、速度归零、姿态平稳、安全接触）
- **次要目标**: 节油（轻量约束）

## 2. 选定的奖励职责
基于 environment_card.md 的 `reward_role_decomposition`，v1 覆盖了所有 mandatory roles 和高优先级的 conditional roles：

| role_id | status | formula operator | rationale |
|---------|--------|------------------|-----------|
| **goal_proximity** | **mandatory（主学习信号）** | `improvement_delta` (Component A) | 使用 `distance_improvement = current_distance - next_distance` 作为核心驱动信号。每步都有梯度，直接关联"接近目标"的任务目标。选择 delta 而非 dense_state_signal 是因为：位置绝对值不能反映进展（agent 可能在不同位置来回移动），只有变化方向能明确引导。 |
| **settling_stability** | **mandatory** | `improvement_delta` + `soft_health_gate` (Component A) | 速度改进 `current_speed - next_speed` 配合 proximity gate（接近目标时权重升高）。这是"软约束"而非硬惩罚，允许快速接近但要求在靠近时减速。使用 delta 而非速度绝对值是因为：鼓励速度降低的行为变化，而非仅仅惩罚高速度状态。 |
| **upright_orientation** | **conditional** | `improvement_delta` + `bounded_signal` gate (Component B) | 仅当接近目标（`|y_pos| < 0.5` 且 `distance < 1.0`）时激活。使用 angle_improvement 和 ang_vel_improvement 的 delta 形式，鼓励姿态不断改善而非仅保持。gate 防止早期不必要的大角度约束压制机动的探索。 |
| **safe_landing_contact** | **conditional** | `joint_condition_proxy` + `gated_signal` (Component C) | 使用双接触 `both_contacts` 作为核心信号，用 `near_target` 和 `low_speed` 作为门控条件。这是"任务完成软近似"——在无显式 success flag 的情况下，通过多条件组合提供完成信号。同时奖励首次双接触（`contact_improvement`）以鼓励探索着陆行为。 |
| **fuel_efficiency** | **conditional** | `action_efficiency`（离散形式）(Component D) | 对每个非零动作施加极小惩罚（-0.05）。权重远小于主导航信号（2.0的distance_improvement），不会压制探索。v1 保持轻量，主要用于抑制纯随机的动作噪声。 |

## 3. excluded roles 及原因
| role_id | reason for exclusion |
|---------|---------------------|
| **explicit_success_bonus** | `info` 字典为空（`explicit_success_flag_available=false`），无法获取成功标志。强行使用会导致不可知行为。 |
| **sparse_reward** | 无法访问终止事件信息（`explicit_failure_flag_available=false`），稀疏奖励不可行。任务需要密集引导信号。 |
| **collision_penalty** | 环境中没有碰撞强度或碰撞事件的可观测信号，碰撞感知仅来自回合终止，不能用于步级奖励。在 v1 中，通过 `goal_proximity` 和 `settling` 间接抑制碰撞相关的错误行为（如高速冲击）。 |
| **terminal_event** | 虽然 formula operator library 提供此算子，但它依赖从观测推断终止条件（如极端姿态+接触地面）。v1 优先用连续的 approach+settling+attitude 信号引导，避免引入硬编码阈值和硬覆盖逻辑。若训练后 crash 率仍高，可在 v2 考虑。 |

## 4. 为什么没有 terminal_success_reward / terminal_failure_penalty
- **terminal_success_reward**: 环境中 `info` 为空，`explicit_success_flag_available=false`，无法在奖励函数内获知"任务是否成功完成"。所有终止信号由环境内部判定，奖励函数无法访问。转而使用 `safe_landing_contact` (Component C) 作为任务完成的**软近似信号**——通过观测中的连续信号（位置、速度、接触）构造，而非依赖不可访问的终止事件。
- **terminal_failure_penalty**: 同理，`explicit_failure_flag_available=false`，无法从 `info` 中区分 crash/出界等失败终止。v1 通过稠密约束（`settling_stability` 抑制高速冲击，`attitude_constraint` 抑制倾斜着陆）在步级先发预防失败，而非事后惩罚。

## 5. 信号冲突检查
- **避免重复计算速度**: `goal_proximity` 已考虑位置改进，`settling` 单独处理速度改进。两者信号正交（位置 vs. 速度），不冲突。
- **避免压制探索**: `fuel_efficiency` 权重极低（-0.05），远小于主导航信号（2.0）；`attitude_constraint` 仅在接近目标时激活（gate=0.1 远距离），不干扰早期机动；`settling` 用的 proximity_gate 在远距离时接近 0，不强纠偏。
- **landing_contact 门控安全**: 要求 `near_target` 且 `low_speed`，防止 agent 在远处故意单脚触地 hack 奖励。

## 6. 组件尺度与平衡
- **主驱动力**: `goal_proximity` (2.0 * distance_improvement)，每步量级约 -2.0 ~ +2.0（取决于移动方向）。
- **速度约束**: `settling` (0.5 * speed_improvement，接近时×4因子)，在接近时约 -2.0 ~ +2.0，不会在数值上压倒位置奖励。
- **姿态约束**: `attitude` (1.0 * angle_improvement + 0.5 * ang_vel_improvement)，仅在接近时激活，量级约 -1.0 ~ +1.0。
- **着陆奖励**: `landing` (3.0 * both_contacts + 2.0 * contact_improvement)，是稀疏但重要的正向信号，额度约 0 ~ 5.0，不会导致刷分（需满足多重条件）。
- **燃料惩罚**: `fuel_penalty` (-0.05)，几乎不影响步级奖励，但在长 episode 中积累约 -5.0 ~ -10.0（100-200步），作为轻量激励。

所有组件在数值上大致可比，不出现单一组件完全支配 total_reward 的情况。使用 `improvement_delta` 形式使得奖励的步间变化尺度和位置/速度本身的量级解耦——agent 不会被绝对位置的量级限制。

## 7. 留到后续迭代的职责
- **fuel_efficiency 调度**: v1 使用固定小权重。若训练后观察到 agent 燃料浪费严重但任务完成率高，v2 可引入 `curriculum_weighting`（随 training_progress 线性增加燃料惩罚权重）。
- **explicit crash avoidance**: v1 通过稠密 settling 和 attitude 间接预防 crash。若 crash 率仍然高（如 >30%），v2 可考虑对极端速度/姿态施加 `hinge` 惩罚（如 `|y_vel| > 2.0` 时 `-w * (|y_vel| - 2.0)`）。
- **time-to-target 优化**: 当前未显式奖励快速到达。若训练后 agent 过度减速（hover），可加入轻量的 per-step existence penalty 或 progress timer。
- **complex multi-stage gating**: 当前使用简单的 proximity gate 和 near-target gate。若出现"接近目标但无法稳定"的阶段转移问题，可引入更细粒度的 multi-stage gated_reward。

## 8. 训练后应观察的 failure modes
基于 environment_card.md 的 failure_modes 表格：

| failure_mode | expected evidence | component adjustment hint |
|-------------|-------------------|--------------------------|
| **hover without approaching** | `distance` 不下降，`distance_improvement ≈ 0` | 增大 `goal_proximity` 权重（2.0→3.0）；减小 `fuel_penalty`（-0.05→-0.02）；考虑在 y-direction 加入单独的位置改进奖励。 |
| **fly out of bounds** | `|x_pos|` 持续增大，episode 突然终止 | 增大横向偏差的隐式惩罚（当前没有显式项，依赖总距离的 L1 范数；可考虑为 `|x_pos|` 单独加 hinge）；检查 `attitude_constraint` gate 是否过早激活导致大幅旋转。 |
| **crash at high speed** | `|y_vel|` 在 `|y_pos| < 0.5` 时仍 >2.0 | 增大 `proximity_gate` 的系数（当前 `k=5` → `k=10`），使速度惩罚在更远距离就开始激活；或对 `|y_vel|` 单独施加 hinge penalty（v2）。 |
| **waste excessive fuel** | 动作非零率 >80%，但位置无明显改善 | 增大 `fuel_penalty`（-0.05→-0.1）；或改为只惩罚无进展时的动作（需 tracking progress，v2）。 |
| **oscillate without settling** | `body_angle` 或 `x_pos` 在 `near_target` 状态下来回摆动 | 提升 `attitude_reward` 的 gate 强度（gate=0.1→0.5 远距离时仍轻微约束）；增大 `ang_vel` 惩罚系数（0.5→1.0）。 |
| **always no-op and fall** | action=0 频率 >90%，直接坠落 crash | `fuel_penalty` 过大压制了引擎使用；降低至 -0.02；确保 `goal_proximity` 的正向奖励能驱动 agent 点火。 |

## 9. 设计总结
本 v1 奖励函数遵循 **role-based component budget**：
- **1 个主学习信号**（`goal_proximity`，使用 `improvement_delta`，2.0 权重）
- **2 个稳定/安全约束**（`settling` 速度约束 + `attitude` 姿态约束，使用带 gate 的 delta 形式）
- **1 个任务完成近似信号**（`landing_contact`，使用 `joint_condition_proxy` + `gated_signal`）
- **1 个轻量效率代价**（`fuel_penalty`，使用离散 `action_efficiency`，0.05 权重）

总计 5 个组件，覆盖了所有 mandatory roles 和最关键的条件职责。所有计算仅使用环境卡片中声明的 obs/next_obs 信号，不依赖 info 字典或未文档化的观测。奖励函数以 `improvement_delta` 为主要数学形式，提供每步梯度，避免 agent 在好状态上停滞。约束项使用 gate 机制防止在错误阶段压制探索。整体设计为后续迭代留有清晰的升级路径。