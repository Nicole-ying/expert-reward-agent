# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v1: 主进展信号 + 软着陆约束 + 碰撞预防 + 时间惩罚
    """
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]

    # 距离计算（欧几里得距离）
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（基于距离改善的差分奖励）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)  # 距离减小得正奖励

    # 2. 软着陆约束：接近垫子时惩罚高速和倾斜姿态
    w_soft = 0.5
    proximity_threshold = 0.3  # 当距离 < 0.3 时激活
    gate = max(0.0, 1.0 - next_dist / proximity_threshold)  # 线性衰减门控
    # 速度用绝对值，姿态用平方（抑制大幅倾斜）
    soft_penalty_intensity = abs(next_vx) + abs(next_vy) + body_angle ** 2
    penalty_soft = -w_soft * gate * soft_penalty_intensity

    # 3. 碰撞预防：水平越界惩罚
    x_bound = 0.9  # 视野假设边界
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    # 4. 碰撞预防：高速垂直坠落惩罚（低于垫子且下落过快）
    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:  # 低于垫子且速度向下超过阈值
        # next_vy 为负值，差值平方惩罚强度随速度增大而增加
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 5. 时间惩罚：推动快速完成任务
    living_penalty = -0.002

    total_reward = (reward_prox + penalty_soft +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "soft_landing_penalty": penalty_soft,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**:  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`。核心是快速、平稳着陆到目标垫，同时抑制不必要的推进消耗。

- **selected reward roles**:  
  1. **goal_proximity**（主学习信号）：每步基于距离变化的差分奖励，鼓励持续靠近目标垫。  
  2. **soft_landing**（稳定约束）：在接近垫子（距离 < 0.3）时，惩罚过大的水平/垂直线速度和机体倾斜，引导减速和姿态水平。  
  3. **crash_prevention**（安全约束）：对水平飞出视野（|x| > 0.9）和高速向下撞击（y < -0.1且垂直速度 < -0.5）施加二次惩罚。  
  4. **living_penalty**（效率代价）：极小固定负奖励，推动尽早完成着陆，避免无限悬停。

- **role_to_signal_mapping**:  
  - `goal_proximity`: `x_position(obs[0])`, `y_position(obs[1])` → 欧氏距离的差分 Δ = dist - next_dist。  
  - `soft_landing`: `next_x_velocity`, `next_y_velocity`, `next_body_angle` + `next_dist` 作为门控条件。使用线性衰减门控将约束限制在接近区域。  
  - `crash_prevention`: `next_x`, `next_y`, `next_vy` 用于边界和危险速度判定。  
  - `living_penalty`: 常数，不依赖信号。

- **每个 role 选择的 formula operator**:  
  - `goal_proximity`: **improvement_delta**（`dist - next_dist`），因为距离是单调进展度量，差分直接反映“前进还是后退”。  
  - `soft_landing`: **soft_health_gate** 结合 **quadratic_penalty** 的思路：用门控因子乘以速度和姿态惩罚项（速度线性，姿态二次），在危险区域渐进激活。  
  - `crash_prevention`: 水平越界用 **hinge penalty**（超出阈值部分的二次惩罚）；垂直坠落用 **speed_limit_penalty**（速度超过阈值时二次惩罚），均为越界即罚。  
  - `living_penalty`: **constant_penalty**。

- **excluded roles 及原因**:  
  - `fuel_efficiency`：推迟到 v2 加入，以免早期过度抑制探索。  
  - `orientation_stabilization`：条件角色，要求在接近且减速时激活；v1 的 soft_landing 已覆盖姿态惩罚，暂不额外拆出。  
  - `landing_success_bonus`：环境无显式成功标志，硬编码软成功条件存在被利用风险，v1 不用。  
  - `sparse_success_only`：无可用 signal，极度稀疏，禁止使用。

- **为什么没有 terminal_success_reward / terminal_failure_penalty**:  
  环境卡片明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，且禁止使用未声明的 `info` 字段。终止原因只能从观测推测，v1 选择用持续约束（软着陆、碰撞预防）代替硬事件奖励/惩罚。

- **留到后续迭代的职责**:  
  - 燃料效率（`fuel_efficiency`）：动作代价，在学会基本着陆后加入。  
  - 姿态平稳专项（`orientation_stabilization`）：若 early 阶段姿态振荡显著，可在 v2 中增强。  
  - 着陆成功软近似（`landing_success_bonus`）：如有需要且能可靠避免误判，可谨慎加入。

- **训练后应观察的 failure modes**:  
  - 长时间悬停、微小推力摆动（episode_length 大）：说明 living_penalty 或主进展信号不足以驱动快速下降，可微调权重。  
  - 高速直接撞击目标垫（reward_prox 为正但 soft_landing 惩罚很低，因距离未进入阈值）：可能需要提早激活软着陆（增大 proximity_threshold 或改用连续 gate）。  
  - 水平飞出视野频繁：crash_prevention_x 门槛可能偏高，可降低 x_bound 并提前给出惩罚。  
  - 始终不使用主引擎，仅依赖方向引擎乱转：说明燃料效率尚未加入，但可能存在姿态过度旋转无法下落的问题；若出现，可在后续加入对无效动作的轻微抑制。