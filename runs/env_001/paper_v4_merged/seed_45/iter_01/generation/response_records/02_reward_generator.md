# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    angvel = obs[5]
    # left_contact, right_contact are not used, but can be extracted for future use
    left_contact = obs[6]
    right_contact = obs[7]

    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # Distance to target center at (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-6
    next_dist = (next_x**2 + next_y**2)**0.5 + 1e-6

    # 1. Main progress signal: improvement in distance (delta)
    progress_delta = dist - next_dist

    # 2. Orientation stability penalty (hinge)
    angle_threshold = 0.3   # rad (~17 deg)
    angvel_threshold = 0.5  # rad/s
    angle_violation = max(0.0, abs(next_angle) - angle_threshold)
    angvel_violation = max(0.0, abs(next_angvel) - angvel_threshold)
    orientation_penalty = -0.1 * angle_violation - 0.05 * angvel_violation

    # 3. Speed safety penalty (hinge on linear velocity components)
    speed_threshold = 0.5   # m/s units assumed
    vx_violation = max(0.0, abs(next_vx) - speed_threshold)
    vy_violation = max(0.0, abs(next_vy) - speed_threshold)
    speed_penalty = -0.05 * (vx_violation + vy_violation)

    total_reward = progress_delta + orientation_penalty + speed_penalty

    components = {
        'progress_delta': progress_delta,
        'orientation_penalty': orientation_penalty,
        'speed_penalty': speed_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动态子类型
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`（2D 月球着陆器）
- **控制类型**: discrete（4 种动作）
- **形态**: 双腿着陆器，配有两个姿态发动机和一个主发动机
- **主要目标**: 到达并稳定停靠在目标平台（(0,0)），同时节省燃料
- **观测空间**: 8 维（位置、速度、姿态、角速度、双腿接触标志）
- **显式成功/失败标志**: 无

## 2. 选用的奖励职责
从环境卡片中拆解的职责中，`v1` 版本选择了以下角色：

| 角色 | 用途 | 可用信号 |
|---|---|---|
| `progress_toward_target` | 主学习信号，鼓励每一步减少到目标点的距离 | `obs[0:2]` (x,y) → `distance` |
| `orientation_stability` | 安全约束，防止身体倾角过大和角速度过高 | `obs[4]` (body_angle), `obs[5]` (angular_velocity) |
| `speed_safety` | 安全约束，抑制过大的水平/垂直线速度，防止硬碰撞 | `obs[2]` (x_velocity), `obs[3]` (y_velocity) |

## 3. 职责‑信号映射与公式算子
- **progress_toward_target**:  
  使用 `improvement_delta` 算子：`reward = old_distance - new_distance`。  
  距离定义为 `(x² + y²)^0.5 + 1e‑6`，每一步距离减少即获得正奖励，驱动机器人向目标移动，同时避免悬停在原地。  
  算子选择依据：`improvement_delta` 迫使 agent 持续改善位置，而不是单纯奖励“靠近”的绝对值，可防止 agent 在远处滞留。

- **orientation_stability**:  
  使用 `dense_state_signal` (hinge) 形式：`penalty = -w_angle * max(0, |angle| - θ_th) - w_angvel * max(0, |angvel| - ω_th)`。  
  阈值设置：倾角阈值 0.3 rad，角速度阈值 0.5 rad/s。这些阈值给出的“安全区”允许小幅摆动，只在越界时给予线性惩罚，相比全时二次惩罚更能容忍正常机动。

- **speed_safety**:  
  同样采用 hinge 惩罚：`penalty = -w_v * (max(0, |vx| - v_th) + max(0, |vy| - v_th))`，速度阈值 0.5 m/s。  
  该约束防止 agent 以过高速度撞击平台，同时不影响低速接近行为。权重较小（0.05），不会压制主进展信号。

## 4. 排除的职责及原因
- **terminal_success_reward / terminal_failure_penalty**:  
  环境无显式成功/失败标志，且 `info` 字段完全为空。当前版本避免从观测人工构造硬终端信号（如组合条件推断），以免引入脆弱的阈值。后续若训练中 agent 无法区分成功终止与失败终止，可考虑引入基于多条件的 `terminal_event` 算子。
- **fuel_efficiency**:  
  虽然环境目标中包含“省燃料”，但 v1 的重点是建立安全的到达与着陆行为。过早加入燃料惩罚可能抑制必要的发动机使用，阻碍 agent 学会飞向目标。燃料成本优化将留给后续迭代。
- **contact_proxy**:  
  双腿接触传感器暂时未使用。`progress_delta` 结合速度惩罚已能引导 agent 下降到平台并经缓冲后自然触发接触。若实验中出现悬停不接触或单脚着陆，未来可增加一个轻量 `soft_landing_proxy`。

## 5. 设计自检
- **终止条件是否有前兆软信号？**  
  是。速度过快（`speed_penalty` hinge）和姿态失稳（`orientation_penalty` hinge）在接近硬边界前提供连续梯度，引导 agent 及时减速和摆正姿态。
- **任务目标是否有直接的进度信号？**  
  是。每一步的距离减小量构成稠密的进步反馈。
- **动作维度 ≥ 6 时的效率约束？**  
  本环境动作为离散 4 维，不影响此规则。
- **是否避免了重写原始奖励、使用未声明字段等问题？**  
  代码完全不依赖 `original_reward`、`info` 或任何未声明的切片，完全基于 `obs` 和 `next_obs`。

## 6. 后续迭代（v2/v3）应考虑的职责
- **燃料消耗惩罚**：引入轻量 `action_efficiency` 算子（如 `-0.005 * (action != 0)`），在稳定行为建立后抑制不必要的发动机点火。
- **接触奖励 / 着陆代理**：若 agent 出现单脚着陆或悬停不接触，可加入基于连续条件的 `joint_condition_proxy` 奖励。
- **速度与角度阈值的动态调整**：根据训练失败模式，可能需要对 `orientation_penalty` 的阈值进行课程式收紧。
- **soft_health_gate**：如果 agent 出现“先冲后死”模式（进步信号为正但仍导致失败），可使用姿态/速度来衰减主奖励。

## 7. 训练后应观察的失败模式
- **低速悬停**：进度奖励不足，agent 可能为了避免速度惩罚而过度减速，导致无法到达目标。可通过提高 `progress_delta` 权重或微调速度阈值解决。
- **高速撞击**：速度阈值设置过高，agent 在高速下仍不受罚，可能 crash。需降低阈值或增加惩罚系数。
- **翻滚/大角度振荡**：姿态惩罚过弱，agent 可能利用大角度机动，导致单脚着陆或翻倒。可增加 `orientation_penalty` 权重。
- **停滞在平台上方未接触**：距离已接近 0 但 episode 未终止，可能因未触发接触而持续消耗燃料。若发生，需引入接触奖励。
