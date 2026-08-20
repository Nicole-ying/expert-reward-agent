# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前位置与目标垫距离
    x_curr = obs[0]
    y_curr = obs[1]
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5

    x_next = next_obs[0]
    y_next = next_obs[1]
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # 主进展信号：到目标的欧氏距离递减
    progress = dist_curr - dist_next

    # 下一步速度（用于约束）
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]

    # 速度硬约束：过快水平移动或过快下降
    x_speed_viol = max(0.0, abs(x_vel_next) - 0.8)
    # 假定 y_vel 向上为正，下降速度为 -y_vel，限制下降速度 ≤ 0.8
    y_speed_viol = max(0.0, -y_vel_next - 0.8)
    speed_penalty = x_speed_viol + y_speed_viol

    # 姿态与角速度稳定约束
    body_angle_next = next_obs[4]
    angular_vel_next = next_obs[5]
    angle_penalty = body_angle_next ** 2 + angular_vel_next ** 2

    # 软着陆接触奖励：双腿着垫且速度平稳时给予一次性正反馈
    left_next = next_obs[6]
    right_next = next_obs[7]
    contact_bonus = 0.0
    if left_next > 0.5 and right_next > 0.5 and abs(x_vel_next) < 0.3 and abs(y_vel_next) < 0.3:
        contact_bonus = 1.0

    # 加权组合
    total = (
        10.0 * progress
        - 1.0 * speed_penalty
        - 0.5 * angle_penalty
        + 2.0 * contact_bonus
    )

    components = {
        "progress": 10.0 * progress,
        "speed_penalty": -1.0 * speed_penalty,
        "angle_penalty": -0.5 * angle_penalty,
        "contact_bonus": 2.0 * contact_bonus
    }

    return float(total), components
```

# reward_v1 设计说明

- **任务家族与动力学子类型**  
  `navigation_goal_reaching`，子类型 `goal_approach_and_soft_contact`。  
  目标：使 2D 着陆器从高处飞向中央目标垫，并实现软着陆（姿态平稳、速度极小、双腿同时接触垫面）。

- **选用的 reward roles**  
  1. **主学习信号 – 距离缩减（progress）**  
     职责：引导 agent 接近目标垫。选择 `improvement_delta` 算子，用欧氏距离 `dist = (x² + y²)^0.5` 的前后差作为每步的正奖励。  
  2. **稳定/安全约束 – 速度限制 & 姿态角/角速度惩罚**  
     职责：防止水平漂移过大、下降过快（坠毁），维持身体姿态。  
     速度用 hinge 形式（`max(0, abs(x_vel)-0.8)`、`max(0, -y_vel-0.8)`），只在超出安全阈值时惩罚。  
     姿态角与角速度用二次惩罚（`body_angle² + angular_vel²`）进行轻量抑制。  
  3. **任务完成近似信号 – 双腿接触且速度平稳时的正反馈（contact_bonus）**  
     职责：在无显式 termination success flag 的情况下，通过连续条件组合（双腿接触 + 水平垂直速度均小）给予一次性软着陆奖励，帮助 agent 识别最终成功状态。

- **角色到信号的映射**  
  - progress ← `obs[0], obs[1], next_obs[0], next_obs[1]` → 距离  
  - speed_penalty ← `next_obs[2], next_obs[3]` → x_vel, y_vel  
  - angle_penalty ← `next_obs[4], next_obs[5]` → body_angle, angular_vel  
  - contact_bonus ← `next_obs[6], next_obs[7]` + 速度条件  

- **使用的公式算子**  
  - `improvement_delta` → progress 主信号  
  - `dense_state_signal` 中的 **hinge** → 速度越界惩罚  
  - `quadratic_penalty` → 姿态/角速度约束  
  - `joint_condition_proxy`（简化乘性变体）→ 双腿接触 + 低速条件，形成任务完成近似信号  

- **被排除的 roles 及原因**  
  - `terminal_success_reward` / `terminal_failure_penalty`：  
    环境未提供显式成功/失败标志（`explicit_success_flag_available=false`，info 为空），且终止原因无法可靠区分（可能是软着陆成功，也可能是坠毁或出界），v1 阶段不使用硬覆盖事件奖励。  
  - `action_efficiency`（动作消耗）：离散动作空间，且首要任务为学会软着陆，效率优化留到后续迭代。  
  - `soft_health_gate`、`dynamic_curriculum`：v1 保持简单，先建立基本飞向目标并减速的能力。  
  - `potential_based_shaping`：已通过 simpler 的 `improvement_delta` 直接表达，无需额外构造 potential。

- **为什么未使用 terminal 奖励**  
  无显式标志且无法可靠推断成功/失败，硬覆盖奖励可能误导成功终止被惩罚或失败终止被奖励。因此依赖连续 progress 信号驱动，接触奖励仅作为正向提示。

- **后续迭代预留的职责**  
  - 动作效率/能量消耗约束（如需减少发动机使用）。  
  - 更精细的 soft_health_gate（如姿态恶化时暂时衰减 progress 奖励）。  
  - 动态课程（例如根据不同高度调整速度阈值）。  
  - 利用 angular_vel 更精细的稳定控制。

- **训练后需观察的 failure modes**  
  - Agent 长时间悬停而不下降（progress 停滞，speed_penalty 低但无进展）。  
  - 高速俯冲撞击垫面（speed_penalty 被触发但 agent 仍学到高 reward 回路，需检查惩罚权重）。  
  - 偏心接触即获得 contact_bonus 导致奖励作弊（例如单腿触发后另一腿立刻接触，但严格条件已包含双腿同时要求）。  
  - 角度过大引起 crash，angle_penalty 能否有效抑制。