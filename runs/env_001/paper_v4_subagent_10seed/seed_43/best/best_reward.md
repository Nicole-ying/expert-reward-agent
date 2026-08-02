# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]

    # Distance to target
    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5
    delta = dist_old - dist_new

    # Health gate: based on body angle and speed
    angle_healthy = 1.0 / (1.0 + 2.0 * angle**2)
    speed = abs(vx) + abs(vy)
    speed_healthy = 1.0 / (1.0 + 0.5 * speed)
    gate = angle_healthy * speed_healthy

    # Progress reward
    w_progress = 3.0
    progress_reward = w_progress * max(0.0, delta) * gate

    # Contact success bonus
    contact_reward = 0.0
    if left_contact == 1.0 and right_contact == 1.0:
        x_thresh = 0.5
        y_thresh = 0.5
        v_thresh = 1.0
        angle_thresh = 0.5

        closeness = max(0.0, 1.0 - abs(x)/x_thresh) * max(0.0, 1.0 - y/y_thresh)
        stability = max(0.0, 1.0 - (abs(vx) + abs(vy))/v_thresh) * max(0.0, 1.0 - abs(angle)/angle_thresh)
        w_contact = 5.0
        contact_reward = w_contact * closeness * stability

    # Angular velocity penalty (hinge)
    angvel_limit = 0.5
    w_angvel = 0.5
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_limit)

    total = progress_reward + contact_reward + angvel_penalty

    components = {
        'progress': progress_reward,
        'contact_success': contact_reward,
        'angvel_penalty': angvel_penalty
    }
    return float(total), components
```

# reward_v1 设计说明

**selected task_family / dynamics_subtype**  
`navigation_goal_reaching` / `goal_approach_and_soft_contact`（2D 着陆器，双腿软着陆）

**selected reward roles**

1. **主学习信号 – approach_progress**  
   使用 `improvement_delta` 算子计算每一步到目标着陆垫的欧氏距离减少量（`delta = dist_old − dist_new`），并且通过 `soft_health_gate` 衰减：只有当身体姿态足够竖直（`angle_healthy` 因子）且线速度适中（`speed_healthy` 因子）时，进度奖励才能充分获得。这迫使 agent 在安全、受控的状态下接近目标，而非“先冲后摔”。  
   算子和证据：`improvement_delta` + `soft_health_gate`，避免 proximity 悬停，同时压制不安全行为。

2. **允许约束 – 接触成功信号（软代理）**  
   当双腿同时接触时（`left_contact == 1` 且 `right_contact == 1`），给予基于位置接近度、低速和竖直姿态的乘积奖励（`joint_condition_proxy` 形式）。该组件提供最终着陆阶段的精确目标信号，弥补进度奖励在接近终点时会趋近于零的梯度缺失。  
   算子：`joint_condition_proxy`（连续乘积），每个因子为 `bounded_signal` 形式 `max(0, 1 − offset/threshold)`。

3. **允许约束 – 角速度惩罚**  
   使用 `dense_state_signal` 中的 hinge 惩罚，对过大的角速度（`|angvel| > lim`）施加轻量负奖励，抑制高频旋转和不稳定振荡。

**role‑to‑signal mapping**  
- `approach_progress` ← `obs[0] (x)`, `obs[1] (y)`, `obs[4] (angle)`, `obs[2:3] (vx,vy)`  
- `contact_success` ← `obs[0:1]`, `obs[2:3]`, `obs[4]`, `obs[6:7]`  
- `angvel_penalty` ← `obs[5]`

**excluded roles**  
- `terminal_success_reward` / `terminal_failure_penalty`：环境中**无显式成功/失败标志**且 `info` 为空，无法可靠区分终止原因。  
- `action_efficiency`：v1 聚焦任务达成与安全着陆，能耗优化留到后续迭代。  
- `dynamic_curriculum` / `gated_reward`：缺乏训练进度参考，过早引入会增加复杂度并可能压制探索。

**设计决策要点**  
- 不使用原始环境奖励 (`original_reward`)。  
- 不使用任何未声明的 `info` 字段。  
- 所有组件均基于环境卡片中明确列出的观测维度，且符合离散动作空间（不需要动作平滑等连续动作特化算子）。  
- 乘在进度奖励上的 `gate` 避免了独立惩罚“不敢行动”的问题：agent 通过改善自身姿态和速度即可恢复进度得分，而不是因为被罚而畏缩。  
- 接触成功奖励仅在双腿触地时激活，防止 agent 在单腿或翻倒时获得正向信号。

**留到后续迭代的职责**  
- 燃料/动作效率优化（`action_efficiency`）  
- 更精细的姿态/速度课程调节（如距离相关约束）  
- 可能的终端事件推断（利用 episode 截断时的观测构造失败/成功回补）

**预期需观察的 failure modes**  
- 如果 `gate` 衰减过快，agent 可能放弃移动，呆在高处减少姿态惩罚——需观察初期是否出现悬停行为。  
- 若 `contact_success` 权重过高，可能导致 agent 在腿刚触地时便放弃稳定调整，引发单腿或翻转后的静止（需检查终止时双腿是否真正稳定接触）。  
- 角速度惩罚系数若过强，可能抑制必要的姿态修正动作，需监控 `angvel_penalty` 的 active rate 和最终姿态误差。