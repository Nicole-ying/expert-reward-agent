# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 航向进展：距离目标越近越好（improvement_delta）
    d_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_prev - d_next
    goal_progress = 1.0 * progress

    # 2. 稳定停靠奖励：靠近目标时鼓励低速、竖直、双腿接触
    proximity_thresh = 0.5
    proximity_gate = max(0.0, 1.0 - d_next / proximity_thresh)

    # 速度分量：速度越低奖励越高
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    vel_thresh = 0.2
    velocity_bonus = 0.5 * max(0.0, 1.0 - speed / vel_thresh)

    # 姿态角分量：角度越小奖励越高
    angle_thresh = 0.1  # 弧度
    angle_bonus = 0.2 * max(0.0, 1.0 - abs(next_obs[4]) / angle_thresh)

    # 接触分量：双腿均接触才有奖励
    contact_bonus = 1.0 * next_obs[6] * next_obs[7]

    stable_bonus = proximity_gate * (velocity_bonus + angle_bonus + contact_bonus)

    # 3. 燃料效率惩罚：使用任何引擎都会受到微小惩罚
    fuel_penalty = -0.01 if action != 0 else 0.0

    total_reward = goal_progress + stable_bonus + fuel_penalty
    components = {
        'goal_progress': float(goal_progress),
        'stable_bonus': float(stable_bonus),
        'fuel_penalty': float(fuel_penalty)
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family**：`navigation_goal_reaching`（主要目标：到达并稳定停靠在目标平台）
- **selected dynamics_subtype**：`goal_approach_and_soft_contact`（离散控制，平面着陆器形态）

**selected reward roles**（来自 environment_card 的 reward_role_decomposition）：

1. **goal_progress_delta_distance**（主学习信号，mandatory）  
   - **公式算子**：`improvement_delta`，计算上一帧与下一帧到目标距离的差值。  
   - **信号映射**：`distance = sqrt(obs[0]^2 + obs[1]^2)` → `progress = distance_prev - distance_next`。  
   - **作用**：驱动飞行器持续向目标靠近，每一步靠近都获得正奖励，远离则受到惩罚。

2. **stable_contact_and_low_velocity_bonus**（稳定/安全约束，mandatory）  
   - **公式算子**：`joint_condition_proxy` 变形 —— 用 proximity gate 乘 速度、角度、接触三个分量的和，避免早期乘积塌缩，同时保留各部件的梯度。  
   - **信号映射**：  
     - `proximity_gate` = max(0, 1 - distance/0.5)  
     - `velocity_bonus` = 0.5 * max(0, 1 - speed/0.2)（speed = sqrt(vx^2+vy^2)）  
     - `angle_bonus` = 0.2 * max(0, 1 - |body_angle|/0.1)  
     - `contact_bonus` = 1.0 * left_contact * right_contact  
   - **作用**：当飞行器进入目标附近（距离 < 0.5）时，额外鼓励降低速度、摆正姿态、并使双腿接触。速度/角度分量提供连续梯度，接触分量提供稀疏但强烈的最终信号。

3. **fuel_efficiency_penalty**（效率/代价，conditional）  
   - **公式算子**：`action_efficiency`（离散动作的微小固定代价）。  
   - **信号映射**：`action != 0` 时施加 -0.01。  
   - **作用**：鼓励减少不必要的引擎使用，同时不严重影响主任务的探索。

**excluded roles 及原因**：  
- **terminal_success_bonus**：环境未提供显式成功标志（`explicit_success_flag_available=false`），且我们的 `stable_bonus` 已通过连续逼近方式覆盖了最终停靠的条件，无需额外的大稀疏奖励。  
- **terminal_failure_penalty**：无可靠、低误报的失败推断信号，且 `goal_progress` 已通过距离 delta 对远离行为给出负向信号，v1 不引入高风险的硬边界惩罚。

**为什么没有使用 terminal event**：  
需要明确、可推断的失败/成功状态，而本环境的终止由多种原因混合且在 info 中无任何标识，仅靠观测猜测可能存在较高误报率。通过稠密化的进度奖励和稳定奖励，agent 已能区分好坏行为，稀疏终止信号可在后续迭代中谨慎加入。

**留到后续迭代的职责**：  
- 更精确的边界检测与出界惩罚（需要视口坐标阈值）。  
- 主引擎与姿态引擎的差异化燃料代价（当前统一为 -0.01）。  
- 接触信号微分（如接触力强度，不可用）。  
- 基于 landing pad 区域的多阶段门控（当前只用一个 proximity gate）。

**训练后应观察的 failure modes**：  
- **hover 或 stand still**：agent 停在远处靠零动作维持较小距离，但无进展——`goal_progress` 将衰减为零，而 `stable_bonus` 因其距离大被 gate 抑制，总奖励趋于零，应观察是否陷入局部惰性。  
- **velocity burst then fall**：agent 依靠短时大推力冲向目标，随后因无法减速而触地弹起或飞出——此时 `goal_progress` 为正（冲近了），但后续 `stable_bonus` 会因为速度大、姿态差而极低，整体 episode 回报应很差，需检查是否出现短期奖励欺骗。  
- **insufficient contact**：agent 学会了接近目标并减速，但始终不能同时接触双腿——`contact_bonus` 始终为 0，episode 可能因 timeout 结束。需观察双腿接触信号是否随策略变化逐渐增加。  
- **燃料极小导致动作过少**：燃料惩罚可能抑制必要的姿态调整，若策略长期无法稳定，可考虑在后续迭代中降低惩罚系数或改为只在成功回合才施加代价。