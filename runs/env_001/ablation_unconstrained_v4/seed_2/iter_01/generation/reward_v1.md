# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    
    w_progress = 1.0
    w_vel = 0.5
    w_att = 0.5
    w_contact = 1.0
    d_close = 2.0
    vx_max = 0.5
    vy_max = 0.5
    angle_max = 0.2
    angvel_max = 0.2
    d_land = 1.0
    
    # 1. 向目标移动的稠密进展奖励 (improvement_delta)
    progress_reward = w_progress * (dist_old - dist_new)
    
    # 2. 接近目标时速度约束 (hinge penalty with gate)
    gate_speed = max(0.0, 1.0 - dist_new / d_close)
    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * gate_speed * (excess_vx**2 + excess_vy**2)
    
    # 3. 接近目标时姿态稳定性约束 (hinge penalty with gate)
    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * gate_speed * (excess_angle**2 + excess_angvel**2)
    
    # 4. 软着陆接触奖励 (continuous closeness × contact average)
    closeness_land = max(0.0, 1.0 - dist_new / d_land)
    avg_contact = (left_contact + right_contact) / 2.0
    contact_bonus = w_contact * avg_contact * closeness_land
    
    total = progress_reward + vel_penalty + att_penalty + contact_bonus
    components = {
        'progress_reward': progress_reward,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty,
        'contact_bonus': contact_bonus
    }
    return float(total), components
```

# reward_v1 设计说明

- **任务族/动力学子类型**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`（2D 离散推力飞行器，到达并柔软着陆于中心目标垫）。

- **选择的责任角色（roles）**  
  1. `progress_to_target`（必需）：鼓励向目标移动。  
  2. `velocity_dampening`（必需）：接近目标时降低线速度，防止高速撞击。  
  3. `attitude_stability`（条件安全约束）：接近目标时限制大角度和大角速度，保证平稳姿态。  
  4. `landing_contact_approximation`（任务完成近似信号）：引导飞行器在目标附近双腿着地，完成真实着陆。

- **职责‑信号映射**  
  - `progress_to_target` → `obs[0:2]`（水平、垂直位置）与 `next_obs[0:2]` → 距离改善。  
  - `velocity_dampening` → `next_obs[2:4]`（水平、垂直线速度）→ 速度超出阈值时惩罚，并由距离门控。  
  - `attitude_stability` → `next_obs[4:6]`（机体角度、角速度）→ 超出阈值时惩罚，同样由距离门控。  
  - `landing_contact` → `next_obs[6:8]`（左、右腿接触）与 `next_obs[0:2]`（位置）→ 接触平均值 × 接近程度。

- **公式算子选择**  
  - `progress_to_target`：**improvement_delta**（`old_measure - new_measure`），每步提供稠密梯度，鼓励持续减少到目标距离。  
  - `velocity_dampening`：**dense_state_signal 的 hinge 形式**（`max(0, abs(v_i) - v_max)^2`）与 soft gate 相乘，仅在接近时生效，避免远距离无谓减速。  
  - `attitude_stability`：同样使用 **hinge penalty** + gate，只在接近时约束，避免早期压制探索。  
  - `landing_contact`：**joint_condition_proxy 的朴素连续化**（连续 closeness × 接触平均），把二元接触信号变为每步可用的 soft proxy。

- **排除的角色及原因**  
  - `fuel_efficiency_minimization`：v1 优先学会任务方向和软着陆，燃料优化留待后续迭代。  
  - `terminal_success_reward` / `terminal_failure_penalty`：环境中无显式 success/failure flag，且 `compute_reward` 不接收 `done` 信号，故不可使用。  
  - `gated_reward` 或 `dynamic_curriculum_reward`：当前无训练进度依赖，v1 不需要。  
  - `action_smoothness_penalty`：无可用的上一动作历史，无法实现。  
  - `boundary_penalty`：缺少水平边界坐标信息，无法直接惩罚出界风险。

- **无 terminal 奖励的原因**  
  环境卡片明确 `explicit_success_flag_available=false`、`explicit_failure_flag_available=false`，且 `info` 为空字典，同时函数签名不包含 `done` 参数。因此无法添加基于 episode 成功/失败的即期奖励，所有奖励必须完全由 `obs`、`next_obs` 构建。

- **留待后续迭代的职责**  
  - 燃料消耗 / 动作代价（调用主引擎或姿态引擎时的惩罚）。  
  - 更精细的着陆判断（如要求角速度几乎为零才给予接触 bonus）。  
  - 基于距离的 dynamic curriculum（早期更注重接近，后期更注重稳定性）。  
  - 对飞出边界的明确惩罚（需环境提供边界信息或通过终止行为推断）。  
  - soft_health_gate：若未来训练中出现“先冲后死”模式（高 progress 但其间 crash），可将姿态或速度指标转化为门控乘到 progress 上。

- **训练后应观察的典型失效模式**  
  - 