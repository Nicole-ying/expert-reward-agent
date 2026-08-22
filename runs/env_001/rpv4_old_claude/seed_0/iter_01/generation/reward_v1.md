# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - terminal_soft_landing_bonus: multi-condition proxy for successful landing
    """
    
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    # === Component A: goal_approach_progress (main learning signal) ===
    # Use potential-based shaping: reward improvement in distance to target
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance  # positive when approaching
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    # === Component B: velocity_brake_near_target (safety constraint) ===
    # Quadratic velocity penalty, scaled by proximity to encourage slow approach
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)  # approaches 1 when close to target
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    # === Component C: orientation_stability (safety constraint) ===
    # Quadratic penalty on angle deviation and angular velocity
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    # === Component D: terminal_soft_landing_bonus (task completion proxy) ===
    # Multi-condition check for successful landing at episode end
    # Only trigger when both contacts active + low velocity + small angle + near target
    both_contacts = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    low_velocity = velocity_magnitude < 0.3
    stable_angle = abs(next_angle) < 0.2
    near_target = next_distance < 0.5
    
    if both_contacts and low_velocity and stable_angle and near_target:
        components["terminal_soft_landing_bonus"] = 50.0
    else:
        components["terminal_soft_landing_bonus"] = 0.0
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务识别
- **selected task_family**: `navigation_goal_reaching`
- **selected dynamics_subtype**: `goal_approach_and_soft_contact`
- **任务核心**: 从随机扰动初始位置导航到目标垫中心，以低速、稳定姿态完成双接触软着陆

## 选中的奖励职责 (selected reward roles)

### 主职责 (mandatory, v1 必须覆盖)
1. **goal_distance_shaping** → 实现为 `goal_approach_progress`
2. **velocity_damping_near_target** → 实现为 `velocity_brake_near_target`
3. **orientation_stability** → 实现为 `orientation_stability`
4. **terminal_outcome_signal** → 实现为 `terminal_soft_landing_bonus`

### 条件职责 (后续迭代考虑)
- **contact_bonus**: 当前 v1 已通过 terminal_soft_landing_bonus 间接覆盖接触条件，不单独加入"首次接触奖励"避免诱导过早触地
- **engine_usage_penalty**: 次要效率目标，留待后续迭代当主任务学习稳定后再引入

## role_to_signal_mapping

| Role | Usable Signals | Selected Formula Operator |
|------|----------------|---------------------------|
| goal_distance_shaping | obs/next_obs 的 x_position, y_position | **improvement_delta** (potential-based shaping): `old_distance - new_distance` |
| velocity_damping_near_target | x_velocity, y_velocity, distance | **quadratic_penalty** + proximity scaling: `-w * proximity * velocity_magnitude^2` |
| orientation_stability | body_angle, angular_velocity | **quadratic_penalty**: `-w * (angle^2 + ang_vel^2)` |
| terminal_outcome_signal | next_obs 全部字段 + terminated 隐含信号 | **joint_condition_proxy**: 多条件乘积检查（接触+速度+角度+位置） |

## 为何没有使用 terminal_success_reward / terminal_failure_penalty

### terminal_success_reward 缺失原因
- `explicit_success_flag_available = false`
- 环境 `info = {}`，没有 `info["success"]` 字段
- 但任务确实需要终局信号，因此使用 **joint_condition_proxy** 从 `next_obs` 的多维状态组合近似判断"成功软着陆"：
  - 双接触点激活 (`next_left_contact > 0.5 and next_right_contact > 0.5`)
  - 低速 (`velocity_magnitude < 0.3`)
  - 稳定姿态 (`abs(next_angle) < 0.2`)
  - 靠近目标 (`next_distance < 0.5`)
- 这是一个 **保守的近似成功信号**，只在明确满足多条件时触发，避免误判

### terminal_failure_penalty 缺失原因
- `explicit_failure_flag_available = false`
- 环境未提供 `info["failure"]` 或 `info["termination_reason"]`
- 虽然终止条件包含 `crash_or_body_contact` 和 `horizontal_position_outside_viewport`，但无法从 obs 直接判断具体是哪种终止类型
- **v1 策略**: 不加显式失败惩罚，而是通过持续的 orientation/velocity 约束和距离塑形隐式引导 agent 避免失败
- 如果后续观察到 agent 无法区分坠毁与正常着陆，可在 v2 中添加基于 next_obs 状态推断的失败惩罚（如检测高速+大角度的异常终止）

## 排除的职责 (excluded roles) 及原因

| Excluded Role | Reason |
|---------------|--------|
| explicit_success_failure_bonus | 环境 info 为空，无显式 success/failure 标志 |
| time_step_penalty_from_truncation | 源码 truncated 恒为 False，无步数截断机制证据 |
| wind_or_disturbance_compensation_bonus | obs 中无风力或扰动强度观测量 |
| action_smoothness_penalty | 无 previous_action 或 action_history 信号 |
| gated_reward (multi-stage) | v1 阶段避免过早引入复杂门控 |
| dynamic_curriculum_reward | 无历史训练数据参考，且 training_progress 未明确需求 |

## 留到后续迭代的职责

1. **engine_usage_penalty** (efficiency optimization)
   - 当前 v1 专注主任务学习，不惩罚引擎使用
   - 后续若 agent 学会着陆但引擎使用过度，可加入轻量 action penalty

2. **contact_bonus** (首次接触奖励)
   - 当前通过 terminal_soft_landing_bonus 隐式覆盖
   - 若 agent 学习缓慢可在 v2 加入"正确时机的首次接触"奖励

3. **terminal_failure_penalty** (显式失败惩罚)
   - 当前依赖约束隐式引导
   - 若 agent 无法区分坠毁与着陆，v2 可添加基于 obs 状态的失败推断

4. **preview_conditioned_reward** (前瞻性调整)
   - 当前 obs 中无明确的"前方信息"维度
   - 若环境后续暴露高度传感器或前方距离探测，可考虑加入

## 训练后应观察的 failure modes

| Failure Mode | Evidence to Check | Possible Intervention |
|--------------|-------------------|----------------------|
| **悬停不着陆** | terminated 很少发生，next_distance 长期 <0.5 但 contact 始终为 0 | 增加 goal_approach_progress 权重，或对长期悬停施加轻微惩罚 |
| **高速冲撞目标** | 频繁终止且终止时 velocity_magnitude > 1.0 | 增大 velocity_brake_near_target 权重（当前 -2.0 → -4.0） |
| **姿态失控翻转** | 终止前 abs(next_angle) 或 abs(next_ang_vel) 持续增大 | 增大 orientation_stability 权重（当前 -1.0 → -2.0） |
| **越界漂移** | horizontal_position_outside_viewport 频发，且 abs(next_x_pos) 持续增大 | 检查 goal_approach_progress 的水平分量贡献，可考虑加入 x_pos 的额外惩罚 |
| **单侧接触不稳定** | terminal_soft_landing_bonus 触发率低，但单侧 contact 触发率高 | 保持当前设计（要求双接触），或放宽为"至少一侧接触+姿态稳定" |
| **距离塑形饱和** | goal_approach_progress 的 episode_mean 接近 0，agent 在中距离停滞 | 检查是否被 velocity_brake 过度抑制，适当降低其权重或调整 proximity_factor |

## 组件尺度与平衡分析

- **goal_approach_progress** (w=10.0): 主学习信号，距离改善 0.1 单位 → +1.0 reward，量级适中
- **velocity_brake_near_target** (w=-2.0): 接近目标时速度 0.5 → -0.5 reward，不会压制远距离加速
- **orientation_stability** (w=-1.0): 角度 0.2 rad → -0.04，角速度 0.5 → -0.125，轻量约束
- **terminal_soft_landing_bonus** (+50.0): 稀疏但强烈的终局信号，补偿回合中累积的约束惩罚

预期平衡：主信号驱动接近目标，约束在靠近时自动生效，终局奖励区分成功与普通终止。