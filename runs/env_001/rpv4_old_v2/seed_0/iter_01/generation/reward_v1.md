# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward function for the soft landing / docking task.

    Roles:
      - approach_target: continuous distance reward (bounded, positive)
      - safe_landing: speed penalty near pad + contact bonus for both legs
      - settlement_stabilization: penalty for velocity/angular movement after contact
    """
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # helpers
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    both_contacts = (left_contact > 0.5) and (right_contact > 0.5)

    # component A: approach_target – positive bounded distance reward
    # using 1/(1 + kp*dist) → max 1.0 when exactly on target
    kp = 1.0
    w_progress = 1.0
    progress_reward = w_progress * (1.0 / (1.0 + kp * dist))

    # component B: safe_landing – speed penalty when close to pad, contact bonus
    threshold_speed = 2.0        # apply speed penalty only within this distance
    w_speed = 0.5                # penalty weight for |vx|+|vy|
    if dist < threshold_speed:
        landing_speed_penalty = -w_speed * (abs(x_vel) + abs(y_vel))
    else:
        landing_speed_penalty = 0.0

    # contact bonus: both legs touching, higher when near center
    w_contact = 2.0
    kc = 1.0
    if both_contacts:
        contact_reward = w_contact * (1.0 / (1.0 + kc * dist))
    else:
        contact_reward = 0.0

    # component C: settlement_stabilization – after contact, encourage full stillness
    w_settle = 0.2
    if both_contacts:
        settlement_penalty = -w_settle * (
            abs(x_vel) + abs(y_vel) + abs(angular_vel) + abs(body_angle)
        )
    else:
        settlement_penalty = 0.0

    total_reward = progress_reward + landing_speed_penalty + contact_reward + settlement_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_speed_penalty": landing_speed_penalty,
        "contact_reward": contact_reward,
        "settlement_penalty": settlement_penalty,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与奖励职责选择

- **task_family**: `navigation_goal_reaching`  
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
- **selected reward roles**:  
  - `approach_target` **(mandatory)** – 鼓励主体向中央目标垫靠近  
  - `safe_landing` **(mandatory)** – 在接近垫子时鼓励软着陆（减速 + 双脚同时接触）  
  - `settlement_stabilization` **(mandatory)** – 接触垫子后鼓励完全静止（速度/角速度/姿态归零）  

**条件职责未在 v1 中包含**：  
- `fuel_efficiency`、`orientation_stability`、`time_penalty` 均推迟到后续迭代，避免过早抑制探索。

**禁用的职责**：  
- `crash_penalty_from_info` – `info` 为空，无可用的失败标签  
- `velocity_tracking` – 无预设速度剖面  
- `precision_docking` – 目标垫足够大，无需亚厘米级精度  

## 2. 信号映射与公式算子选择

| 组件 | 角色 | 使用的观测 | 公式算子 |
|------|------|------------|----------|
| `progress_reward` | `approach_target` | `x_position`, `y_position` (构建距离) | `bounded_signal`: `1/(1+kp*dist)` – 连续、始终有梯度，距离越小奖励越大，最大趋近 1.0 |
| `landing_speed_penalty` | `safe_landing` | `x_velocity`, `y_velocity` + `dist` | `dense_state_signal` 的 hinge 形式：仅当 `dist < threshold` 时激活绝对速度惩罚，避免远距离抑制机动 |
| `contact_reward` | `safe_landing` | `left_contact`, `right_contact`, `dist` | 乘积形式的 `joint_condition_proxy`（连续 bounded）：双脚接触且越靠近中心奖励越高，给着陆提供强正信号 |
| `settlement_penalty` | `settlement_stabilization` | `x_vel`, `y_vel`, `angular_vel`, `body_angle` | `dense_state_signal` 的二次/线性组合：双脚接触后惩罚所有运动，引导彻底静止 |

## 3. 为何不使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available = false`，`explicit_failure_flag_available = false`  
- `info` 字段完全为空，无法读取任何终止原因标签  
- 因此奖励中禁止使用 `info["success"]` 或 `info["failure"]`，转而通过观测构造软的进展和着陆信号  

## 4. 留到后续迭代的职责

- **fuel_efficiency**：接近垫子时对非零动作施加轻微代价，优化能量消耗  
- **orientation_stability**：接近垫子时单独惩罚角偏差，在当前版本中由 `settlement_penalty` 的部分项覆盖  
- **time_penalty**：全局微小负奖励以鼓励快速完成任务，v1 先不引入以免影响探索  

## 5. 训练后应观察的 failure modes 及可能干预

根据 `environment_card.md` 列出的典型失败模式：

- **主体不移动或只使用无推力动作** → 检查 `progress_reward` 变化；可能需增大 `w_progress` 或暂时关闭 `landing_speed_penalty`  
- **高速撞击垫子后弹飞或 crash** → 检查延迟减速；可增大 `w_speed` 或降低 `threshold_speed`  
- **在垫子周围反复振荡但不落地** → `contact_reward` 不够强；可提高 `w_contact` 或增加 `settlement_penalty` 权重  
- **过度使用姿态引擎导致旋转失控** → 暂未加入独立姿态约束；可在后续迭代中增强 `orientation_stability` 或加重接触后的 `body_angle` 惩罚  
- **过早耗尽燃料或因时间惩罚不敢动作** → 本版本未引入效率和时间惩罚，该风险较低  

v1 设计通过一个连续的正向距离奖励 + 近距离速度刹车 + 接触奖励 + 接触后静止惩罚，构建了一个从移动到着陆再到稳定的完整梯度通道，避免对失败标签的依赖，同时为后续加入燃料效率和姿态约束预留了清晰的扩展点。