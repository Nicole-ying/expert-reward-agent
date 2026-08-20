# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs indices:
    # 0: x_position (relative to pad center)
    # 1: y_position (relative to pad height)
    # 2: x_velocity
    # 3: y_velocity
    # 4: body_angle
    # 5: angular_velocity
    # 6: left_support_contact (0.0 or 1.0)
    # 7: right_support_contact (0.0 or 1.0)

    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft = next_obs[6]
    nright = next_obs[7]

    # Squared distance to pad center
    dist_sq = nx**2 + ny**2
    dist = (dist_sq + 1e-8) ** 0.5

    # -- 1) approach progress: penalize squared distance (dense signal) --
    w_pos = 1.0
    pos_reward = -w_pos * dist_sq

    # -- 2) soft landing: speed & attitude penalty gated by distance --
    k_gate = 10.0
    gate = 1.0 / (1.0 + k_gate * dist)       # near pad -> gate ~1, far away -> gate ~0

    w_vel = 0.5
    vel_penalty = -w_vel * (nvx**2 + nvy**2) * gate

    w_angle = 0.5
    angle_penalty = -w_angle * (nangle**2) * gate

    w_angvel = 0.5
    angvel_penalty = -w_angvel * (nangvel**2) * gate

    # -- 3) safe contact: reward both legs on the pad near center --
    contact_proxy = nleft * nright               # both legs touching -> 1.0, else 0.0
    contact_threshold = 0.3
    contact_factor = max(0.0, 1.0 - dist / contact_threshold)
    w_contact = 10.0
    contact_reward = w_contact * contact_proxy * contact_factor

    total_reward = pos_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        'pos': pos_reward,
        'vel': vel_penalty,
        'angle': angle_penalty,
        'angvel': angvel_penalty,
        'contact': contact_reward
    }

    return total_reward, components
```

# reward_v1 设计说明

## 1. 任务画像选择
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **control_type**: discrete (4 actions)

## 2. 选用的奖励职责及其公式算子

| 职责 (role_id) | 使用的信号 (usable signals) | 公式算子 | 数学形式 |
|---|---|---|---|
| `approach_progress` | x_position, y_position | `dense_state_signal` (quadratic penalty) | `-w_pos * (x² + y²)`，以着陆垫为原点，鼓励位置靠近垫中心。 |
| `soft_landing` | x_velocity, y_velocity, body_angle, angular_velocity | `quadratic_penalty` + 距离门控 (自定义 bounded gate) | `-w * signal² * gate(dist)`，远处 gate≈0，不抑制接近速度；靠近目标时 gate→1，强制减速、摆正姿态。 |
| `safe_contact` | left_contact, right_contact, x_position, y_position | `joint_condition_proxy`（乘积形式）| 当双腿同时接触且距离中心小于阈值时给予正向奖励，用 `contact_factor = max(0, 1 - dist/threshold)` 实现连续衰减，避免稀疏到局部。 |

## 3. 未使用的职责及原因

- **`fuel_efficiency`** 与 **`time_penalty`**：v1 专注于学会安全着陆，动作代价或步数惩罚会压制早期探索、阻碍 agent 学习使用引擎，因此延后到 v2 再引入。
- **`explicit_success_bonus`**：环境没有显式 `success` 标志，无法使用。
- **`remaining_fuel_reward`**：观测中不含燃料量，无法使用。

## 4. 为什么没有 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available` 为 **false**，`explicit_failure_flag_available` 为 **false**。
- 环境 `info` 始终为空，无法从终止原因中获取成功/失败信号。
- 所有终止时的奖惩信息都由**每步观测中的连续信号**代理：距离惩罚、接触奖励和速度姿态惩罚共同区分安全着陆与坠落/飞出等失败终止。

## 5. 设计中的预防措施与预期 failure modes

- **利用风险防范**：纯距离最小化可能导致高速撞击，因此加入了距离门控的速度/姿态惩罚——远处允许高速，靠近后必须减速，避免仅靠负距离奖励导致的冲击行为。
- **虚假成功防护**：`safe_contact` 强制要求双腿接触**且**垫中心距离小于阈值，防止在视野边缘双腿接触时误发奖励。
- **训练后需要观察的 failure modes**：
    - 长期悬停在垫上方不降落（应增大接触奖励或降低接触阈值）；
    - 靠近垫后仍保持高速，导致撞击后弹飞（需调整门控参数使惩罚更陡）；
    - 主引擎频繁使用、姿态引擎几乎不用导致燃料耗尽（待 v2 引入动作代价）；
    - 着陆后反复弹跳（接触奖励中断 + 速度惩罚增加应自然缓解，否则考虑添加短暂的缓冲时间）。

## 6. 后续迭代预留

- 可添加 `fuel_efficiency`（离散动作惩罚）来降低燃料消耗；
- 可引入 `time_penalty`（弱负步数惩罚）鼓励快速完成；
- 若飞行器出现剧烈姿态振荡，可增加 `action_smoothness_penalty`（需要 history）；
- 如果接触学习困难，可尝试 `soft_contact_proxy` 进一步平滑奖励。
