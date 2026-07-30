# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_w   = obs[1]
    quat_x   = obs[2]
    quat_y   = obs[3]
    # quat_z = obs[4]  # not used directly
    v_x      = obs[13]  # forward velocity
    v_y      = obs[14]  # lateral velocity

    # ---- upright projection ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ---- soft health gate for forward reward ----
    z_low_safe  = 0.35
    z_high_safe = 0.85
    gate_z_low  = max(0.0, min(1.0, (body_z - 0.2) / (z_low_safe - 0.2)))
    gate_z_high = max(0.0, min(1.0, (1.0 - body_z) / (1.0 - z_high_safe)))
    gate_z = gate_z_low * gate_z_high

    up_min      = 0.5
    up_thr      = 0.7
    gate_up     = max(0.0, min(1.0, (up_z - up_min) / (up_thr - up_min)))
    health_gate = gate_z * gate_up

    # ---- forward progress (main learning signal) ----
    w_fwd   = 1.0
    forward = w_fwd * v_x * health_gate

    # ---- body height safety (hinge quadratic penalty) ----
    w_h       = 10.0
    low_hinge = max(0.0, z_low_safe - body_z)
    high_hinge= max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge ** 2 + high_hinge ** 2)

    # ---- upright orientation (hinge quadratic penalty) ----
    w_up          = 5.0
    upright_error = max(0.0, up_thr - up_z)
    upright_penalty = -w_up * (upright_error ** 2)

    # ---- lateral stability (quadratic penalty) ----
    w_lat          = 0.2
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action magnitude (light energy/smoothness proxy) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a ** 2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (forward + height_penalty + upright_penalty +
                    lateral_penalty + action_penalty)

    components = {
        "forward":          forward,
        "height_penalty":   height_penalty,
        "upright_penalty":  upright_penalty,
        "lateral_penalty":  lateral_penalty,
        "action_penalty":   action_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像摘要
- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: multi_legged_body_locomotion (quadruped, 8‑DOF torque control)
- **primary objectives**: 持续前向速度，保持身体高度 (0.2, 1.0)，维持躯干竖直
- **secondary objectives**: 侧向稳定性、能耗/动作平滑 (v1 仅轻量覆盖)
- **信号约束**: 无 global x/y 位置，无脚地接触信息，无显式 success/failure flag，info 为空

## 2. 选用的 reward roles 与信号映射

| role                  | type        | 信号来源                   | formula operator               |
|-----------------------|-------------|----------------------------|--------------------------------|
| forward_velocity      | mandatory   | `obs[13]` (v_x)            | dense_state_signal (linear) × soft_health_gate |
| body_height_safety    | mandatory   | `obs[0]` (body_z)          | dense_state_signal (hinge quadratic penalty) |
| upright_orientation   | mandatory   | `obs[1:5]` → up_z         | dense_state_signal (hinge quadratic penalty) |
| lateral_velocity_penalty | conditional | `obs[14]` (v_y)          | quadratic_penalty              |
| action_magnitude      | conditional | `action` (8D torque)       | quadratic_penalty (极轻)        |

## 3. 各角色设计理由

- **forward_velocity**：线性正奖励 `w_fwd * v_x` 提供每步梯度，直接驱动前进。为防止“先冲后死”（score 高但 terminated 率高），乘上 `health_gate`。gate 由身体高度和 upright 投影的连续因子构成，当身体接近危险边界时 gate 平滑衰减至 0，切断前进奖励，迫使 agent 在安全范围内优化速度。
- **body_height_safety**：使用 hinge quadratic penalty，只在高度进入 `(0.35, 0.85)` 之外时激活。相比全时二次惩罚，hinge 允许 agent 在安全区间内自由调整姿态，避免“不敢动”。二次形式在接近终止边界 (0.2/1.0) 时提供强梯度。
- **upright_orientation**：与高度类似，使用 hinge quadratic penalty（阈值 0.7）。目的不是完全锁死躯体摆动，而是防止大幅度倾斜导致翻滚。
- **lateral_velocity_penalty**：极轻二次惩罚（w=0.2），抑制明显侧向漂移，但不限制必要的探索。
- **action_magnitude**：8维动作空间默认容易产生高频振颤，`0.005` 的轻惩罚只作为平滑引导，不压制探索。

## 4. 排除的 roles 及原因

- **action_smoothness（动作平滑）**：需要访问上一时刻的动作，但 `compute_reward` 接口未保存历史，无法实现。
- **energy_efficiency（能耗优化）**：v1 不引入膝、踝力矩速度乘积等精确能耗项，仅以极轻扭矩幅度作为代理。
- **gait pattern / contact reward**：无可用的脚地接触信息。
- **terminal_success / terminal_failure reward**：环境无显式 success/failure flag，info 为空，无法实现。

## 5. 训练后应观察的 failure modes

1. **agent 选择站立不动**：若 forward + penalties 总和在 v_x≈0 时高于慢速前进，应微调权重（增大 w_fwd 或降低 height/upright 惩罚）。
2. **前冲后仍摔倒**：检查 gate 衰减是否太慢；可收窄安全区间或加重 hinge 惩罚。
3. **身体高度偏好上限/下限**：hinge 阈值可能需要根据实际步态微调（观察 body_z 分布）。
4. **急速偏航 / 侧向漂移**：考虑增大 lateral 惩罚或引入 yaw rate 惩罚（obs[18]），留到 v2。
5. **动作震颤**：当前极轻 action_penalty 可能无效，后续可引入关节角速度惩罚 (obs[19:27])。

所有未使用的健康/效率信号（角速度、关节速度等）将保留到后续迭代，在 v1 取得稳定前进策略后再加入。