# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.
    
    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: discourages high speed when close to the target
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: soft bonus for simultaneous low speed, upright posture, double contact and proximity to target
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)          # near target → gate ≈ 1; far away → gate ≈ 0
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: soft landing proxy (joint condition)
    w_landing = 2.0
    # proximity factor: exp(-dist^2 / 0.25)
    prox_factor = 2.718281828 ** (-dist**2 / 0.25)
    # velocity factor: exp(-(|vx|+|vy|)^2 / 0.1)
    vel_abs_sum = abs(vx) + abs(vy)
    vel_factor = 2.718281828 ** (-vel_abs_sum**2 / 0.1)
    # angle factor: exp(-angle^2 / 0.01)
    angle_factor = 2.718281828 ** (-angle**2 / 0.01)
    # contact gate (0/1 product)
    double_contact = left_contact * right_contact
    landing_proxy = w_landing * prox_factor * vel_factor * angle_factor * double_contact

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动力学子类型
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **目标**：2D 小车从随机位置出发，到达目标平台（相对坐标 `(0,0)`）并稳定停靠，同时尽量减少引擎使用。

## 2. 选中的奖励职责及信号映射
| role | 信号来源 | formula operator | 实现方式 |
|------|---------|-----------------|---------|
| **position_approaching** (主学习信号) | `dist = sqrt(x² + y²)` | `-distance_to_target`（dense_state_signal） | `distance_penalty` |
| **velocity_reduction** (安全约束) | `vx, vy, dist` | 带距离衰减门的二次速度惩罚 | `velocity_penalty` |
| **orientation_stabilization** (安全约束) | `angle, ang_vel` | 绝对值惩罚 | `orientation_penalty` |
| **safe_landing_confirmation** (任务完成近似) | `dist, vx, vy, angle, left_contact, right_contact` | 多因子乘积的 soft proxy | `landing_proxy` |

## 3. 公式算子选择依据
- **distance_penalty**：连续负距离奖励，每步提供到达方向的梯度。
- **velocity_penalty**：平方速度惩罚乘以 `1/(1+dist)`，使得智能体在远处时可以高速接近，靠近目标时强制减速，防止硬着陆。
- **orientation_penalty**：对姿态误差和角速度的绝对值惩罚，避免车辆翻滚或单侧接触。
- **landing_proxy**：使用指数衰减的乘积构造软着陆奖励——`exp(-dist²/T₁) * exp(-‖v‖²/T₂) * exp(-angle²/T₃) * contact_L * contact_R`。该奖励仅在同时满足低距离、低速度、小角度和双接触时显著，形成“着陆完成”的连续梯度。

## 4. 排除的职责及原因
- **fuel_efficiency_penalty**：v1 阶段默认不引入动作效率惩罚，避免压制探索和机动。
- **time_efficiency_encouragement**：无可用的时间/步数信号，且 training_progress 被禁止使用。
- **contact_force_smoothness**：环境仅提供布尔接触标志，无接触力信息。
- **terminal_success_reward / terminal_failure_penalty**：`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，info 为空，无法使用硬性的成功/失败信号。

## 5. 设计遵循的原则
- **信号可用性优先**：严格使用 `obs` / `next_obs` 中声明的 8 个维度，未使用 info、original_reward 或未声明的观测。
- **稠密性**：主学习信号（距离惩罚）每步提供梯度；约束和着陆代理也是稠密/半稠密的。
- **尺度平衡**：`distance_penalty`（~-1 到 0）与 `landing_proxy`（最大约 +2）在接近目标时可比，不会互相淹没；速度、姿态惩罚权重量级较小，仅起稳定作用。
- **避免可利用捷径**：速度惩罚通过距离门动态调整，防止 agent 过早减速或刻意低速刷分；landing proxy 要求双接触且所有因子同时达标，避免在错误位置获得虚假奖励。

## 6. 留待后续迭代的职责
- 燃料消耗惩罚（action-based）
- 动态课程调整权重
- 更复杂的分段进度或时间效率考虑

## 7. 训练后应重点观察的 failure modes
- **过早 settled 在错误位置**：若 landing_proxy 在距离较大时仍产生显著奖励，需降低 T₁ 或增加权重平衡检查。
- **高速撞击**：速度惩罚 gate 或权重不足时可能出现，应观察 crash 前的速度分布。
- **姿态振荡**：若 orientation 惩罚与动作组合造成不稳定，可尝试增大角速度惩罚或使用平方形式。
- **单侧接触刷分**：landing_proxy 因 contact 乘积为 0 防止了单侧着陆奖励，但要确认双接触标志在成功姿态下真正为 1。
