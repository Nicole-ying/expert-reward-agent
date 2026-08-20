# Response Record

## 诊断报告

### Evidence

| 信号 | 数据 | 结论 |
|---|---|---|
| `contact_reward` | ep_sum=0.0, active_rate=0%, 所有 checkpoint 均为 0 | **死组件**：`left_contact * right_contact` 要求双足同时触地，训练中几乎不可能发生 |
| `vel_damping` | ep_sum=0.0, active_rate=0%, 所有 checkpoint 均为 0 | **死组件**：gate 阈值 `dist_thresh=1.0` 太窄，agent 极少进入该范围 |
| `progress_reward` | ep_sum 从 -3.2 → +0.1，active 100% | 正常工作，agent 学会了靠近目标 |
| `orientation_penalty` | ep_sum 从 -2.5 → -0.033，active 100% | 正常工作，agent 学会了保持竖直 |
| Score trend | -758 → -155（best -145 at cp90%） | 在改善但远未达标（目标 200），最佳仅出现在训练中期，随后退化 |

### Behavior Diagnosis

当前策略学到的是：**靠近目标区域 + 保持竖直姿态，但从未真正着陆**。两个负责引导最终着陆阶段的组件完全死亡：

- `contact_reward`：双足同时触地是极小概率事件，agent 在训练中永远触发不了，因此学习不到"着陆是好"的信号。
- `vel_damping`：门控阈值 1.0 相对于初始距离（可能 >10）太窄，agent 几乎从不进入有效区域。

结果是 agent 学会了朝目标飞、保持竖直，但缺乏任何着陆梯度——它不知道到达 pad 后该做什么。Score 在 cp90% 达到最佳后回退（-145 → -155），说明后期出现了 reward hacking 或策略漂移（所有有效信号都已被充分利用，但缺少着陆信号来突破瓶颈）。

### 干预方案 — L2

**语义方向**：将死组件 `contact_reward` 从"双足触地二元条件"改为**基于距离的软着陆代理（proximity-based soft landing proxy）**。同时修复 `vel_damping` 的门控阈值使其激活。两者服务于同一语义：**引导智能体在接近目标垫时减速、保持姿态、最终着陆**。

核心变换：
- `contact_reward`：`left_contact * right_contact` → `exp(-dist²/σ²) * exp(-speed²/σ²) * exp(-angle²/σ²)`，无论是否实际触地，只要处于"着陆区"就能获得连续梯度信号
- `vel_damping`：`dist_thresh` 从 1.0 → 3.0，扩大门控激活范围

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v2: proximity-based soft landing proxy replaces dead contact_reward;
        widened vel_damping gate so it actually activates.
    """
    # Extract relevant signals
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # 1. Progress reward (improvement_delta on Euclidean distance)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # positive when approaching
    w_progress = 1.0
    progress_reward = w_progress * progress

    # 2. Orientation penalty (quadratic_penalty)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft landing proxy (proximity-based, continuous — replaces dead contact_reward)
    # Rewards being in the "landing zone": close to pad + low speed + upright.
    # Creates a smooth gradient from approach → hover → touch down.
    sigma_dist = 1.5                                    # landing zone radius
    sigma_vel = 0.5                                     # acceptable landing speed
    sigma_angle = 0.3                                   # acceptable tilt

    dist = dist_curr
    speed = (vx ** 2 + vy ** 2) ** 0.5

    proximity_factor = 2.718281828 ** (- (dist ** 2) / (2.0 * sigma_dist ** 2))
    vel_factor = 2.718281828 ** (- (speed ** 2) / (2.0 * sigma_vel ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (2.0 * sigma_angle ** 2))

    w_landing = 5.0
    soft_landing_proxy = w_landing * proximity_factor * vel_factor * angle_factor

    # 4. Near-pad velocity damping (wider gate so it activates)
    dist_thresh = 3.0                                   # was 1.0 — far too narrow
    gate = max(0.0, 1.0 - dist / dist_thresh)           # 0 far away, 1 at origin
    w_vel_damp = 0.3
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + soft_landing_proxy + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```

### 预期效果

| 组件 | 旧行为 | 预期新行为 |
|---|---|---|
| `soft_landing_proxy` | 永远为 0 | active_rate > 0，随 agent 靠近 pad 逐渐增长，形成着陆梯度 |
| `vel_damping` | 永远为 0 | agent 进入 3.0 半径后激活，主动减速 |
| Score | -155 plateau | 应有明显跃升，因为 agent 首次获得了"着陆区"的正向信号 |
