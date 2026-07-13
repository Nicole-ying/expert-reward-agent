# 设计理由
着陆质量奖励（landing_quality）当前只在机体接触垫子的少数步上产生信号，其余 99% 以上的步数奖励为 0，严重缺乏密集引导。因此将其从“依赖接触的稀疏乘积”改为**连续接近‑减速奖励**：直接使用 `dist_factor * speed_factor` 作为着陆质量 proxy，每一步都有梯度，并大幅降低系数（`w_landing=0.1`）使其 per‑step 量级与主进展信号 `approach_progress` 匹配，避免新奖励主导。其他组件保持不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next positions relative to landing pad center
    dx_curr, dy_curr = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_curr = (dx_curr**2 + dy_curr**2) ** 0.5
    dist_next = (dx_next**2 + dy_next**2) ** 0.5

    # Approach progress: positive when getting closer to the pad
    approach_delta = dist_curr - dist_next

    # Next‑step velocity and safety threshold
    vx_next, vy_next = next_obs[2], next_obs[3]
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    safe_speed = 0.2

    # Next‑step orientation and angular velocity
    angle_next = next_obs[4]
    angvel_next = next_obs[5]

    # --- Reward weights ---
    w_approach = 2.0
    w_vel_penalty = 5.0           # per unit speed above safe_speed
    w_angle = 0.5                 # penalty on squared body angle
    w_angvel = 0.1                # penalty on squared angular velocity
    w_landing = 0.1               # continuous landing proxy (reduced from 5.0)

    # Component 1: dense progress towards the pad
    approach_reward = w_approach * approach_delta

    # Component 2: speed constraint (hinge – only penalise when exceeding safe threshold)
    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    # Component 3: angular stability (quadratic penalties)
    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # Component 4: continuous landing‑quality proxy (no contact requirement)
    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: score=-37.99，91% 的 episode 超时截断，velocity_penalty 均集 -96 主导负分，landing_quality 虽 19.8 但仅靠极少接触步供给。
- **behavior**: agent 长时间徘徊，未能完成着陆，在速度惩罚压制下不敢或无法有效逼近目标。
- **signal**: 着陆信号极度稀疏（接触步 active_rate 远低于 5%），速度惩罚强度过高，正向引导不足。
- **level**: Level 2（稀疏项连续化），触发条件：landing_quality 几乎只在与垫接触的极少数步才非零。
- **hypothesis**: 将着陆奖励改为与距离、速度反向关联的连续信号，为每一步提供朝向目标减速的梯度，使 agent 能更早、更稳定地学习接近与减速行为。
- **risk**: 可能鼓励 agent 在目标上方低速盘旋而不实际接触，需后续加入接触奖励或速度门控以确保最终着陆。