# Response Record

# 设计理由
当前失败源于上一轮新增的 `touchdown_success` 一次性奖励被 agent 通过反复弹跳/反复接触大量刷取（均值 755.0，远超所有其他信号），导致整体奖励被该组件支配，训练退化为“弹跳刷分”而非稳定着陆。  
本轮以 best 骨架（iter4）为基础，将该稀疏事件奖励**替换**为连续 `grounded_quality` 组件：仅在任一支撑腿接触且速度低于安全阈值时发放一个随速度减小而线性增大的有界正向信号。  
此举保留了“接触后低速是好状态”的梯度，但不再依赖一次性发放（避免重复触发），且奖励量级（最大 0.3/步）与原有 `landing_quality` 共同构成温和的着陆引导，不太可能再次淹没主进展信号。

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

    # Contact flags
    left_contact_next = next_obs[6] > 0.5
    right_contact_next = next_obs[7] > 0.5

    # --- Reward weights (best skeleton) ---
    w_approach = 2.0
    w_vel_penalty = 0.5
    w_angle = 0.5
    w_angvel = 0.1
    w_landing = 0.1
    w_grounded = 0.3        # new continuous grounded quality weight

    # Component 1: dense progress towards the pad
    approach_reward = w_approach * approach_delta

    # Component 2: speed constraint (hinge)
    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    # Component 3: angular stability (quadratic penalties)
    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # Component 4: continuous landing‑quality proxy (unchanged)
    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor

    # Component 5 (REPLACED): grounded quality — only when leg contact and low speed
    grounded_reward = 0.0
    if left_contact_next or right_contact_next:
        # linear factor: 1 at zero speed, 0 at safe_speed, 0 above
        speed_ratio = max(0.0, 1.0 - speed_next / safe_speed)
        grounded_reward = w_grounded * speed_ratio

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward + grounded_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
        "grounded_quality": grounded_reward,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: touchtown_success 均值 755.0（远超其它），episode 平均长 884，terminated 仅 3/20，agent 显然通过重复接触刷取一次性奖励。
- **behavior**: agent 陷入了弹跳‑接触循环以最大化 touchtown 奖励，未能学会稳定着陆。
- **signal**: 一次性 touchtown 奖励缺少单次领取限制，导致可被恶意重复获取，彻底淹没了着陆相关的真实进展信号。
- **level**: Level 2 ‑ 数学结构变换（稀疏一次性事件 → 连续有界状态 reward）
- **hypothesis**: 移除可刷取的一次性奖励，改为只奖励“接触且低速”的连续信号后，agent 将被引导向真正稳定的着陆，而不再弹跳，得分有望回到 best 历史水平并继续提升。
- **risk**: 若 grounded_quality 的最大值 0.3 与 landing_quality 叠加后仍偏弱，可能延迟着陆速度，但可通过后续调整系数解决。
