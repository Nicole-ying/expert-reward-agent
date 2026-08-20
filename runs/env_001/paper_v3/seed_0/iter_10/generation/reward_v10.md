1. `evidence`：最终评估20个episode全部truncated在1000步、无提前终止，外部score=-30.78，但landing_proxy episode均值高达4404（占magnitude_share 88.2%），distance_penalty均值-580.92，orientation和velocity_panelty接近零；agent依靠高额连续proxy存活满步数却未真正着陆。
2. `behavior_diagnosis`：agent学会了在目标附近慢速悬停（avg dist≈0.58、低速、正立），持续收割landing_proxy的稠密正值奖励，但因缺少终止激励从未完成settled着陆，导致外部得分为负。
3. `signal_completeness`：position_approaching、velocity_reduction、orientation_stabilization职责均存在，但**safe_landing_confirmation完全缺失**——没有任何稀疏成功奖励或终止驱动信号；同时landing_proxy作为状态绝对值奖励，允许"占据好状态即可持续获奖"，造成proxy与任务完成脱节。
4. `selected_level`：Level 2，证据模式匹配**state_to_improvement**（占据好状态即可持续获奖）和**proxy_to_completion_alignment**（proxy高但外部得分负），当前几何均值+接触bonus结构无法修复"维持状态即得奖"的根本缺陷。
5. `selected_intervention`：将landing_proxy从**状态绝对值奖励**变换为**势能差奖励**（potential-based shaping），即`Φ(next_state) - Φ(state)`，其中Φ为approach_quality的几何均值；维持好状态奖励归零，只有改进动作获得正反馈。
6. `falsifiable_hypothesis`：势能差结构消除悬停套利空间后，agent必须持续改进接近质量才能获得正奖励；当改进空间耗尽（已处于最优状态）时自然触发settled终止，从而结束负的distance_penalty累积，形成"尽快完成"的间接激励。
7. `expected_next_round`：landing_proxy的episode_sum_mean应大幅下降（不再累积恒定正值），signed_share不再主导；episode_length应出现提前终止（非truncated），外部score应上升；若出现策略茫然（active_rate下降、徘徊），则说明势能尺度不足需要增大。
8. `main_risk`：势能差可能诱导agent在目标附近来回振荡（远离再靠近以制造正delta），若distance_penalty和velocity_penalty的抑制力不足，可能出现高频抖动而非稳定着陆。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target, gated by distance
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: potential-based shaping reward that rewards IMPROVEMENT
      in approach quality, not the absolute state. Computed as
      w * (approach_quality(next_obs) - approach_quality(obs)).
      Zero reward for maintaining a constant good state, positive only
      for progress toward the landing configuration.
    """
    # Unpack observations
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]
    prev_angle = obs[4]

    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component A: distance penalty (core progress signal) ---
    dist = (x**2 + y**2) ** 0.5
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # --- Component B: velocity penalty (damped by distance to target) ---
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # --- Component C: orientation stabilization penalty ---
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # --- Component D: landing proxy (potential-based shaping) ---
    # Bounded approach quality function: geometric mean of 3 continuous factors.
    # This is the potential function Phi(state).
    def approach_quality(px, py, pvx, pvy, pangle):
        d = (px**2 + py**2) ** 0.5
        prox = max(0.0, 1.0 - d / 2.5)
        sp = (pvx**2 + pvy**2) ** 0.5
        vel = max(0.0, 1.0 - sp / 2.0)
        ang = max(0.0, 1.0 - abs(pangle) / 0.5)
        # Geometric mean enforces joint satisfaction; zero if any factor is zero
        return (prox * vel * ang) ** (1.0 / 3.0)

    # Potential at previous state (obs) and current state (next_obs)
    phi_prev = approach_quality(prev_x, prev_y, prev_vx, prev_vy, prev_angle)
    phi_curr = approach_quality(x, y, vx, vy, angle)

    # Potential-based shaping: reward = gamma * Phi(next) - Phi(now)
    # Using gamma=1 since environment has no explicit discount in reward design.
    # w_potential controls the scale of the improvement reward.
    w_potential = 8.0
    landing_proxy = w_potential * (phi_curr - phi_prev)

    # --- Total reward ---
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```