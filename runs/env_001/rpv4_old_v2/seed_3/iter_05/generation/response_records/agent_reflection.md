# Response Record

1. `evidence`：iter4 得分 227.06、全部 20 回合以 terminated 成功结束，len=490.1；safe_contact_reward 主导 magnitude share 83.9% 但 active_rate 仅 7%，progress_reward 每步均值 0.0041 仅为 safe_contact 的约 1/16；orientation_stability 和 velocity_damping 全覆盖但量级很小，未见行为破坏。
2. `behavior_diagnosis`：agent 已学会精准着陆并稳定在目标平台上，100% 成功，但接近过程的激励过弱，导致接近阶段可能偏长（len≈490），存在加速空间。
3. `signal_completeness`：已有距离减少引导、姿态与速度约束、着陆奖励，缺少推力使用惩罚，但现有成功表现下可暂不引入，保留当前职责结构以减小风险。
4. `selected_level`：Level 2 — 组件公式方向正确，仅 per‑step 量级严重失衡（progress 极弱），通过调整单组件系数即可改善接近效率。
5. `selected_intervention`：唯一修改 `progress_reward`，将其系数从 1.0 提升至 2.0，以增强接近过程的引导强度。
6. `falsifiable_hypothesis`：更强的 progress 激励会使 agent 在前期更快缩小距离，从而缩短 episode 长度；同时 safe_contact 仍维持着陆后高奖励，不会破坏成功终止率。
7. `expected_next_round`：len 应继续下降（预计 <450），score 可能小幅上升或持平，terminated 保持 20/20，progress_reward 的 episode_sum 和 per‑step 量级将明显增大，safe_contact 的 magnitude share 预计下降。
8. `main_risk`：progress 过强可能导致 agent 以过高速度接近 pad，velocity_damping 和 orientation_stability 在 gate 作用下控制能力可能不足，若出现 crash 增多需后续削弱 progress 或加强速度约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Potential = -(x^2 + y^2), reward = potential_curr - potential_next
    #   Positive when distance to pad decreases, zero when stationary.
    #   Coefficient increased from 1.0 to 2.0 to strengthen approach incentive.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 2.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    #   Penalise speed more strongly when near the pad.
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)          # alpha = 2.0
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Keep body upright and angular velocity small.
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (joint_condition_proxy)
    #   Give positive reward when both legs touch the pad,
    #   vehicle is at pad height, and vertical velocity is low.
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    # y position close to zero (pad height)
    proximity_to_pad = max(0.0, 1.0 - abs(next_obs[1]) / 0.2)   # threshold_y = 0.2
    # vertical velocity small
    soft_vel = max(0.0, 1.0 - abs(next_obs[3]) / 0.2)           # threshold_vy = 0.2
    safe_contact_reward = 1.0 * both_contacts * proximity_to_pad * soft_vel

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return float(total_reward), components
```
