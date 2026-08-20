# Response Record

1. **evidence**：Iter 1 以 joint_condition_proxy 为主的 safe_contact 取得了 144 分，per‑step 仅 0.278，配合 progress delta 和轻量姿态/速度罚项；Iter 2 将 safe_contact 改为一次性 20 的 landing bonus 后 safe_contact 的 episode_sum 飙升至 852、active rate 降至 7.1%，姿态和速度罚项被淹没，score 降至 124 且 terminated 数从 1 增至 13，表明稀疏大额事件奖励诱导了危险撞击行为而非安全着陆。
2. **behavior_diagnosis**：agent 在 Iter 2 中学会了尽快让双脚触碰 pad 以获取大额 bonus，但放弃了姿态控制与减速，导致大量 crash/failure 终止；目前策略无法区分“硬撞击后崩溃”与“软着陆并稳定”。
3. **signal_completeness**：任务需要接近目标、减速、保持姿态、安全接触并稳定，且要求尽可能快、尽可能少用发动机。当前缺少对水平速度、水平位置偏差、角速度的着陆条件约束，也无发动机使用代价，导致仅追求接触事件。需加入更全面的着陆联合条件、发动机罚项和更均衡的 progress 动力。
4. **selected_level**：Level 3 重建。连续预判❌且 Iter 2 稀疏事件奖励方向已被证伪，必须更换主信号框架。
5. **selected_intervention**：以 progress improvement_delta、多因子几何平均的 landing_proxy、姿态二次罚项和离散动作罚项的全新骨架，舍弃一次性 sparse bonus。
6. **falsifiable_hypothesis**：用多因子几何平均构造着陆代理奖励（position、velocity、angle 等多条件联合满足）并提供持续梯度，能引导 agent 同时满足所有安全着陆条件，从而减少 crash 并提升 settled 终止率；加入小的发动机罚项可以抑制无意义动作。
7. **expected_next_round**：safe_contact 更名为 landing_proxy 后的 episode_sum 应下降并保持适中的 active rate，progress 和 orientation 的 per‑step 量级升高；score 应超过 144 并朝着 200 收敛，terminated 比例提高且不再伴随大量负分 episode。
8. **main_risk**：多因子几何平均对每个因子的边界敏感，阈值设定若不准确可能仍导致 landing_proxy 过早/过晚激活；若 landing_proxy 系数过大再度淹没 progress，agent 可能又陷入局部行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Encourage moving toward the pad center.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 2.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: landing_proxy (joint_condition_proxy, geometric mean)
    #   Reward simultaneous satisfaction of all landing conditions:
    #   feet contact, centered position, low velocity, low angle,
    #   low angular velocity.
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # bounded proximity factors: 1/(1 + k*|error|)
    proximity_x = 1.0 / (1.0 + 10.0 * abs(next_obs[0]))   # target radius ~0.1
    proximity_y = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))   # pad height ~0
    low_xvel    = 1.0 / (1.0 + 5.0 * abs(next_obs[2]))    # tolerable x speed
    low_yvel    = 1.0 / (1.0 + 5.0 * abs(next_obs[3]))    # tolerable y speed
    low_angle   = 1.0 / (1.0 + 10.0 * abs(next_obs[4]))   # upright angle
    low_angvel  = 1.0 / (1.0 + 2.0 * abs(next_obs[5]))    # angular velocity

    num_factors = 7
    product = (both_contacts * proximity_x * proximity_y *
               low_xvel * low_yvel * low_angle * low_angvel)
    # geometric mean to prevent collapse of a single factor
    landing_proxy_reward = 4.0 * (product ** (1.0 / num_factors))

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Suppress large angles and angular velocities at all times.
    # ------------------------------------------------------------
    angle_penalty = 2.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.5 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: action_penalty (action_efficiency)
    #   Small fuel penalty for any engine use (actions 1,2,3).
    # ------------------------------------------------------------
    action_penalty = -0.03 * (1.0 if action != 0 else 0.0)

    total_reward = (progress_reward +
                    landing_proxy_reward +
                    orientation_stability_reward +
                    action_penalty)

    components = {
        "progress_reward": progress_reward,
        "landing_proxy_reward": landing_proxy_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "action_penalty": action_penalty,
    }

    return float(total_reward), components
```
