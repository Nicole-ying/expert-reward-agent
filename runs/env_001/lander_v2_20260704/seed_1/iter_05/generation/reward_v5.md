**evidence**: Score=-113.99, 18/20 early termination, distance_cost=-75.73 dominates (57.7% magnitude share), approach_landing_reward=+44.68 (34.0%). |penalty/progress| ratio=(75.73+10.86)/44.68≈1.94, far above 0.5 trigger. Last change bumping proximity_improvement 5.0→6.0 caused regression from best -87.91.

**behavior_diagnosis**: Agent crashes rapidly (~70 steps) without landing; the quadratic distance penalty accumulates relentlessly while improvement-based rewards flatline whenever random forces or oscillation prevent further progress, making early termination the least-bad strategy.

**signal_completeness**: Core shaping signals (proximity improvement, stability constraint, landing contact) are structurally present and reachable; the failure is scale imbalance, not missing responsibilities.

**selected_level**: Level 1 — distance_cost role and monotonic math form are reasonable, but its coefficient makes penalty dominate progress at 1.94≫0.5, a clear scale problem per penalty_magnitude_control evidence.

**selected_intervention**: Reduce w_dist from 1.0 to 0.08 (12.5× reduction, exceeding the 10× minimum for penalty dominance), applied to best code (iter 3 base with proximity_improvement=5.0), no other component changed.

**falsifiable_hypothesis**: Lowering distance penalty by 12.5× should bring |penalty/progress| below 0.5, allowing approach_landing_reward to yield net-positive episodes when the agent makes even modest progress, reducing the "crash early" incentive and enabling landing skill acquisition.

**expected_next_round**: distance_cost episode_sum_mean should drop to ~-6, ratio toward ~0.3-0.5, score should improve from -113.99, early termination rate should decline, episode length may initially increase as exploration replaces fast crashing.

**main_risk**: The distance penalty was the primary approach driver; if proximity_improvement alone is too sparse/noisy, the agent may survive longer but drift without sufficient target-directed incentive, failing to reach the pad at all.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一步状态
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离与稳定性成本
    w_dist = 0.08
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05

    distance_cost = -w_dist * (nx ** 2 + ny ** 2)
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 改善量型 approach_landing_reward：有界差分替代持续状态值
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    curr_dist = (nx ** 2 + ny ** 2) ** 0.5
    prev_vel = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    curr_vel = (vx ** 2 + vy ** 2) ** 0.5
    prev_contact = (obs[6] + obs[7]) / 2.0
    curr_contact = (left_contact + right_contact) / 2.0

    # 接近改善：向目标靠近即获奖
    proximity_improvement = 5.0 * max(0.0, prev_dist - curr_dist)

    # 减速改善：仅在靠近着陆垫时奖励减速（避免远距离时惩罚必要的高速接近）
    proximity_gate = max(0.0, 1.0 - curr_dist / 1.5)
    velocity_improvement = 10.0 * proximity_gate * max(0.0, prev_vel - curr_vel)

    # 姿态改善：减小倾角即获奖
    angle_improvement = 3.0 * max(0.0, abs(obs[4]) - abs(angle))

    # 触地改善：建立支撑腿接触的转移奖励（非持续占有）
    contact_improvement = 30.0 * max(0.0, curr_contact - prev_contact)

    approach_landing_reward = proximity_improvement + velocity_improvement + angle_improvement + contact_improvement

    total_reward = distance_cost + stability_cost + approach_landing_reward

    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'approach_landing_reward': approach_landing_reward
    }

    return float(total_reward), components
```