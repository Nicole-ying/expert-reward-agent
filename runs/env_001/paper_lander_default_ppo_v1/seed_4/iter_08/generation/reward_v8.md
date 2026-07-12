**evidence**: score crashed from 197.60 (iter5 best) to -24.03 (iter7) with settlement_bonus episode_sum_mean exploding from 0.942→15.403→25.241, while terminations rose to 19/20 and avg length collapsed to 148 steps. settlement_bonus dominates at 80.9% magnitude share with only 37.4% active rate, creating sparse reward spikes that overshadow proximity_reward (3.7% magnitude) and speed_penalty_gated (14.8%).

**behavior_diagnosis**: the agent is crashing rapidly in most episodes (19/20 terminated, 8/20 score<-50 before step 150). it gets net positive settlement deltas (+25.24 avg) before crashing, suggesting it learns to briefly improve the settled product (approach, touch, slow down) but fails to achieve stable landing. the settlement signal's extreme magnitude relative to other components creates a "lottery" where lucky runs score 261 but most crash.

**signal_completeness**: all necessary signals are present (approach guidance via proximity_reward, speed constraint via gated penalty, landing incentive via settlement_bonus, orientation via penalty), but the settlement_bonus coefficient creates a scale imbalance that makes the continuous approach and safety signals negligible during training credit assignment.

**selected_level**: Level 1 — the settlement_bonus has correct mathematical form (valid potential-based delta with product of contact×proximity×stillness, all bounded [0,1]), correct sign, and is not exploitable in principle. the failure is purely scale: 150.0 coefficient creates extreme dominance that destabilizes training.

**selected_intervention**: reduce settlement_bonus coefficient from 150.0 to 30.0. no other component changed.

**falsifiable_hypothesis**: reducing the settlement coefficient 5× should shrink its magnitude share from ~81% to ~50%, giving proximity_reward and speed_penalty_gated sufficient relative weight to guide the approach phase. the agent should stop being driven into crashing local optima by sparse settlement spikes and instead learn a gradual approach-then-land trajectory.

**expected_next_round**: settlement_bonus episode_sum_mean should drop proportionally (~5 instead of ~25 for same behavior); episode length should increase as crashing decreases; score should improve toward best (197.60) or beyond; early terminal count (<150 steps, score<-50) should decrease from 8/20.

**main_risk**: if 30.0 is too low, the final landing phase may lack sufficient incentive to complete the settle, resulting in the agent hovering near the platform without touching down — this would manifest as long episodes with good proximity but low settlement completion.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft_contact, nright_contact = next_obs

    # ----------------------------------------------------------------
    # 1. Goal proximity: potential-based delta shaping
    # ----------------------------------------------------------------
    distance = (x**2 + y**2)**0.5
    next_distance = (nx**2 + ny**2)**0.5
    proximity_reward = 2.0 * (distance - next_distance)

    # ----------------------------------------------------------------
    # 2. Orientation penalty: keep body upright and stable
    # ----------------------------------------------------------------
    orientation_penalty = -0.1 * (angle**2) - 0.1 * (angvel**2)

    # ----------------------------------------------------------------
    # 3. Soft landing: punish high velocities when near the pad
    # ----------------------------------------------------------------
    proximity_gate = 1.0 / (1.0 + 5.0 * distance)
    speed_sq = vx**2 + vy**2
    speed_penalty_gated = -0.5 * speed_sq * proximity_gate

    # ----------------------------------------------------------------
    # 4. Settlement improvement: potential-based delta (coefficient reduced)
    #    Reduced from 150.0 to 30.0 to prevent scale domination over
    #    proximity_reward and speed_penalty_gated during training.
    # ----------------------------------------------------------------
    # Current settled score (from obs)
    current_avg_contact = (left_contact + right_contact) / 2.0
    current_prox_gate = 1.0 / (1.0 + 5.0 * distance)
    current_speed_sq = vx**2 + vy**2
    current_stillness = 1.0 / (1.0 + 10.0 * (current_speed_sq + angvel**2))
    current_settled = current_avg_contact * current_prox_gate * current_stillness

    # Next settled score (from next_obs)
    next_avg_contact = (nleft_contact + nright_contact) / 2.0
    next_prox_gate = 1.0 / (1.0 + 5.0 * next_distance)
    next_speed_sq = nvx**2 + nvy**2
    next_stillness = 1.0 / (1.0 + 10.0 * (next_speed_sq + nangvel**2))
    next_settled = next_avg_contact * next_prox_gate * next_stillness

    # Full potential-based delta with reduced coefficient
    settlement_bonus = 30.0 * (next_settled - current_settled)

    # ----------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------
    total_reward = (
        proximity_reward +
        orientation_penalty +
        speed_penalty_gated +
        settlement_bonus
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "speed_penalty_gated": speed_penalty_gated,
        "settlement_bonus": settlement_bonus
    }

    return float(total_reward), components
```