# Response Record

Now let me work through the decision process.

---

## 1. evidence
Score=197.60 (98.8% of target 200), settlement_bonus dominates at 715.78 mean (98.6% magnitude share), proximity_reward is PBRS at 2.73, orientation_penalty and speed_penalty_gated are small negative, 11/20 terminated (settled) and 9/20 truncated (time limit), and the rejected draft was semantically identical to iter 5 which already achieved 197.60.

## 2. behavior_diagnosis
The agent succeeds in settling on 11/20 episodes and farms massive continuous settlement reward, but 9/20 episodes hit the time limit without ever achieving stable settlement—the persistent state-value bonus rewards those who settle but fails to push the marginal episodes over the line because it pays for duration rather than transition.

## 3. signal_completeness
All necessary roles are present (proximity, soft landing, orientation), and the settlement bonus provides dense feedback, but the continuous state-value form makes it farmable and does not specifically reward the act of achieving settlement over occupying it.

## 4. selected_level
**Level 2**: The same skeleton family has been used for iter 3, iter 5, and the rejected duplicate; the continuous settlement bonus is a persistent state value being farmed (68.2% active but paying 715 per episode), and the `state_to_improvement` + `persistent_to_transition_event` patterns directly match the evidence of proxy farming without completion for the truncated episodes.

## 5. selected_intervention
Transform `settlement_bonus` from a continuous state-value reward (`2.0 * avg_contact * proximity_gate * stillness`) into a transition-based improvement reward (`coefficient * max(0, settled_score(next) - settled_score(current))`), targeting the settlement component exclusively.

## 6. falsifiable_hypothesis
By rewarding only the improvement in settled-ness rather than duration of occupation, the agent can no longer farm the settled state, and the marginal 9 truncated episodes will receive sharper gradient toward the transition into full settlement, increasing the termination rate and closing the 2.40 score gap.

## 7. expected_next_round
- settlement_bonus episode_sum_mean should drop substantially (no longer accumulating per-step)
- settlement_bonus active_rate may increase (more steps with positive delta during approach)
- terminated count should rise above 11/20
- score should increase toward or past 200

## 8. main_risk
Without continuous reward for staying settled, the agent may oscillate near the pad (settle, drift, re-settle) to harvest multiple transition rewards; the speed_penalty_gated and orientation_penalty must counteract this, but if they are too weak, the agent could exploit the transition structure.

---

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
    # 4. Settlement improvement bonus (TRANSITION-BASED)
    #    Rewards the delta in settled_score, not persistent occupation.
    #    settled_score = avg_contact * proximity_gate * stillness
    #    Only positive improvements are rewarded; regression is not penalized
    #    (speed_penalty and orientation_penalty already cover regression).
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

    # Reward only improvement: becoming more settled than before
    settlement_bonus = 400.0 * max(0.0, next_settled - current_settled)

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
