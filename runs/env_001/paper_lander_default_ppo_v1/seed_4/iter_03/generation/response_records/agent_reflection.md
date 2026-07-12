# Response Record

## 1. evidence
contact_bonus dominates at 95.7% magnitude share (249.44 mean, active_rate 68.9%) while proximity_reward sits at only 1.0%; episodes average 749 steps with only 6/20 terminating — the agent is collecting persistent contact rewards on the pad without ever becoming still enough to trigger `body_not_awake_or_settled`.

## 2. behavior_diagnosis
The agent reliably reaches the pad and achieves double-contact, but then "rests" there making tiny residual movements that keep the body awake while continuously farming contact_bonus, preventing settlement termination.

## 3. signal_completeness
Proximity (potential-based delta), soft-landing (gated speed penalty), and orientation penalty are all present and functional; the critical missing piece is a *settlement-completion* signal — the current contact_bonus rewards the persistent state of being in contact rather than the terminal condition of being truly still.

## 4. selected_level
**Level 2**: `persistent_to_transition_event` / `proxy_to_completion_alignment` — the contact_bonus is a state-value proxy that can be farmed indefinitely without task completion; it must be restructured to require the stillness that naturally precedes termination.

## 5. selected_intervention
Replace `contact_bonus` (persistent contact-on-pad reward) with `settlement_bonus`: multiply the contact×proximity gate by a *stillness* factor `1/(1+10·(speed²+angvel²))` evaluated on `next_obs`, so the bonus only fires when the agent is nearly motionless — a state from which `body_not_awake_or_settled` should immediately terminate the episode.

## 6. falsifiable_hypothesis
The stillness gate removes the farming plateau: agents can only earn meaningful settlement_bonus when they are truly still, which is exactly the condition that triggers environment termination; this should convert the 14 truncated hover-episodes into successful settlements and push the score toward or past 200.

## 7. expected_next_round
`settlement_bonus` active_rate should collapse from 68.9% to a small fraction (only the final pre-termination steps); `terminated` count should rise substantially from 6/20; `episode_length` should drop; `score` should increase as more episodes reach successful settlement instead of timing out.

## 8. main_risk
The stillness threshold may be miscalibrated — if too strict the settlement signal vanishes entirely and agents lose incentive to land at all; if too loose agents may still find a narrow wiggle-room to farm without triggering `body_not_awake_or_settled`.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft_contact, nright_contact = next_obs

    # ----------------------------------------------------------------
    # 1. Goal proximity: potential-based delta shaping
    #    Positive when moving toward target, negative when moving away
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
    # 4. Settlement bonus: only rewards being truly still with both
    #    contacts near the pad.  Stillness is the missing ingredient
    #    that separates "farming contact on the pad" from "settled".
    #    Evaluated on next_obs so the bonus reflects the state after
    #    the action, aligning with the environment's settle detector.
    # ----------------------------------------------------------------
    next_proximity_gate = 1.0 / (1.0 + 5.0 * next_distance)
    next_speed_sq = nvx**2 + nvy**2
    stillness = 1.0 / (1.0 + 10.0 * (next_speed_sq + nangvel**2))
    settlement_bonus = 2.0 * nleft_contact * nright_contact * next_proximity_gate * stillness

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
