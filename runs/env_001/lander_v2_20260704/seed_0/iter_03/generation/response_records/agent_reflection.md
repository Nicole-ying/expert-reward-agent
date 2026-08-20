# Response Record

## 1. evidence
18/20 episodes terminate early with score < -50, episode length ~89 steps, and the `soft_landing_proxy` fires in only 0.6% of steps (≈0.5 steps/episode), meaning the landing signal is effectively absent during learning; `distance_reward` is strongly positive (6.15/episode) showing the agent learns to move toward target but then crashes.

## 2. behavior_diagnosis
The agent rushes toward the target (fueled by scaled-up distance_reward from iter 1→2) but arrives at speed and crashes, because it receives virtually no feedback about slowing down, leveling out, or making gentle leg contact—the sparse all-or-nothing landing bonus never activates during exploration.

## 3. signal_completeness
Progress toward target is well-covered by `distance_reward`, and a weak stability hint exists, but the critical "decelerate and land gently" signal is essentially absent: `soft_landing_proxy` has an active_rate of 0.6%, far below the ~1% threshold where sparse events become unreachable for credit assignment.

## 4. selected_level
**Level 2** — the `soft_landing_proxy` active_rate is near zero and its hard-binary mathematical form collapses feedback, directly matching the `sparse_to_dense` trigger: "completion/landing bonus 的 active_rate 低于约1%时，通常无法提供可达的学习引导。"

## 5. selected_intervention
Transform `soft_landing_proxy` from a hard five-condition binary bonus into a continuous `landing_quality` shaping component: proximity-gated sum of bounded quality factors (speed, angle, angular velocity, leg contact), each normalized to [0,1], with coefficient scaled to provide meaningful but not dominant gradient.

## 6. falsifiable_hypothesis
Providing continuous, partially-achievable landing feedback will let the agent discover deceleration and stabilization near the target, reducing crash rate and improving external score without creating a proxy-exploit plateau.

## 7. expected_next_round
`landing_quality` active_rate should rise substantially from 0.6% to at least 20-40%; early termination rate should drop; external score should improve (less negative); episode length may increase as fewer episodes end in early crashes.

## 8. main_risk
A proximity-gated continuous proxy can create a local optimum where the agent hovers near the target collecting partial credit without committing to final touchdown—if score improves but landings remain rare next round, further gating or a completion-alignment transform will be needed.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current distance to target (from current obs)
    dist_before = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # Next distance to target (from next_obs)
    dist_after = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Progress signal: reward reduction in distance, penalize increase
    distance_reward = (dist_before - dist_after) * 5.0

    # 2. Stability constraint: discourage high speed, tilt, and rotation
    stab_weight = 0.01
    stability_penalty = -stab_weight * (
        abs(next_obs[2]) + abs(next_obs[3]) + abs(next_obs[4]) + abs(next_obs[5])
    )

    # 3. Continuous landing quality (replaces sparse binary soft_landing_proxy)
    # Proximity gate: activates gradually within 0.5 distance of target
    proximity_gate = max(0.0, 1.0 - dist_after / 0.5)

    # Quality factors, each in [0, 1]
    speed_quality = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)
    angle_quality = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    angvel_quality = max(0.0, 1.0 - abs(next_obs[5]) / 0.3)
    contact_score = (next_obs[6] + next_obs[7]) / 2.0

    # Landing quality: proximity-gated average of quality factors, scaled
    landing_quality = 2.0 * proximity_gate * (
        speed_quality + angle_quality + angvel_quality + contact_score
    ) / 4.0

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```
