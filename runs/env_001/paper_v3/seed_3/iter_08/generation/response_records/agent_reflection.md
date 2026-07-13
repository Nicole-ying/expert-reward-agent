# Response Record

## Diagnosis

1. **evidence**: Score=34.64, len=955, 15/20 truncated at ~1000-step ceiling, 5/20 terminated (likely successful landings). approach_improvement contributes only 13.82/episode (bounded by initial distance × weight), while contact_reward dominates at 60.3 despite 2.1% active_rate. The improvement-based approach has a hard ceiling and provides negligible per-step gradient once near the target.

2. **behavior_diagnosis**: The agent survives long episodes (~955 steps) without crashing but lacks urgency — it drifts or hovers inefficiently, only managing to land in 5/20 episodes before hitting the time limit. The approach signal is too weak to drive efficient descent.

3. **signal_completeness**: All four mandatory roles (approach, soft-landing, stabilization, contact) are structurally present, but the approach signal is crippled by its improvement-based form — total possible reward is capped at `initial_distance × 10.0`, and per-step reward collapses to zero when the agent stops making forward progress or oscillates.

4. **selected_level**: **Level 2** — `state_to_improvement` transformation. The improvement-based approach (`max(0, delta_dist)`) has a hard ceiling and provides zero reward when the agent hovers or moves laterally without net distance reduction. The truncated-episode evidence directly shows the agent lacks sufficient per-step guidance to complete the task within the time budget. Scale increase alone cannot fix a bounded-by-initial-distance signal.

5. **selected_intervention**: Replace `approach_improvement = w * max(0, prev_dist - curr_dist)` with a state-based bounded proximity reward `w / (1.0 + curr_dist)`. This provides continuous per-step feedback at every distance, unbounded over the episode, with strong gradient near the target where precision matters most.

6. **falsifiable_hypothesis**: A state-based proximity reward removes the hard ceiling on approach signal, giving the agent continuous per-step gradient toward the target regardless of step-to-step delta. This should reduce the truncated rate (more episodes landing before timeout) and raise the average score, because the agent is no longer starved of approach feedback when hovering or making slow progress.

7. **expected_next_round**: `approach_reward` magnitude_share should rise substantially (from 18.2% toward majority), episode_length should decrease (fewer truncations), score should increase as more episodes terminate with successful contact, and the score_range lower bound should rise above negative values.

8. **main_risk**: A pure state-based proximity reward risks the `state_to_improvement` pitfall — the agent could learn to hover near the pad collecting proximity reward without actually landing. The existing distance-gated velocity penalty and contact reward are the safeguards; if the agent plateaus at medium distance, the velocity gate may need tightening.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state (post-action)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: State-based proximity reward (replaces improvement-based) ---
    # Continuous per-step reward that grows as distance to pad shrinks.
    # Bounded above at 1.0 (when dist=0), provides gradient at all distances.
    curr_dist = (x**2 + y**2) ** 0.5
    w_approach = 1.0
    comp_approach = w_approach / (1.0 + curr_dist)

    # --- Component 2: Soft landing velocity penalty (distance-gated) ---
    gate_vel = 1.0 / (1.0 + curr_dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel
    w_vel = 0.1
    comp_soft_landing = -w_vel * vel_penalty

    # --- Component 3: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    comp_stabilization = -w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 4: Successful contact reward ---
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach + comp_soft_landing + comp_stabilization + comp_contact
    components = {
        "approach_proximity": comp_approach,
        "soft_landing_velocity": comp_soft_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components
```
