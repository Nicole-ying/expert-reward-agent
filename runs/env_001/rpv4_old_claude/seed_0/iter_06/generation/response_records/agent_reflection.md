# Response Record

# Analysis

## 1. evidence
Iter 2 achieved 142.3 score with 65% termination rate (13/20) and 721 len. Iter 3-5 progressively degraded: iter 3 dropped to 37.0 (score -105 vs best), iter 4 recovered slightly to 103.6 (still -38 vs best), iter 5 collapsed to -38.4 with only 15% termination (3/20) and 952 len. The rejected draft attempted to revert goal_approach_progress coefficient from 80.0 back to 10.0, which is identical to iter 2 best—this is not a valid new intervention.

Per-step magnitudes in iter 5: goal_approach_progress ≈ 0.097/step, contact_landing_reward ≈ 0.010/step, velocity_brake ≈ -0.033/step, orientation ≈ -0.005/step. Goal_approach dominates but episodes fail to terminate successfully (truncated 17/20).

## 2. behavior_diagnosis
Agent learned approach behavior (goal_approach_progress active 100%, signed_share 58.7%) but fails to execute stable landing: only 3/20 episodes terminate successfully, 17/20 hit time limit. The policy reaches vicinity but cannot satisfy landing conditions, suggesting contact_landing_reward (active_rate 0.3%, episode_sum 9.6) is too weak to guide final touchdown phase.

## 3. signal_completeness
- Approach guidance: present (goal_approach_progress, delta-based)
- Safety constraints: present (velocity_brake, orientation_stability)
- Landing completion signal: present but ineffective (contact_landing_reward triggers in only 0.3% of steps, contributes 6.1% magnitude_share vs 70.5% for approach). The cubic-root product modulator collapses when any quality factor is low, making contact reward nearly zero during exploration.

## 4. selected_level
**Level 2: Mathematical form correction** on contact_landing_reward. Active_rate 0.3% indicates the product-based quality_factor collapses to near-zero during learning. The geometric mean `(v * a * p)^0.333` still multiplies three factors, causing collapse when agent has not yet learned to satisfy all three simultaneously.

## 5. selected_intervention
Replace the multiplicative quality_factor in contact_landing_reward with **additive averaging**: `quality_factor = (velocity_quality + angle_quality + proximity_quality) / 3.0`. This allows partial credit when agent satisfies any subset of conditions, enabling gradual learning rather than requiring joint perfection from the start.

## 6. falsifiable_hypothesis
Additive quality averaging will increase contact_landing_reward's active magnitude during training, allowing agent to learn contact-seeking behavior even when velocity/angle/proximity are not simultaneously optimal. This should increase termination rate and reduce episode length compared to iter 5.

## 7. expected_next_round
- contact_landing_reward active_rate should increase from 0.3% to >10%
- contact_landing_reward magnitude_share should increase from 6.1% to >20%
- terminated episodes should increase from 3/20 to >8/20
- episode length should decrease from 952 to <850
- score should recover toward iter 2 best (142.3) or exceed it

## 8. main_risk
If additive averaging makes contact reward too easy to obtain, agent may exploit by touching ground repeatedly without satisfying velocity/angle requirements, leading to crashes or unstable landings. Monitor for score regression or increased crash-like early terminations.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward with ADDITIVE quality averaging
    """
    
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    contact_active = max(next_left_contact, next_right_contact)
    
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    quality_factor = (velocity_quality + angle_quality + proximity_quality) / 3.0
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```
