# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 1.0 * (prev_dist - next_dist)   # delta reward

    # --- soft landing velocity penalty: quadratic penalty on velocity components ---
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2)

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: reward simultaneous leg contact, modulated by horizontal centering ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian-like decay with horizontal offset
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 50.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```

# reward_v1 设计说明

## Selected task_family / dynamics_subtype
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **control_type**: discrete (4 actions)

## Selected reward roles
Following environment_card.md’s mandatory roles:
1. **goal_proximity** → `dense progress signal` via improvement_delta
2. **soft_landing_velocity_penalty** → `safety / health constraint` via quadratic_penalty
3. **upright_orientation** → `stability constraint` via quadratic_penalty
4. **contact_success** → `task completion proxy` via joint_condition_proxy combined with bounded_signal decay

## role_to_signal_mapping and formula operators used

| role | signals used | operator | rationale |
|------|--------------|----------|-----------|
| `goal_proximity` | `next_obs[0], obs[0], next_obs[1], obs[1]` | **improvement_delta** (previous_distance – next_distance) | Pushes agent to continually reduce Euclidean distance to pad; avoids stagnation because no reward is given for holding a fixed distance. |
| `soft_landing_velocity_penalty` | `next_obs[2], next_obs[3]` | **quadratic_penalty** on both velocity components | Squared penalty provides smooth gradient and strongly discourages high velocities, especially near ground. |
| `upright_orientation` | `next_obs[4], next_obs[5]` | **quadratic_penalty** on body angle and angular velocity | Squared form penalizes large deviations smoothly, encouraging stable upright posture. |
| `contact_success` | `next_obs[6], next_obs[7]` (binary contacts), `next_obs[0]` (horizontal position) | **joint_condition_proxy** (both legs must touch) multiplied by **bounded_signal** (Gaussian centering decay) | Large sparse bonus only when both legs contact, strongly modulated by horizontal offset to prevent exploiting landings far from the pad. |

## Excluded roles and reasons
- **explicit_success_flag_reward**: info is always `{}`; no success flag available.
- **explicit_failure_penalty**: no failure flag; crashes and out-of-bounds are only detectable from termination, which we do not observe explicitly.
- **fuel_efficiency** (action penalty): deferred to later iterations to avoid suppressing early exploration.
- **angular_velocity_damping_near_surface**: conditionally gated by altitude; not introduced in v1 to keep design simple.
- **per_step_time_penalty**: constant negative penalty omitted to avoid risky urgent behavior before safe landing is established.

## Why no terminal_success_reward or terminal_failure_penalty
The environment **does not provide** any explicit success/failure flag (`info` is empty) and does not expose termination reasons. The only way to infer a soft landing is through the combination of `next_obs` states (both contacts + low velocity + upright), which we already reward with `contact_bonus` and penalize speed/attitude errors. Adding a terminal‑event operator would require us to guess termination from observations, but since termination happens right after the step we process, the same state is already covered by the per‑step components.

## Responsibilities left for future iterations
- **Fuel efficiency** (`action` penalty) once basic landing is reliable.
- **Altitude‑gated angular‑velocity damping** to suppress last‑moment rotation.
- **Adaptive gate** (e.g., `soft_health_gate`) that scales down `goal_proximity` when tilt or speed becomes dangerously high.
- **Potential‑based shaping** if simple delta proves insufficient for complex trajectories.
- **Temporal urgency** (small per‑step negative) only after success rate is high.

## Training failure modes to monitor
1. **High‑speed landing**: agent crashes into pad with large vertical velocity but still gets contact bonus; check `velocity_penalty` weight and consider making contact bonus conditionally dependent on low speed.
2. **Hovering**: agent stays upside, near target but never descends; may need to add gradual descent encouragement (e.g., potential that increases with lower altitude).
3. **Single‑leg landing**: agent tilts excessively, only one leg contacts; `contact_bonus` only triggers on double contact and `upright_penalty` should prevent this. If persists, increase orientation penalty weight.
4. **Off‑pad touchdown**: both legs contact but horizontal offset is large; centering decay mitigates this, but may need stronger decay or additional horizontal‑position penalty in `contact_bonus`.
5. **Excessive engine use**: agent keeps firing main engine after settled; later add fuel cost to teach engine‑off resting.