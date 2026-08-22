# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
    hull_angle = obs[0]             # body pitch angle
    hull_angvel = obs[1]            # body angular velocity
    horizontal_speed = obs[2]       # forward speed

    # 1. Forward progress (main learning signal)
    #    Direct dense reward for forward velocity; negative speed penalized.
    forward_progress = horizontal_speed   # coefficient = 1.0

    # 2. Balance and fall‑prevention penalty
    #    Quadratic hinge: only penalise when outside safe region.
    angle_threshold = 0.4       # rad, ~23°, safe swing allowed
    angvel_threshold = 1.0      # rad/s

    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angvel_excess = max(0.0, abs(hull_angvel) - angvel_threshold)

    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    total_reward = forward_progress + balance_penalty

    components = {
        'forward_progress': forward_progress,
        'balance_penalty': balance_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- `task_family`: locomotion_continuous_control
- `dynamics_subtype`: planar_bipedal_gait (torque‑controlled 4‑DoF biped)

## selected reward roles
1. **forward_progress** (mandatory) – the main learning signal.
2. **balance_penalty** (mandatory) – a stability/safety constraint that also serves as a soft fall‑prevention term.

## role‑to‑signal mapping
| role | signal | index / source |
|------|--------|----------------|
| forward_progress | horizontal_speed | `obs[2]` |
| balance_penalty | hull_angle, hull_angular_velocity | `obs[0]`, `obs[1]` |

## chosen formula operators
- **forward_progress**: `dense_state_signal` – linear positive reward on `horizontal_speed`.  
  The signal is used raw (coefficient = 1.0) so that negative speed (backward motion) yields negative reward, actively discouraging retreat.
- **balance_penalty**: `dense_state_signal` (hinge + quadratic penalty).  
  `max(0, |value| - threshold)²` provides zero penalty inside a safe band and smooth gradient when the boundary is crossed. The quadratic term for `hull_angle` is deliberately weighted higher (3.0) to strongly deter large tilts that would otherwise lead to a fall.

## excluded roles and why
- **fall_termination_penalty** – cannot be implemented safely because the step function lacks a `terminated` flag, and `info` is empty. The risk of misclassification is too high. Instead, large‑angle tilt is handled continuously by the quadratic hinge penalty, which naturally produces a strong deterrent when the body angle approaches dangerous values.
- **energy_efficiency** – deferred to a later iteration to avoid suppressing exploration early in training.
- **alive_bonus** – not used because the forward‑progress reward already incentivises staying alive while moving forward; a flat per‑step bonus could encourage “stand still and survive” behaviour.
- **goal_reaching_bonus** – excluded because no reliable signal for reaching the end of the terrain exists (no position, no explicit success flag).

## why no terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available = false`, `explicit_failure_flag_available = false` as stated in the environment card.  
- Failure is instead addressed by the continuous tilt penalty that grows sharply when the body angle leaves the safe region, providing a gradient‑rich predecessor to the catastrophic failure.

## roles deferred to later iterations
- `energy_efficiency` (minor action‑magnitude penalty) once stable walking is established.
- `smart_terrain_anticipation_reward` (lidar‑based preview) only if future failure analysis shows a need for anticipatory gait adjustments.

## expected failure modes to observe during training
- **velocity burst then fall**: agent may still accelerate aggressively if the forward reward dominates before the tilt penalty catches up; watch for episodes with high early speed followed by abrupt termination.
- **cautious shuffling**: if balance penalty is too high, agent might minimise tilt by taking tiny steps, leading to very low forward progress.
- **backward evasion**: because negative speed is penalised, agent might occasionally step backward when close to obstacles – acceptable as long as it does not become the dominant gait.
- **utilisation of lidar** is not incentivised, so the policy may ignore terrain preview entirely; this is acceptable for v1 while the primary gait is learned.