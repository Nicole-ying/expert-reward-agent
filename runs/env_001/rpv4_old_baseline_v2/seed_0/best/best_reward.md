# Reward Component Analysis

## 1. Global score assessment
The current policy achieves an average score of **-70.35**, far below the target of **200**. All 20 evaluation episodes are truncated at 1000 steps (20/20 truncated) with zero early successes. This indicates that the agent never learns to land successfully and the reward programme is fundamentally misspecified. **A complete rewrite of the reward function is required**, moving away from a bounded-proximity-plus-sparse-landing framework to one that directly drives descent, deceleration, leg contact, and uprightness.

## 2. Component-by-component diagnosis

### proximity
- **episode_sum_mean**: 1448  
- **signed_share**: 96.7%  
- **active_rate**: 100%  
- **Trajectory**: extremely high magnitude, dominating all other terms. It encourages staying near the target in *position space* but provides almost no differentiation between hovering safely at \(y \approx 0.3\) and actually touching the pad with zero velocity. The bounded form `1/(1+│x│)+1/(1+│y│)` saturates early, removing any incentive to reach exactly zero or to control speed/angle. **This component must be replaced** by an unbounded penalty that pushes \(x\) and \(y\) all the way to zero.

### landing_reward
- **episode_sum_mean**: 0  
- **active_rate**: 0% (flat, never triggered)  
- **Cause**: the threshold `y < 0.2` was never crossed because the proximity term gives nearly the same reward at slightly higher \(y\) and the agent avoids the risk of crashing. Even if triggered, the multiplicative design without any intermediate shaping makes it a sparse signal that RL cannot exploit. **This component is useless and must be replaced** by a continuous, contact-based reward that activates gradually as legs touch the ground and the vehicle stabilises.

### fuel_penalty
- **episode_sum_mean**: -50  
- **signed_share**: -3.3%  
- **active_rate**: 100%  
- **Magnitude**: tiny relative to proximity (3.3% magnitude share). The penalty `-0.05` per nonzero action is far too small to deter wasteful engine use. It can be kept, but scaled slightly and balanced against the new reward structure.

## 3. Design of the new reward
The new reward abandons bounded proximity and sparse landing. It uses:
- **Unbounded linear penalties** on \(x, y\) to force reaching the origin.
- **Velocity and angle penalties** to demand a soft, upright touchdown.
- **A continuous contact-stability reward** that gives positive reinforcement when legs touch the pad **and** velocity/angle are low. This term never relies on a hard height threshold; it is active whenever there is leg contact, creating a smooth gradient toward stable landing.
- **A small fuel penalty** to promote efficient engine use.

By tuning the relative magnitudes (dist ~ 0.2, vel ~ 0.1, angle ~ 0.5, landing up to ~10), the agent receives a strong gradient to descend, decelerate, touch the pad, and settle. The total reward in a successful short episode should be positive and can approach the desired score.

# Revised Reward Function

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack relevant observations
    x = obs[0]          # horizontal distance to target pad
    y = obs[1]          # vertical distance (height) above pad
    x_vel = obs[2]      # horizontal velocity
    y_vel = obs[3]      # vertical velocity (positive upward)
    angle = obs[4]      # body pitch angle
    left_contact = obs[6]
    right_contact = obs[7]

    # 1. Distance penalty – unbounded, drives x and y to zero
    dist_penalty = -0.2 * (abs(x) + abs(y))

    # 2. Velocity penalty – encourages low speeds for soft landing
    vel_penalty = -0.1 * (abs(x_vel) + abs(y_vel))

    # 3. Angle penalty – keeps the vehicle upright
    ang_penalty = -0.5 * abs(angle)

    # 4. Continuous contact-stability reward – activated by leg contact,
    #    scaled by how stable the arrival is.
    contact_factor = (left_contact + right_contact) / 2.0
    # stability is high when speed and angle are small
    stability = 2.0 / (1.0 + abs(x_vel) + abs(y_vel) + 10.0 * abs(angle))
    landing_reward = 5.0 * contact_factor * stability

    # 5. Fuel efficiency penalty – small disincentive for any engine use
    fuel_penalty = -0.02 * (1.0 if action != 0 else 0.0)

    # Total reward
    total_reward = dist_penalty + vel_penalty + ang_penalty + landing_reward + fuel_penalty

    components = {
        "dist_penalty": dist_penalty,
        "vel_penalty": vel_penalty,
        "ang_penalty": ang_penalty,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```