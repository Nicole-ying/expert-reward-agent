def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unbox next_obs for the state after the action
    next_x  = next_obs[0]
    next_y  = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle     = next_obs[4]
    next_angvel    = next_obs[5]
    left_contact   = next_obs[6]
    right_contact  = next_obs[7]

    # --- helper for exponential (e^z) ---
    def exp(z):
        return 2.718281828 ** z

    # --- helper for tanh using e^(-2z) ---
    def tanh(z):
        e_neg = exp(-2.0 * z)
        return (2.0 / (1.0 + e_neg)) - 1.0

    # ========== 1. approach_signal (main driving term) ==========
    # encourage being close to the landing pad: r ∈ [0, 2]
    dist_next = (next_x**2 + next_y**2) ** 0.5
    alpha = 2.0
    r_approach = 2.0 * (1.0 - tanh(alpha * dist_next))

    # ========== 2. stability_penalty (safety / smoothness) ==========
    w_vel    = 0.02
    w_angle  = 0.02
    w_angvel = 0.02
    r_stability = (
        - w_vel    * (abs(next_vx) + abs(next_vy))
        - w_angle  * abs(next_angle)
        - w_angvel * abs(next_angvel)
    )

    # ========== 3. landing_bonus (soft proxy for successful landing) ==========
    # only fires when the agent is almost perfectly on the pad,
    # very slow, upright, and both legs touching.
    contact_factor = 0.5 * (left_contact + right_contact)
    angle_factor   = 1.0 / (1.0 + 2.0 * next_angle**2)

    w_land = 30.0
    r_land = w_land * (
        exp(-10.0 * next_x**2) *
        exp(-10.0 * next_y**2) *
        exp(-5.0  * next_vx**2) *
        exp(-5.0  * next_vy**2)
    ) * angle_factor * contact_factor

    # ========== total ==========
    total_reward = r_approach + r_stability + r_land

    components = {
        'approach_signal':  r_approach,
        'stability_penalty': r_stability,
        'landing_bonus':     r_land,
    }

    return float(total_reward), components