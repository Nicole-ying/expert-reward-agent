# Response Record

**1. evidence** — score=139.53, all 20 episodes hit 1000-step truncation with 0 terminations; safe_contact_bonus dominates at 250.15 (90.4% magnitude, 73.7% active_rate) while progress_reward is only 2.71 (1.0%); agent is farming persistent contact state near the target without triggering the settled termination condition.

**2. behavior_diagnosis** — The agent reaches the target area, puts both legs down, and stays there collecting ~0.5 per step from the state-valued safe_contact_bonus, but never achieves full settling (likely due to continuous engine firing at 94.3% rate), causing all episodes to truncate at 1000 steps rather than terminate successfully.

**3. signal_completeness** — Progress guides approach, stability constrains velocity/angle, fuel penalizes engine use; the critical flaw is that safe_contact_bonus is a persistent state reward (`next_left * next_right`) that pays out continuously once both legs touch regardless of whether the task actually completes, creating an exploitable proxy that diverges from task completion.

**4. selected_level** — **Level 2**: `state_to_improvement` transformation. The evidence directly matches the "state reward exploitation" pattern — occupying the "both legs touching" state near the target yields ongoing reward without requiring the settled termination.

**5. selected_intervention** — Transform `safe_contact_bonus` from a persistent state reward to a **transition-based** reward: `(1 - obs[6]*obs[7]) * next_left * next_right` fires only when both legs newly achieve contact, not continuously. Raise `w_contact` from 0.5 to 10.0 to compensate for drastically reduced trigger frequency.

**6. falsifiable_hypothesis** — By eliminating the recurring contact payout, the agent can no longer farm the contact state; after landing, remaining signals (near-zero progress, small stability penalty, -0.02 fuel per firing) create pressure toward action=0 and low velocity, which should allow the settled termination to trigger, shortening episodes and raising external score toward 200.

**7. expected_next_round** — `safe_contact_bonus` episode_sum_mean should drop from ~250 to a small single-digit value (firing only 1-2 times per episode); terminated count should rise above 0 (ideally several episodes achieve settled termination); episode_length should decrease below 1000 for successful episodes; external score should increase toward target.

**8. main_risk** — The agent may learn to "bounce" (lose and regain dual-leg contact repeatedly) to farm the transition bonus; the proximity gate and stability penalty partially mitigate this, but if observed next round, a velocity-gated landing condition or cooldown may be needed.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_left = next_obs[6]
    next_right = next_obs[7]

    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward ------------------
    w_progress = 2.0
    progress = dist - next_dist
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated) ------------------
    gate = 1.0 / (1.0 + dist)
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe contact bonus (transition-based) ------------------
    w_contact = 10.0
    prox_factor = 1.0 / (1.0 + 10.0 * next_dist)
    contact_before = obs[6] * obs[7]         # 1.0 if both legs already touching
    contact_now = next_left * next_right      # 1.0 if both legs touching after step
    contact_transition = (1.0 - contact_before) * contact_now  # 1.0 only on newly achieved dual contact
    safe_contact_bonus = w_contact * prox_factor * contact_transition

    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }

    return float(total_reward), components
```
