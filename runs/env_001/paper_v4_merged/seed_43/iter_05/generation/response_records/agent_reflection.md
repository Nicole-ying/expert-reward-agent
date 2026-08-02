# Response Record

1. evidence: score=-87.2, length=143.7, 9/20 early terminal; landing_speed_gate has mean 127.3, 96.4% share, but is a multiplicative factor, not an intentional reward, causing signal confusion; all episodes terminate (no truncation), often by crash or out-of-bounds, and posture signal is unused.
2. behavior_diagnosis: The agent is rewarded primarily by the spurious gate component (sum ≈127 per episode) rather than progress toward the pad, leading to optimization for high gate values (e.g., slowing near the pad) without sustained advancement, resulting in negative overall score and many early crash terminations.
3. signal_completeness: Core signals (potential‑based progress, contact reward, action cost) exist; however, the landing_speed_gate as an independent reward dominates and corrupts the gradient, while missing posture constraints (body_angle, angular_velocity) leave crashes unpenalised.
4. selected_level: Level 2 – structural change, because the landing_speed_gate component is active 100% of the time and contributes ~100% signed share as a non‑reward artefact, requiring removal from the component output to restore signal integrity.
5. selected_intervention: Remove `landing_speed_gate` from the returned components dictionary; keep its computation and multiplication intact (used for shaping) but stop emitting it as a reward term.
6. falsifiable_hypothesis: Eliminating the parasitic gate reward will force the agent to rely on `shaped_progress` (real distance reduction) and `landing_contact_reward`, which should keep the score stable or gradually improve over subsequent iterations, and will make room for later posture‑related signals to reduce early terminations.
7. expected_next_round: The `landing_speed_gate` component will vanish from statistics, `shaped_progress` will become the dominant positive signal, and the evaluation score should not drop (predicted range −90 to −80); episode length may remain similar.
8. main_risk: The positive incentive may become too weak after gate removal, causing training to stall or score to decrease; a subsequent coefficient boost for `shaped_progress` may then be needed.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    left_contact = float(next_obs[6])
    right_contact = float(next_obs[7])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Landing speed gate (internal factor, not a reward component) ----------
    # Encourages low speed when close to the pad.
    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)   # 0 at >=0.5, 1 at 0
    speed_cost_input = speed_next * proximity_factor
    landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_cost_input)

    # Apply gate to progress shaping (no separate reward component)
    shaped_progress = progress_shaping * landing_speed_gate

    # ---------- 3. Efficiency: action penalty ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. Landing contact bonus ----------
    contact_sum = left_contact + right_contact
    contact_factor = contact_sum / 2.0
    proximity = max(0.0, 1.0 - dist_next / 0.8)
    landing_contact_reward = 0.2 * contact_factor * proximity

    total_reward = shaped_progress + action_cost + landing_contact_reward

    components = {
        "progress_shaping": progress_shaping,
        "shaped_progress": shaped_progress,
        "action_cost": action_cost,
        "landing_contact_reward": landing_contact_reward
    }

    return float(total_reward), components
```
