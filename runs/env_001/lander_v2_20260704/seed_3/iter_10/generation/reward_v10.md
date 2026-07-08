Based on the training feedback, I can now diagnose and intervene.

## Diagnosis

### 1. Evidence
All 20 episodes hit the 1000-step truncation limit with score=-33.97; `proximity_quality` contributes 809.85 per episode (99.8% magnitude share) despite zero contact (`contact_bonus` active_rate=0%). The agent collects massive reward simply by hovering near the platform without ever touching down.

### 2. Behavior Diagnosis
The agent exploits `proximity_quality` — a dense, always-active proximity signal gated only on distance/speed/angle, never on contact. It learns to hover stably near the platform (low speed, low angle, moderate distance) for the full 1000 steps, collecting ~0.81/step without completing the landing task.

### 3. Signal Completeness
The current reward lacks the critical **contact gate**: `proximity_quality` is structurally decoupled from task completion. The `contact_bonus` (additive, ungated on quality) never fires because the agent has no incentive to make contact. The best historical version (iter 7, score=198.14) had a contact-gated `soft_landing_proxy`, which forced the agent to land to receive the main reward.

### 4. Selected Level
**Level 2: `proxy_to_completion_alignment`** — the dense proxy signal rises without task completion. Evidence directly shows the proxy is exploitable (809/episode while never landing). A structural transformation is required: re-gate the landing quality signal on foot contact.

### 5. Selected Intervention
Replace the decoupled `proximity_quality` + `contact_bonus` pair with a single **contact-gated `soft_landing_proxy`**, reverting to the structure that worked in iter 7. One evidence-based modification from the best version: use rational decay functions (`1/(1+kx)`) instead of exponentials, providing wider gradient for landing-quality improvement once contact is made, while the contact gate still prevents the pre-contact hover exploit.

### 6. Falsifiable Hypothesis
With `contact_avg` as a multiplicative gate, the proxy becomes zero when no foot touches the platform. The agent must make contact to access the main reward stream, eliminating the hover exploit. The rational decay (vs. exponential) should preserve gradient at moderate deviations, helping the agent refine landing posture after initial contact.

### 7. Expected Next Round
- `soft_landing_proxy` magnitude_share still dominant but with `active_rate` > 0% (contact required)
- `contact_bonus` removed; contact signal now integrated into proxy gate
- Episodes should begin terminating before truncation (len < 1000), indicating actual landings
- Score should improve substantially over -33.97, approaching or exceeding the best of 198.14

### 8. Main Risk
If the rational decay is too permissive, the agent may accept sloppy landings (moderate speed/angle at contact) and score less than the best version. If so, next round should tighten decay constants or revert to exponentials.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. Progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraints: light penalty on speed, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * speed - 0.01 * angle - 0.005 * ang_vel

    # 3. Contact-gated landing quality: zero when no foot contact, rewards good landing posture.
    #    Rational decay 1/(1+kx) preserves gradient farther from optimum than exponentials.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    dist = d_next
    soft_landing_proxy = 3.0 * contact_avg / ((1.0 + 2.0 * dist) * (1.0 + speed) * (1.0 + 5.0 * angle))

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```