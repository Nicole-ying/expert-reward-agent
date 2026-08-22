# Response Record

`evidence`: V6 bounded quadratic bonus (w=0.15) caused 8/20 early terminations (40% crash rate) vs V3 best with 0/20 crashes; forward_progress per surviving step was higher in V6 (~2.03) than V3 (~1.80), indicating the agent was pushed into unstable speed regimes; stability penalties remain tiny at ~3.4% of total reward magnitude, insufficient to deter crash-prone behavior; last round changed speed bonus from unbounded to bounded — it made things worse, so repeating that pattern is excluded.

`behavior_diagnosis`: The agent is over-incentivized for speed and under-penalized for instability: it achieves higher forward velocity but crashes in 40% of episodes, pulling the average score down by ~790 points from best; the 60% of episodes that survive are fast but the instability cost dominates the evaluation mean.

`signal_completeness`: Forward progress and stability penalties are present and active (100% active_rate); no missing essential signal is evident from the evidence — the issue is relative scale, not missing duties; a healthy height/action penalty is not required for basic locomotion given V3's prior success at 3260 without them.

`selected_level`: Level 1 — the stability penalty components have correct sign and mathematical form (linear absolute value), but their magnitude relative to forward_progress (~2-3%) is too small to constrain the agent when the speed bonus pushes it toward unstable gaits; this is a pure scale imbalance, not a missing signal.

`selected_intervention`: Starting from V3 best structure, increase both stability penalty weights in a single coherent adjustment — w_angle from 0.5 to 0.8, w_angvel from 0.1 to 0.2 — while keeping forward_progress (1.0) and forward_speed_bonus (0.1, unbounded v²) unchanged.

`falsifiable_hypothesis`: Stronger stability penalties will force the agent to learn more robust, controlled locomotion patterns that maintain torso uprightness and low angular velocity even when exploring higher speeds; this should eliminate the 40% crash rate while preserving or improving forward velocity, pushing the score beyond V3's 3260 plateau toward the 3800 target.

`expected_next_round`: Episode length should return to ~1000 with 0/20 or near-zero early terminations; score should exceed 3260; torso_angle_penalty and torso_angular_velocity_penalty magnitude_share should increase from ~3.4% to ~6-8% without collapsing forward_progress; forward_speed_bonus may initially decrease as the agent adapts to stability constraints, then recover as it finds efficient fast-stable gaits.

`main_risk`: Over-penalization could make the agent overly conservative, reducing forward velocity and causing the score to stagnate below V3's 3260; if this occurs, the next round should explore a Level 2 structural change rather than further scale tweaking.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V7 reward: V3 best structure + strengthened stability penalties (Level 1 scale fix).
    
    Evidence: V6's bounded speed bonus caused 40% crash rate. V3 was stable at 3260.
    Hypothesis: stability penalties at ~3% of reward are too weak to constrain high-speed
    exploration. Increasing them to ~6-8% should allow safe exploration of faster gaits.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress (unchanged from V3)
    w_fwd2 = 0.1         # quadratic bonus for higher speeds (unchanged from V3)
    w_angle = 0.8        # torso tilt penalty (V3: 0.5, strengthened for crash prevention)
    w_angvel = 0.2       # torso angular velocity penalty (V3: 0.1, strengthened)

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Quadratic bonus: disproportionately rewards higher forward speeds
    forward_speed_bonus = w_fwd2 * (v_fwd ** 2)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "forward_speed_bonus": forward_speed_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components
```
