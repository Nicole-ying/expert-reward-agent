Now I have complete evidence. Let me provide my full analysis.

---

## Full Diagnosis: Why v2 is Catastrophically Broken

### The Big Picture

| Metric | v1 (iter 1) | v2 (iter 2, current) |
|---|---|---|
| Score | **-2.98** | **-113.74** |
| Episode length | 813.4 | 68.4 |
| Terminated (success) | 5/20 | 20/20 (all failures) |
| Agent behavior | Approaches target, hovers, rarely lands | Crashes immediately |

v2 took a working foundation (v1: score=-2.98, survived 813 steps, learned to approach the target) and destroyed it. The agent now terminates in 68 steps — all failures. No checkpoint shows any learning: every component is flat from cp10% to cp100%, meaning the reward signal provides **zero usable gradient**.

### Component-by-component diagnosis

| Component | Evidence | Root Cause | Action |
|---|---|---|---|
| `approach` (5× delta) | Flat ~8.49 across all checkpoints. Agent lives only 68 steps, so total delta is tiny. | Too weak (5× vs v1's 30×). Short episodes prevent any accumulation. | **Restore to 15×** — between v1 and v2, strong enough to guide |
| `position_quality` (1.5×) | ~8.86, slowly rises 7.67→9.01. Dense but diffuse — diluted over 68 crash steps. | Coefficient too low, and the product form (pos×vel×ang) collapses when any factor is bad | **Replace** with simpler `altitude_reward` that gives a clear vertical gradient |
| `contact_reward` | ~4.06, only 3% active (2 steps/ep). Agent briefly touches ground then dies. | Contact is too rare to shape behavior; agent never experiences sustained landing | **Keep but strengthen** and add partial-contact discovery pathway |
| `hard_landing` | ~-3.59, 1.5% active. Penalizes the symptom (fast contact) not the cause. | Reactive: fires only after the agent is already crashing. Cannot prevent the crash. | **Replace** with proactive `ground_speed_penalty` that fires *before* contact |
| `attitude_penalty` | ~-9.72, always active. v1 was -1.35 — v2 is **7× worse** because coefficient was tripled. | The 3× strengthening was counterproductive: it just adds constant negative drag | **Revert** to v1's mild coefficients |
| `speed_penalty` | ~-10.99, flat, always active. Largest negative component. | Universal drag on all motion — penalizes necessary descent as much as dangerous crashing | **Remove** flat penalty entirely |
| `fuel_cost` | ~-0.11, negligible (5% active) | Irrelevant at this stage | **Remove** |
| `step_cost` | ~-6.84 (=68.4×0.1), purely a function of episode length | Adds negative baseline without guiding behavior; irrelevant when agent can't survive | **Remove** |

### Root Cause Summary

The v2 reward creates a **net-negative per-step reward** (~-0.20/step). The agent's optimal policy under this reward is to terminate as quickly as possible. With no positive gradient to follow, learning is impossible — all checkpoint values are flat because the agent never improves.

v1 succeeded because it had a **net-positive signal**: `proximity_progress` (30× delta) dominated at 83% magnitude, creating a clear gradient toward the target. Its only flaw was no incentive to finish landing (hence hovering).

---

## Strategy for v3

**Return to v1's structure** (few components, net-positive, strong proximity signal) and add exactly what v1 was missing:

1. **`proximity_progress` (15×)**: Restore the strong delta-distance signal that worked in v1. Weight vertical 2× because descending is the critical action.

2. **`altitude_reward`**: NEW — dense per-step reward `0.1/(1+3y)`. Creates continuous incentive to descend. At y=0: 0.1/step; at y=1.5: 0.018/step. Smooth gradient.

3. **`soft_landing` (15×)**: 3× stronger than v1 (was 5×). Gives ~15/step for perfect two-leg landing, creating a strong "landing is better than hovering" signal.

4. **`partial_contact` (3×)**: NEW — any leg contact gets partial credit. Creates a discovery pathway: hover → touch one leg → touch both legs → perfect landing.

5. **`attitude_penalty`**: Revert to v1 coefficients (-0.5×angle², -0.2×ang_vel²). Mild, doesn't dominate.

6. **`ground_speed_penalty`**: NEW — proactive penalty when y<0.5 and speed>0.5. Fires *before* the crash, teaching the agent to slow down near ground.

**Removed**: `speed_penalty` (flat drag), `step_cost` (negative baseline), `fuel_cost` (irrelevant), `hard_landing` (reactive), old `position_quality` (replaced by altitude).

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3: back to v1's proven skeleton + descent pressure + stronger landing.

    Core changes from v2:
    - Remove: flat speed_penalty, step_cost, fuel_cost, hard_landing, position_quality
    - Restore: strong proximity_progress (15x, between v1's 30x and v2's 5x)
    - Add: altitude_reward for continuous descent incentive
    - Add: partial_contact for easier landing discovery
    - Strengthen: soft_landing 3x vs v1 (15.0 vs 5.0)
    - Add: proactive ground_speed_penalty (fires before crash, not after)
    - Revert: attitude_penalty to v1's mild coefficients
    """
    # ── state extraction ──
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_x, old_y = obs[0], obs[1]

    # ── 1. proximity_progress: weighted delta-distance ──
    # Vertical weighted 2x — descending is the critical action.
    # 15x coefficient is between v1 (30x) and v2 (5x).
    old_wdist = (old_x ** 2 + 2.0 * old_y ** 2) ** 0.5
    new_wdist = (x ** 2 + 2.0 * y ** 2) ** 0.5
    proximity_progress = 15.0 * (old_wdist - new_wdist)

    # ── 2. altitude_reward: dense incentive to descend ──
    # Max 0.1 at y=0, decays smoothly with height.
    # Gives the agent a reason to commit to going down instead of hovering.
    altitude_reward = 0.1 / (1.0 + 3.0 * y)

    # ── 3. soft_landing: contact-based rewards ──
    speed = (vx ** 2 + vy ** 2) ** 0.5
    both_contact = left_contact * right_contact
    any_contact = max(left_contact, right_contact)

    # Quality gates: landing is only rewarded when gentle and upright
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 3.0 * abs(angle))

    # Full bonus for perfect two-leg landing (3x stronger than v1)
    soft_landing = 15.0 * both_contact * speed_factor * angle_factor
    # Partial credit for any leg contact — easier to discover
    partial_contact = 3.0 * any_contact * speed_factor * angle_factor

    # ── 4. attitude_penalty: mild, same as v1 ──
    attitude_penalty = -0.5 * (angle ** 2) - 0.2 * (ang_vel ** 2)

    # ── 5. ground_speed_penalty: proactive crash prevention ──
    # Only activates when near ground AND moving fast — warns before impact.
    # Scales linearly with proximity to ground: stronger as y→0.
    ground_speed_penalty = 0.0
    if y < 0.5 and speed > 0.5:
        ground_speed_penalty = -3.0 * speed * (0.5 - y) / 0.5

    # ── total ──
    total_reward = (
        proximity_progress
        + altitude_reward
        + soft_landing
        + partial_contact
        + attitude_penalty
        + ground_speed_penalty
    )

    components = {
        "proximity_progress": proximity_progress,
        "altitude_reward": altitude_reward,
        "soft_landing": soft_landing,
        "partial_contact": partial_contact,
        "attitude_penalty": attitude_penalty,
        "ground_speed_penalty": ground_speed_penalty,
    }
    return float(total_reward), components
```