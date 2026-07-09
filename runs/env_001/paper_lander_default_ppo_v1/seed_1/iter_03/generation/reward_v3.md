`evidence`: score=-106.27, all 20 episodes terminate at avg len=68.55; soft_landing_bonus active_rate=0.4% (≈0.27 steps/ep), essentially never fires; progress_reward=2.28/ep at 92% active rate but dwarfed by environment contribution of ≈-107.7/ep (gap between total score and component sum); agent crashes before achieving landing configuration.

`behavior_diagnosis`: agent makes some progress toward pad (positive progress_reward) but crashes quickly every episode. The landing bonus conditions are too strict to provide any learnable gradient for the final approach phase, leaving the agent with no signal to slow down or stabilize near the pad.

`signal_completeness`: missing a dense approach/landing signal bridging "getting close" and "touching down." The sparse binary bonus at 0.4% active rate fails the `sparse_completion_proxy` heuristic and provides no usable gradient.

`selected_level`: Level 2 — `sparse_to_dense` transformation of soft_landing_bonus, triggered by active_rate < 1% with all episodes crashing before landing.

`selected_intervention`: replace the hard 6-condition binary `soft_landing_bonus` with a continuous product of bounded proximity, speed, and angle scores. Each factor uses `max(0, 1 - value/threshold)` so partial improvement yields partial reward, creating a gradient the agent can follow from the progress-guided approach into a controlled landing.

`falsifiable_hypothesis`: the continuous landing approach reward should raise soft_landing_bonus active_rate substantially (target > 10%), increase its episode_sum_mean, and — by providing gradient toward the landing configuration — reduce crash rate (some episodes should begin surviving longer or achieving landing).

`expected_next_round`: soft_landing_bonus active_rate should rise well above 0.4%; episode length should increase as agent learns to decelerate near pad; score should improve (less negative) as fewer episodes end in early crashes.

`main_risk`: product form may still collapse to near-zero if any one condition is far from target (especially angle or speed early in approach), potentially requiring per-factor coefficient tuning or switching to a sum-of-bounded-terms if active_rate remains low.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：distance-based progress（只奖励接近，不惩罚远离） ----
    prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_distance = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_distance - next_distance
    progress_reward = max(progress, 0.0) * 2.0

    # ---- 稳定/安全约束：轻量级惩罚 ----
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    stability_penalty_value = (
        0.01 * abs(vx)
        + 0.01 * abs(vy)
        + 0.01 * abs(angle)
        + 0.005 * abs(angular_vel)
    )
    stability_penalty = -stability_penalty_value

    # ---- 连续着陆逼近信号：替代稀疏二值 bonus ----
    distance_to_pad = next_distance
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle_abs = abs(angle)

    proximity_score = max(0.0, 1.0 - distance_to_pad / 1.0)
    speed_score = max(0.0, 1.0 - speed / 0.5)
    angle_score = max(0.0, 1.0 - angle_abs / 0.3)

    soft_landing_bonus = proximity_score * speed_score * angle_score * 2.0

    # ---- 总奖励 ----
    total_reward = progress_reward + soft_landing_bonus + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```