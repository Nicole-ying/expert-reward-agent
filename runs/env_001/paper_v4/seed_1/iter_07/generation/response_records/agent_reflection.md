# Response Record

`evidence`: current score=-16.87 vs best=120.02, all 20 episodes truncated at 1000 steps (never land), shaped_progress sums to only 4.61 over full episodes, landing_quality active_rate=0.0% (never triggered), angular_vel_penalty active but tiny at -0.14; previous switch from iter5's landing_bonus (transition event, score=120) to iter6's landing_quality (persistent state reward) caused complete collapse.

`behavior_diagnosis`: agent wanders for the full 1000 steps without ever touching the landing pad; shaped_progress provides weak directional guidance (~0.0046/step) but no signal exists for the final approach conditions (leg contact, low speed, upright); the persistent landing_quality is structurally unreachable because its 4-factor product (`is_landed * near_target * low_speed * upright`) requires simultaneous satisfaction of all conditions before any feedback appears.

`signal_completeness`: progress signal present but too weak alone to close the final gap; stability penalty functional; **landing completion signal structurally unreachable** — a persistent state reward that only activates post-landing cannot guide an agent that never discovers landing.

`selected_level`: Level 2 — `persistent_to_transition_event`. Evidence: iter5's transition-based landing_bonus achieved score=120 with successful landings; iter6's persistent landing_quality active_rate=0.0% proves the state-reward structure is unreachable from exploration. The mathematical form must change from "reward every step in landed state" to "reward the act of making contact."

`selected_intervention`: replace `landing_quality` (persistent state reward) with `landing_bonus` (soft transition event). New structure: `15.0 * max(0, contact_now - contact_before) * near_target * upright * low_speed`. The `max(0, Δcontact)` creates a soft transition that fires on any increase in leg contact, providing gradient throughout the descent. Quality multipliers scale the bonus size but partial contact with imperfect posture still yields non-zero reward.

`falsifiable_hypothesis`: making the landing signal a transition-based reward (firing on contact increase rather than persistent occupation) should make it discoverable during exploration — the agent will receive positive feedback for any leg contact increase, however partial, creating a gradient toward full landing that shaped_progress alone cannot provide. The `max(0, Δcontact)` term also naturally prevents farming since sustained contact yields zero delta.

`expected_next_round`: landing_bonus active_rate should rise from 0% to >0%; episode_length should drop below 1000 as agents begin to land; score should improve significantly toward or beyond the iter5 baseline of 120.

`main_risk`: the soft delta-contact bonus could reward incidental leg bumps during mid-air tumbling rather than intentional landings, creating a proxy that diverges from the task goal. The quality multipliers (near_target, upright, low_speed) mitigate this by heavily discounting off-target contact.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：距离改善 × 姿态门控（保持不变）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next

    angle_abs = abs(next_obs[4])
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚（保持不变）
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆奖励：软转移事件（替代持续状态奖励）
    # 检测腿部接触的增量：接触增加即给分，完全接触后增量归零自然防刷
    contact_now = next_obs[6] * next_obs[7]
    contact_before = obs[6] * obs[7]
    new_contact = contact_now - contact_before
    if new_contact < 0.0:
        new_contact = 0.0

    # 着陆质量因子：靠近目标、直立、低速
    near_target = 1.0 / (1.0 + 3.0 * abs(next_obs[0]))
    upright = 1.0 / (1.0 + 3.0 * angle_abs)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + speed)

    landing_bonus = 15.0 * new_contact * near_target * upright * low_speed

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return (float(total_reward), components)
```
