## 诊断报告

1. **evidence**：score=200.99 已超过 target=200，terminated=13/20 (65%)，truncated=7/20 (35%)，landing_quality 的 episode_sum_mean=316.34 占 signed_share 97.8%，远超其他组件总和，表明着陆后 agent 在平台上持续累积大量状态奖励。

2. **behavior_diagnosis**：agent 已学会软着陆（65% 成功率），但 episode 平均长达 626 步，landing_quality 在着陆后每步持续给予 ≈1.0 奖励（contact=1 且 vy≈0、angle≈0），导致 agent 缺乏快速触发 settled 终止的动机——"占据好状态即可持续获奖"。

3. **signal_completeness**：四类必要职责（progress、orientation_stability、soft_landing、crash_prevention）均有对应组件，但 landing_quality 作为持续状态奖励而非着陆事件奖励，使得奖励与"完成着陆"这一事件脱节。着陆后持续获奖掩盖了其他信号在训练中的相对作用。

4. **selected_level**：Level 2 — `persistent_to_transition_event`。landing_quality 当前是持续状态值（每步 contact×quality），证据符合"占据好状态即可持续获奖"模式；单纯降系数（Level 1）不会改变其持续累积的本质，着陆后仍会因步数多而主导训练信号。

5. **selected_intervention**：将 `landing_quality` 拆分为两个子组件——① `landing_event`：仅在接触从无到有的过渡步给予一次性奖励（contact_transition × quality）；② `landing_stability`：接触持续期间给予极小的维持奖励。其他三组件（progress、orientation_penalty、approach_velocity_penalty）保持不变。

6. **falsifiable_hypothesis**：将主要奖励集中到着陆瞬间应使 landing 相关组件的 episode_sum_mean 从 316 降至更合理范围（预计 20–60），magnitude_share 从 97.8% 显著下降，同时 terminated 比例至少保持 13/20；若 agent 在着陆后不再有持续高奖励可赚，可能更快触发 settled，episode_length 应缩短。

7. **expected_next_round**：landing_event 的 active_rate 应 ≈5–10%（仅过渡步触发），landing_stability 的 episode_sum_mean 应 <10；总 score 可能下降但仍应在 target 附近；terminated 比例保持或改善；episode_length 应下降。

8. **main_risk**：着陆事件奖励仅在一次过渡步发放，若系数设置过低则软着陆引导不足，agent 可能退化到只依赖 progress 冲向平台而导致硬着陆。需确保 landing_event 的一次性奖励足够显著（约 20–30）以维持着陆动机。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights
    w_progress = 1.0
    w_angle_penalty = 0.2
    w_landing_event = 25.0       # one-time bonus on first contact transition
    w_landing_stability = 0.05   # tiny sustaining reward while on ground
    w_approach_vel = 0.3
    a_v = 10.0
    b_angle = 10.0

    # distances
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Progress
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing: split into event (transition) and stability (sustaining)
    contact_prev = max(obs[6], obs[7])
    contact_next = max(next_obs[6], next_obs[7])
    contact_transition = float(contact_next > 0.5 and contact_prev <= 0.5)

    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    quality = vertical_speed_factor * angle_factor

    landing_event = w_landing_event * contact_transition * quality
    landing_stability = w_landing_stability * contact_next * quality

    # 4. Approach-phase velocity penalty
    proximity = 1.0 / (1.0 + next_dist)
    velocity_sq = next_obs[2] ** 2 + next_obs[3] ** 2
    approach_velocity_penalty = -w_approach_vel * proximity * velocity_sq

    total_reward = (
        progress
        + orientation_penalty
        + landing_event
        + landing_stability
        + approach_velocity_penalty
    )

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_event": landing_event,
        "landing_stability": landing_stability,
        "approach_velocity_penalty": approach_velocity_penalty
    }

    return float(total_reward), components
```