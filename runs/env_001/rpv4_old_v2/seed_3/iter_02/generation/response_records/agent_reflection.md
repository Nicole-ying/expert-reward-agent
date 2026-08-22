# Response Record

1. **evidence**：当前 strategy 的 safe_contact_reward 占据 97.7% 的 magnitude share 且 per‑step 量级 0.324，而 progress、orientation、velocity 三个组件的 signed share 合计不足 2%，形成明显的“接触持续奖励”主导。
2. **behavior_diagnosis**：agent 学会了到达平台并保持双脚接触、低速度、小角度，从而持续获取 safe_contact_reward，但仅 1/20 的分集触发了终止（疑似 settled），其余全部被时间截断，说明 agent 缺少尽快完成稳定着陆的动力，倾向于“停在平台上赖着不走”。
3. **signal_completeness**：现有职责已覆盖接近目标、减速、姿态稳定与接触判断，缺少将“着陆完成”与“稳定终止”对齐的终端激励，但由于无显式成功标志，必须通过调整接触奖励的时间结构来间接激励更快稳定。
4. **selected_level**：Level 2 – 修改组件的数学形态，将 safe_contact_reward 从持续状态值奖励改为仅着陆瞬间的事件奖励，以消除赖在平台上获取无限累计回报的动机。
5. **selected_intervention**：唯一干预目标是 safe_contact_reward，将其公式改为检测双腿接触从无到有的瞬间事件，给予固定大额奖励并完全移除持续奖励。
6. **falsifiable_hypothesis**：移除持续奖励后，agent 将失去在平台上消极等待的动机，转而追求更早满足环境内部的稳定终止条件，从而使 terminated 比例上升，episode 长度显著缩短，同时 score 因更早完成而提高。
7. **expected_next_round**：terminated 比例应从 1/20 提升，平均 episode 长度将从 ~965 明显下降，safe_contact_reward 的 magnitude share 将不再是持续累积而是由事件主导，score 有望接近或超过 200。
8. **main_risk**：若着陆事件奖励不够大，agent 可能因为缺乏足够正向激励而不敢尝试着陆（尤其在前期惩罚累积下），导致 score 下降甚至策略不收敛。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (event_based landing bonus)
    #   Reward only the moment both legs first touch the pad.
    #   No continuing reward for staying on the pad.
    # ------------------------------------------------------------
    legs_touching_now = (obs[6] > 0.5) and (obs[7] > 0.5)
    legs_touching_next = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    if (not legs_touching_now) and legs_touching_next:
        safe_contact_reward = 20.0
    else:
        safe_contact_reward = 0.0

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return total_reward, components
```
