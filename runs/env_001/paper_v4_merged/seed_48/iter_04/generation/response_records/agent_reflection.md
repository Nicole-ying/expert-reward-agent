# Response Record

1. `evidence`：当前得分 215.22，best 243.73，len 557.25；soft_landing_progress episode_sum_mean 497.16 占 signed_share 96.2%，主导奖励；distance_delta 仅 2.7%，engine_penalty 1.0%；无 dead component；20/20 terminated，无 truncated，表明 agent 能存活但 episode 较长。
2. `behavior_diagnosis`：agent 学会了在接近目标后保持在低速、低角度、双腿接触的状态，持续累积 soft_landing_progress 状态奖励，拖延完成稳定着陆，形成奖励 hacking，导致 episode 变长、得分较 best 下降。
3. `signal_completeness`：距离变化、速度、角度、接触等信号均已使用，职责完备，但 soft_landing_progress 的绝对状态奖励形式引入了“占据好状态即持续获奖”的漏洞。
4. `selected_level`：Level 2 结构变换。触发证据模式：占据好状态即持续获奖 → state→improvement。
5. `selected_intervention`：将 `soft_landing_progress` 从当前状态绝对值 `near_goal * low_speed * low_angle * contact_bonus` 改为基于改善的 `max(0, quality_new - quality_old)`，其中 quality 定义不变，权重 `w_soft = 2.0` 保留。
6. `falsifiable_hypothesis`：改善量使 agent 无法通过停滞在好状态获得持续奖励，必须进一步改善或尽快完成着陆以停止 episode，从而缩短 episode length，使 soft_landing_progress 份额下降、distance_delta 作用相对增强，预期 score 回升。
7. `expected_next_round`：episode_length 应显着下降（<400），soft_landing_progress 的 episode_sum_mean 骤降（<200），signed_share 降至 50% 以下，distance_delta signed_share 上升至 10%+；score 至少回升至 230+。
8. `main_risk`：改善奖励在停滞时为零，可能使 early training 信号过于稀疏，导致探索不足、收敛变慢或得分暂时下降；quality 恶化未被惩罚可能诱发振荡。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs / next_obs : [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]

    # ---- quality based on current obs ----
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    near_goal_old = 1.0 / (1.0 + 5.0 * dist_old)
    speed_sq_old = obs[2]**2 + obs[3]**2
    low_speed_old = 1.0 / (1.0 + 10.0 * speed_sq_old)
    abs_angle_old = abs(obs[4])
    low_angle_old = 1.0 / (1.0 + 20.0 * abs_angle_old)
    contact_bonus_old = 1.0 + 2.0 * (obs[6] * obs[7])
    quality_old = near_goal_old * low_speed_old * low_angle_old * contact_bonus_old

    # ---- quality based on next_obs ----
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    near_goal_new = 1.0 / (1.0 + 5.0 * dist_new)
    speed_sq_new = next_obs[2]**2 + next_obs[3]**2
    low_speed_new = 1.0 / (1.0 + 10.0 * speed_sq_new)
    abs_angle_new = abs(next_obs[4])
    low_angle_new = 1.0 / (1.0 + 20.0 * abs_angle_new)
    contact_bonus_new = 1.0 + 2.0 * (next_obs[6] * next_obs[7])
    quality_new = near_goal_new * low_speed_new * low_angle_new * contact_bonus_new

    # improvement-only soft landing progress (prevents reward farming on a good state)
    soft_progress = max(0.0, quality_new - quality_old)

    # distance improvement
    delta_distance = dist_old - dist_new

    # engine usage penalty
    engine_penalty = 1.0 if action != 0 else 0.0

    w_dist = 10.0
    w_soft = 2.0
    w_engine = 0.01

    total = (w_dist * delta_distance +
             w_soft * soft_progress -
             w_engine * engine_penalty)

    components = {
        'distance_delta': w_dist * delta_distance,
        'soft_landing_progress': w_soft * soft_progress,
        'engine_penalty': -w_engine * engine_penalty,
    }
    return float(total), components
```
