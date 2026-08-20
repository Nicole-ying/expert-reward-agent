**1. evidence** — score=240已超target(200)，landing_shaping_reward占据98% magnitude share且episode_sum_mean≈60，progress_reward和stability_penalty几乎为零贡献；score_range=[-45,303]说明多数episode成功但仍有少数崩溃；20/20 terminated（非截断）且len=231相比上轮613大幅缩短，表明势能塑形有效加速了着陆。

**2. behavior_diagnosis** — 策略学会了快速接近目标并稳定着陆（大多数episode得分接近上限~300），但少数episode出现严重负分（-45），可能是困难初始条件下着陆失败或crash；潜在问题是势能塑形只奖励接近+低速+正姿态，不要求真实接触，agent可能在某些episode中未完成实际触地就触发了settled终止。

**3. signal_completeness** — 过程引导（potential shaping）和稳定性约束（stability_penalty）完备且可达，但缺少落地接触确认信号。obs[6]/obs[7]（支撑脚接触标志）完全未被使用，这是唯一声明但未利用的任务完成相关信号。

**4. selected_level** — **Level 2**（proxy_to_completion_alignment）：当前proxy（proximity×speed×angle的势能差）不要求真实接触，现有证据（偶尔负分episode + contact信号未使用）支持添加接触对齐组件来区分「靠近目标悬停」与「实际着陆」。

**5. selected_intervention** — 新增`landing_contact_reward`组件：一次性转移奖励，仅在双脚从无接触到有接触的转移帧触发，且同步门控要求近目标+低速+姿态稳定。系数5.0，不改动任何现有组件。

**6. falsifiable_hypothesis** — 接触转移奖励为「真实着陆」提供唯一正反馈，应减少agent在目标附近悬停后因settled终止而不触地的情况，score_range下限应收窄（负分episode减少或消失），同时不破坏已有成功episode。

**7. expected_next_round** — landing_contact_reward的active_rate应很低（每episode最多触发一次），episode_sum_mean约3-5（取决于成功着陆比例），landing_shaping_reward应保持主导地位，总score保持或略升，score_range下限改善。

**8. main_risk** — 若门控阈值过松（尤其是dist和speed阈值），agent可能在crash着陆时也触发bonus，形成reward hacking；当前选用dist<0.5、合速度<1.0、|angle|<0.5的三重门控应能有效防止。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5: proxy_to_completion_alignment — add landing_contact_reward.
    One-time transition bonus when both feet first make contact near target
    with low speed and stable angle. Existing v4 components unchanged.
    """
    # -- Distance to target
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (unchanged)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty (unchanged)
    w_vel    = 0.001
    w_angle  = 0.001
    w_angvel = 0.0001

    stability_penalty = (
        -w_vel    * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle  * abs(next_obs[4])
        -w_angvel * abs(next_obs[5])
    )

    # -- 3. Landing potential shaping (unchanged from v4)
    proximity_curr    = 1.0 / (1.0 + 5.0 * dist_current)
    speed_curr        = abs(obs[2]) + abs(obs[3])
    speed_factor_curr = 2.0 / (2.0 + speed_curr)
    angle_factor_curr = 1.0 / (1.0 + abs(obs[4]))
    potential_current = proximity_curr * speed_factor_curr * angle_factor_curr

    proximity_next    = 1.0 / (1.0 + 5.0 * dist_next)
    speed_next        = abs(next_obs[2]) + abs(next_obs[3])
    speed_factor_next = 2.0 / (2.0 + speed_next)
    angle_factor_next = 1.0 / (1.0 + abs(next_obs[4]))
    potential_next = proximity_next * speed_factor_next * angle_factor_next

    w_potential = 80.0
    landing_shaping_reward = w_potential * (potential_next - potential_current)

    # -- 4. Landing contact reward (NEW: transition-based, multi-gated)
    obs_had_contact = (obs[6] > 0.5) and (obs[7] > 0.5)
    next_has_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    made_new_contact = float((not obs_had_contact) and next_has_contact)

    near_target = dist_next < 0.5
    low_speed   = (abs(next_obs[2]) + abs(next_obs[3])) < 1.0
    stable_angle = abs(next_obs[4]) < 0.5

    w_contact = 5.0
    landing_contact_reward = w_contact * made_new_contact * float(
        near_target and low_speed and stable_angle
    )

    # -- Total reward
    total_reward = progress_reward + stability_penalty + landing_shaping_reward + landing_contact_reward

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "landing_contact_reward": landing_contact_reward
    }

    return float(total_reward), components
```