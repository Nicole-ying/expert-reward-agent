1. `evidence`：safe_contact_bonus的episode_sum_mean=521.69占magnitude_share 95.6%，但active_rate仅8.3%，内部奖励总和约503而外部score仅114.5，说明transition-based contact bonus被bouncing反复触发转化为内部收益，外部评估不认可该行为；progress_reward仅2.67（0.5%），被完全淹没；上一轮iter2与iter1相同骨架但得分从139.53降至114.50，len从1000降至907，表明该结构已不稳定。

2. `behavior_diagnosis`：agent在着陆垫附近反复弹跳，每次双腿重新接触即触发contact_transition（约75次/episode），以此刷取高额内部奖励，而非完成稳定着陆；16/20 episode被截断说明agent存活到时间上限但未真正settle，4/20提前终止可能是弹跳失控导致crash。

3. `signal_completeness`：progress_reward提供到达中心的过程引导（active_rate 99.7%，职责存在且可达），但尺度远小于contact bonus；缺失的是"稳定着陆"而非"接触事件"的反馈——当前transition bonus在弹跳→再接触循环中被反复领取，与真正的稳定着陆目标错位。

4. `selected_level`：Level 2，触发条件为`proxy_to_completion_alignment`证据明确——内部奖励高而外部score低，transition事件的active_rate 8.3%远超单次着陆应有的~0.1%（1次/1000步），弹跳导致proxy与外部任务错位；仅降系数（Level 1）不足以消除弹跳激励结构。

5. `selected_intervention`：将safe_contact_bonus从transition-based（`(1-contact_before)*contact_now`）改为**持续性接触质量奖励**：`contact_now * dist_factor * vel_factor * angle_factor`，用next_obs的速度和姿态因子门控，使得弹跳时高速度压低奖励、只有真正低速稳定着陆才能获得有意义的正向反馈。

6. `falsifiable_hypothesis`：弹跳伴随着高速度（vy符号交替、|v|较大），vel_factor会将其contact奖励压缩至接近0，agent无法继续通过弹跳刷分；同时持续接触在低速+竖直+中心时给予稳定正向激励，策略应转向减速→稳定接触→settle的路径，外部score应上升。

7. `expected_next_round`：safe_contact_bonus的episode_sum_mean和magnitude_share应大幅下降（预计从521→10~50量级），active_rate应上升（从8.3%→成为持续状态的高占比），progress_reward的signed_share应显著提升，外部score应改善，terminated（成功settle）比例应上升。

8. `main_risk`：vel_factor和angle_factor的bounded函数可能使contact奖励整体过小（塌缩），导致着陆激励不足、agent仅在中心徘徊而不尝试着陆；若下一轮active_rate接近0且terminated比例不升，需调整门控阈值或系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward ------------------
    w_progress = 2.0
    progress = dist - next_dist
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated) ------------------
    gate = 1.0 / (1.0 + dist)
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe contact reward (sustained quality-gated) ------------------
    w_contact = 1.0
    contact_now = next_left * next_right                              # 1.0 only if both legs touch
    dist_factor = 1.0 / (1.0 + 10.0 * next_dist)                      # near center
    vel_factor = 1.0 / (1.0 + 5.0 * (next_vx ** 2 + next_vy ** 2))   # low velocity gating
    angle_factor = 1.0 / (1.0 + 5.0 * (abs(next_angle) + abs(next_angvel)))  # upright + still
    safe_contact_reward = w_contact * contact_now * dist_factor * vel_factor * angle_factor

    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_reward

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_reward": safe_contact_reward
    }

    return float(total_reward), components
```