1. `evidence`：历史最佳 iter3 得分 91.82，其 contact_bonus 使用 softly gated 的近地鼓励+接触奖励；此后连续三轮修改该组件均导致得分下降（iter4 13.47, iter5 -39.29），且当前 contact_bonus 占比 98% 但外部分数为负，表明该组件已驱动劣质着陆行为。
2. `behavior_diagnosis`：agent 学会在着陆垫附近以不完全稳定的状态频繁触发双腿接触以索取巨额 contact_bonus，但速度、角度和角速度依然较大，导致外部任务评分极低（快速但剧烈触及，未实现软着陆）。
3. `signal_completeness`：当前骨架包含进度、接触奖励、燃料惩罚、姿态惩罚，信号职责完备；主要缺陷是 contact_bonus 的具体形式传递了错误激励，而非缺少必要职责。
4. `selected_level`：Level 2，因为需要在历史最佳骨架基础上改变 contact_bonus 的数学形态，去除鼓励下降速度‑0.5 的近地引导，并在接触时严格对齐稳定软着陆条件。
5. `selected_intervention`：唯一目标组件为 `contact_bonus`，修改为仅在双腿同时接触时依据速度、角度、角速度的严格稳定性门控发放高额奖励，移除原近地持续奖励。
6. `falsifiable_hypothesis`：去除近地引导可消除悬停或缓慢下落动机；落地奖励严格化会迫使 agent 在接触瞬间满足速度≈0、竖直姿态≈0，从而提升真正的软着陆成功率与外部得分。
7. `expected_next_round`：外部 score 应明显上升（目标 >50）；`len` 可能缩短；`terminated` 次数持平或增加；`contact_bonus` 的 episode_sum 及 active_rate 均下降，但 progress 等组件相对占比提升。
8. `main_risk`：如果稳定性门控过严，接触奖励可能引发稀疏梯度，agent 可能放弃着陆尝试而只依赖 progress 奖励，导致 landing 成功率下降或撞毁；需监控 contact_bonus 是否完全熄火。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- progress reward (kept from best) ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle
    progress_reward = 5.0 * progress_delta * gate_health

    # ---------- contact bonus: only reward stable soft landing ----------
    contact_bonus = 0.0
    if left_contact_next > 0.5 and right_contact_next > 0.5:
        # strict stability gating near zero velocity and zero angle
        stability = (2.718281828 ** (-5.0 * speed_sq)) * \
                    (2.718281828 ** (-10.0 * (angle_next**2))) * \
                    (2.718281828 ** (-5.0 * (abs(angvel_next)**2)))
        contact_bonus = 50.0 * stability

    # ---------- fuel penalty (kept unchanged) ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- near-ground orientation penalty (kept unchanged) ----------
    orientation_penalty = 0.0
    if y_next < 0.5:
        orientation_penalty = -0.5 * (angle_next**2) - 0.2 * (abs(angvel_next)**2)

    total_reward = progress_reward + contact_bonus + fuel_penalty + orientation_penalty

    components = {
        "progress": progress_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "orientation_penalty": orientation_penalty
    }
    return float(total_reward), components
```