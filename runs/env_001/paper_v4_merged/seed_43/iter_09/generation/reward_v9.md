1. `evidence`：所有 episode 在 ~84 步终止，score=-124，全部为早期坠毁；soft_landing_bonus 从未触发，speed_penalty 活跃率 0.8% 却贡献 -32% 负份额，angle_penalty 活跃率 0.2%；历史最优骨架在 iter4 (score -87, len 144) 使用了 contact_reward + speed_gate + progress_shaping，后续迭代丢弃这些组件后得分恶化。
2. `behavior_diagnosis`：agent 在接近目标时无法减速和稳定姿态，快速坠毁，从未实现双脚接触着陆；现有奖励缺乏有效的安全约束和完成引导。
3. `signal_completeness`：缺失连续的接触反馈和任务完成近似信号；姿态和速度约束仅靠极低活跃度的惩罚组件，没有健康门控机制；成功信号完全不可达。
4. `selected_level`：Level 3 重建 — 同一骨架族连续 4 轮未刷新 best，且当前得分仅为目标的 -62%，历史最佳也远未过半。
5. `selected_intervention`：全新骨架，以 progress_delta * angle_gate * contact_factor 为主进展，叠加连续 success_proxy（几何平均）和近距离速度惩罚，外加轻量动作成本；移除所有僵尸组件。
6. `falsifiable_hypothesis`：通过衰减不安全状态下的进展奖励和提供密集的完成倾向信号，agent 应学会在接近目标时减速、调整姿态并尝试双脚接触，从而延长 episode 长度并提升 score。
7. `expected_next_round`：score 应上升至 -80 以上，episode_length 突破 100，success_bonus 组件 active_rate > 50%，speed_penalty 活跃率上升且幅度合理。
8. `main_risk`：angle_gate 将 progress 衰减至 0.3 倍可能过度抑制必要的旋转探索，延长学习时间；success_bonus 连续乘积可能诱导 agent 在未完全稳定的状态下过早起接触，导致坠毁。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nl_contact, nr_contact = next_obs

    # distances to target (0,0)
    dist = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # 1. progress delta: positive when approaching target
    delta_dist = dist - dist_next
    progress = 1.0 * delta_dist

    # 2. angle gate: linearly decay progress when body angle exceeds safe range
    safe_angle = 0.5  # radians
    gate_angle = max(0.3, 1.0 - abs(nangle) / safe_angle)

    # 3. contact factor: encourage both legs on ground
    if nl_contact == 1 and nr_contact == 1:
        contact_factor = 1.0
    elif nl_contact == 1 or nr_contact == 1:
        contact_factor = 0.7
    else:
        contact_factor = 0.4

    # shaped progress: main learning signal with safety and contact modulation
    shaped_progress = progress * gate_angle * contact_factor

    # 4. speed penalty near ground to promote gentle landing
    close_threshold = 0.5
    speed_penalty = 0.0
    if dist_next < close_threshold:
        speed_norm = abs(nvx) + abs(nvy)
        speed_penalty = -0.1 * speed_norm

    # 5. continuous success proxy using geometric mean of proximity, stability and contact
    proximity = 1.0 / (1.0 + 10.0 * dist_next)
    speed_norm_eucl = (nvx**2 + nvy**2) ** 0.5
    stability = 1.0 / (1.0 + 3.0 * speed_norm_eucl + 3.0 * abs(nangle))
    contact_quality = (nl_contact + nr_contact) / 2.0  # in [0,1]
    # geometric mean to avoid product collapse; add tiny epsilon for zero case
    product = proximity * stability * contact_quality
    eps = 1e-6
    success_factor = (max(product, eps)) ** (1.0 / 3.0)
    success_bonus = 5.0 * success_factor

    # 6. action cost: small penalty for any engine use
    action_cost = -0.01 if action != 0 else 0.0

    total_reward = shaped_progress + success_bonus + speed_penalty + action_cost

    components = {
        "progress": progress,
        "gate_angle": gate_angle,
        "contact_factor": contact_factor,
        "shaped_progress": shaped_progress,
        "speed_penalty": speed_penalty,
        "success_bonus": success_bonus,
        "action_cost": action_cost
    }
    return float(total_reward), components
```