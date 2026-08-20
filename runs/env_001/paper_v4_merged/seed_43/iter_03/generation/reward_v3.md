evidence：所有episode在68步terminated（快速坠毁），score=-117，progress_shaping占90%但无法阻止坠落；danger_penalty active_rate=0%（未捕获任何失败状态）；action_cost与angle_hinge均微量；leg contact信号（obs[6],[7]）完全未被使用。
behavior_diagnosis：agent采取几乎不点火（action_cost active_rate 4.8%）的自由落体策略，通过progress_shaping在距离缩小时获得正向奖励，但迅速坠毁或出界，导致episode极短且总奖励为负。
signal_completeness：缺少明确的安全着陆吸引子，现有danger_penalty因阈值与真实失败模式不匹配而失效，未利用leg contact用于着陆成功引导。
selected_level：Level 2 — structural transform，因前轮迭代得分停滞且僵尸组件（danger_penalty active_rate=0%）未实现设计意图，需移除并替换为新职责信号。
selected_intervention：删除danger_penalty，新增landing_contact_reward组件，基于支撑脚接触和到目标距离的连续bounded factor，以提供着陆指向性奖励。
falsifiable_hypothesis：引入landing_contact_reward后，agent将学会使用引擎控制下降以实现支撑脚触地并靠近中心，从而减少坠毁终止，episode长度上升、score显著改善，landing_contact_reward的active_rate＞0且episode_sum_mean为正。
expected_next_round：landing_contact_reward active_rate > 5%，score至少提升20点（如＞-95），episode length可能出现增长；progress_shaping magnitude_share下降但保持正贡献。
main_risk：接触奖励可能导致agent优先追求单脚接触而忽略姿态稳定，引起侧翻或缓慢漂移但仍最终坠毁，需要下一轮观察是否需要加入姿态约束限制。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    nangle = float(next_obs[4])
    left_contact = float(next_obs[6])
    right_contact = float(next_obs[7])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Stability constraint: body angle hinge ----------
    angle_threshold = 0.3
    angle_hinge = -0.5 * max(0.0, abs(nangle) - angle_threshold)

    # ---------- 3. Efficiency bonus: action penalty ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. Landing contact bonus ----------
    contact_sum = left_contact + right_contact       # in {0, 1, 2}
    dist_to_target = dist_next
    # reward proximity when legs touch the pad, max when directly centered
    contact_factor = contact_sum / 2.0               # 0.0 to 1.0
    proximity = max(0.0, 1.0 - dist_to_target / 0.8) # 1.0 at perfect center, 0 beyond 0.8
    landing_contact_reward = 0.2 * contact_factor * proximity

    total_reward = progress_shaping + angle_hinge + action_cost + landing_contact_reward

    components = {
        "progress_shaping": progress_shaping,
        "angle_hinge": angle_hinge,
        "action_cost": action_cost,
        "landing_contact_reward": landing_contact_reward
    }

    return float(total_reward), components
```