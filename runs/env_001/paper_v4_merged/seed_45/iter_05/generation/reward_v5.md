1. `evidence`：上一轮 score=195.26（new_best），terminated=50%，completion_proxy 占据 signed_share 98.3%、active_rate 仅 64.5%；speed/angle/angvel 三种惩罚 active_rate 均 <5% 且数值接近 0，属于僵尸组件。
2. `behavior_diagnosis`：agent 已能在一半 episode 成功着陆（terminated），另一半超时未触发着陆终止，说明在最终稳定着陆阶段仍需更强的连续引导。
3. `signal_completeness`：completion_proxy 所依赖的 contact_factor 对双脚接触要求过严，导致在空中或单脚接触时乘积常归零，缺失从悬空到着陆的过渡奖励。
4. `selected_level`：Level 2 结构变换 —— 修改 completion_proxy 内部 contact_factor 的计算方式，从硬性双脚接触改为带底部的软接触，以提高 proxy 激活率。
5. `selected_intervention`：仅修改 contact_factor 公式为 `0.1 + 0.9 * (nleft + nright) / 2.0`，保证未接触时仍有 0.1 的最小因子，避免乘积塌缩。
6. `falsifiable_hypothesis`：proxy 的 active_rate 将上升至接近 100%，奖励密度增加，推动 agent 更快完成最终着陆，从而提高 terminated 率和 mean score，缩小方差。
7. `expected_next_round`：completion_proxy 的 active_rate > 90%，episode_sum_mean 上升，terminated 比例＞50%，整体 score > 200，且 score 方差下降。
8. `main_risk`：悬停步也获得了微小奖励，可能诱导 agent 盘旋而不触发终止，导致 truncated 比例升高；需下一轮关注 terminated 率是否下降。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # Distances to target center (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # 1. Progress towards center
    progress_delta = 5.0 * (dist - next_dist)

    # 2. Completion proxy (geometric mean of conditions)
    proximity = max(0.0, 1.0 - next_dist / 0.8)
    velocity_moderation = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.5)
    orientation_stability = max(0.0, 1.0 - abs(nangle) / 0.2)
    angvel_moderation = max(0.0, 1.0 - abs(nangvel) / 0.3)
    # Soft contact factor: always at least 0.1, reaching 1.0 when both feet touch
    contact_factor = 0.1 + 0.9 * (nleft + nright) / 2.0

    proxy_product = proximity * velocity_moderation * orientation_stability * angvel_moderation * contact_factor
    completion_proxy = 1.0 * (proxy_product ** 0.2) if proxy_product > 0 else 0.0

    # 3. Safety penalties (hinge, low thresholds)
    speed_threshold = 0.4
    vx_violation = max(0.0, abs(nvx) - speed_threshold)
    vy_violation = max(0.0, abs(nvy) - speed_threshold)
    speed_penalty = -0.1 * (vx_violation + vy_violation)

    angle_threshold = 0.2
    angle_violation = max(0.0, abs(nangle) - angle_threshold)
    angle_penalty = -0.2 * angle_violation

    angvel_threshold = 0.3
    angvel_violation = max(0.0, abs(nangvel) - angvel_threshold)
    angvel_penalty = -0.1 * angvel_violation

    total_reward = progress_delta + completion_proxy + speed_penalty + angle_penalty + angvel_penalty

    components = {
        'progress_delta': progress_delta,
        'completion_proxy': completion_proxy,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components
```