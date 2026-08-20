`evidence`：stability_penalty 占据79% magnitude_share，episode_sum_mean=-9.68 远大于 distance_progress 的 2.24（比值约4.3，远超0.5经验预警线）；soft_landing_proxy active_rate仅0.4%，几乎不触发；上一轮将distance_reward改为distance_progress，得分从-114微升至-109.88，未改stability_penalty。

`behavior_diagnosis`：agent在约68步内快速接近目标（distance_progress累计2.24，约为最大可能值3.0的75%）但无法减速完成着陆，最终全部以crash终止。稳定性惩罚过强，导致学习信号被惩罚主导但又不具备足够的阶段性引导来防止crash。

`signal_completeness`：progress信号和稳定性约束在数学形态上合理，但尺度严重失衡。软着陆proxy过于稀疏（0.4%），缺失有效的着陆引导。引擎推力代价缺失。

`selected_level`：Level 1 —— stability_penalty 的职责和数学形态未被证伪，问题在于其系数相对 progress 过强，|ratio_to_progress|≈4.3 >> 0.5 明确触发尺度修复条件。

`selected_intervention`：唯一目标组件为 stability_penalty，将四个子项的系数从 -0.1 统一降至 -0.008（约12.5倍缩小），使预期 |penalty/progress| 降至约0.35。

`falsifiable_hypothesis`：降低稳定性惩罚后，agent将不再被过度约束，能更自由地探索接近目标后的减速与着陆行为；progress信号的相对权重提升应使策略学习到更平衡的趋近-减速曲线，soft_landing_proxy的触发机会也可能因更多接近目标区域的探索而增加。

`expected_next_round`：episode_length可能因探索更自由而略微变化，distance_progress预期保持或略增，stability_penalty的magnitude_share应降至40%以下，soft_landing_proxy的active_rate可能微升，external score应有实质改善（不再被过度惩罚拖累）。

`main_risk`：降低稳定性惩罚后agent可能以更高速度冲向目标区域，若无足够的减速信号，crash可能提前而非延后；soft_landing_proxy仍过于稀疏，可能不足以提供着陆引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next observation variables
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress signal: reward getting closer to target, penalize moving away
    prev_dist = (prev_x**2 + prev_y**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    # 2. Stability constraint: light penalty on high speeds and large angles
    #    Coefficient reduced 12.5x from -0.1 to -0.008 to fix |penalty/progress| > 0.5 dominance
    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    # 3. Soft landing proxy: reward simultaneous near-target, low-speed,
    #    upright attitude and both legs contacting the pad.
    prox_dist_thresh = 0.3
    prox_vel_thresh = 0.2
    prox_angle_thresh = 0.1

    condition = (
        curr_dist < prox_dist_thresh and
        abs(vx) < prox_vel_thresh and
        abs(vy) < prox_vel_thresh and
        abs(angle) < prox_angle_thresh and
        left_contact > 0.5 and
        right_contact > 0.5
    )
    reward_landing = 1.0 if condition else 0.0

    total_reward = reward_dist + reward_stability + reward_landing

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "soft_landing_proxy": reward_landing
    }

    return float(total_reward), components
```