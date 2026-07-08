# Response Record

## 1. evidence
当前迭代（iter 7）得分 -108.69，episode 长度仅 68.45，所有 episode 以 terminated 结束，distance_reward 占 magnitude 的 72.9% 且 landing_quality 仅 0.241/步；相比之下历史最佳 iter 6 使用相同骨架（landing_quality=0.2 系数）得分 -33.72，episode 长度 254.80，landing_quality 达 0.495/步。

## 2. behavior_diagnosis
agent 学会快速冲向目标附近（平均距离约 0.97），但因速度过快或姿态失控而坠毁，导致 episode 提前终止；landing_quality 信号虽然在接近目标时提供柔和着陆引导，但其相对强度不足以在大多数训练运行中阻止冲撞行为，iter 6 的优异表现属于训练不稳定中的幸运发现。

## 3. signal_completeness
职责基本完备：distance_reward 提供向目标前进的梯度，stability_penalty 约束速度和姿态，landing_quality 在接近目标时奖励缓慢、水平、双腿接触的着陆状态。但线性无界的 distance_reward 在远距离时过于主导（-5 至 -10/步），使得 landing_quality（最大 1.0/步）无法在早期学习阶段形成有效对抗信号，导致 agent 倾向快速冲撞而非学习减速。

## 4. selected_level
Level 2：同一骨架家族已在 iter 3/4/6/7 迭代 4 轮，3/4 次得分为 -108 左右，仅 iter 6 偶然成功，说明当前数学结构存在系统性训练不稳定。需要对 distance_reward 做 `unbounded_to_bounded` 结构变换，将线性无界惩罚改为有界饱和形式，从结构上消除"冲撞是局部最优"的问题。

## 5. selected_intervention
唯一修改 distance_reward 组件：从 `-1.0 * distance_to_target`（线性无界）变为 `-2.0 * (1.0 - e^(-distance_to_target / 2.0))`（指数饱和有界）。保持 landing_quality=0.2 系数和 stability_penalty 不变。

## 6. falsifiable_hypothesis
将有界饱和距离奖励与 landing_quality 配合，远距离时距离惩罚被限制在 -2.0 以内，不再提供"越快接近越好"的激励；靠近目标时梯度与原线性形式相同（-1.0），保持精确着陆引导；landing_quality 的相对权重因此提升，agent 在接近目标区域时被迫减速以获取正向奖励，从而减少冲撞终止。

## 7. expected_next_round
episode 长度应显著增加（>150），terminated 比例可能仍高但应出现更长的存活轨迹；landing_quality 的 episode_sum_mean 和 per-step 均值应上升；distance_reward 的 magnitude_share 应从 72.9% 下降；score 应刷新 best 或至少接近 -33.72。

## 8. main_risk
指数饱和在远距离（d>6）时梯度非常弱（<0.05），如果初始生成位置过远，agent 可能在早期探索中难以发现向目标移动的方向；若出现此情况，下轮需调整饱和速率或结合稀疏方向引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Distance reward: bounded exponential saturation to prevent rushing
    #    Replaces linear -distance which created a local optimum of fast crashing approach.
    #    -2*(1 - e^(-d/2)) asymptotes at -2.0, gradient at d=0 is -1.0 (same as original).
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -2.0 * (1.0 - 2.718281828 ** (-distance_to_target / 2.0))

    # 2. Light stability penalty (preserved from best, iter 6)
    stability_penalty = -(
        0.15 * abs(x_vel) +
        0.05 * abs(y_vel) +
        0.2 * abs(body_angle) +
        0.2 * abs(angular_vel)
    )

    # 3. Landing quality: sum-based joint satisfaction (preserved from best, iter 6)
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    sum_of_factors = prox_factor + speed_x_factor + speed_y_factor + angle_factor + contact_factor
    landing_quality = 0.2 * sum_of_factors

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```
