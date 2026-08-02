1. **evidence**：所有20个评估episode均提前终止（terminated=20/20，len≈68），总得分-113.7。组件中progress的episode_sum_mean=1.12且active_rate=100%，soft_landing的magnitude_share=39.6%但active_rate仅0.9%（偶尔触发的大值），angle_penalty从未激活，shaped奖励总和约1.8/回合，但环境返回的外部得分仍大幅为负，说明代理正追求无约束的progress，导致高速碰撞或出界。
2. **behavior_diagnosis**：代理学会了利用步间距离减少量（delta_dist）获取正向奖励，并形成了快速垂直下降的策略；这种策略无视着陆安全要求，导致在68步左右因高速撞击或出界而终止，外部得分极低，未见成功着陆。
3. **signal_completeness**：当前信号集合缺少对下降速度的直接约束，progress是开放式的线性奖励，在高速下降时值最大，反而鼓励了致命行为；soft_landing虽有速度因子但仅在接触后激活，无法在下降阶段提供引导；因此，信号职责虽看似完备，但优化过程因progress无界而扭曲。
4. **selected_level**：Level 2 — 结构变换，触发基础是“progress的数学形态为unbounded线性正奖励，且外部得分在shaped奖励持续为正的情况下仍为负”，属于“proxy 提高但外部分数不升”的证据模式，需要对主正向信号施加边界约束。
5. **selected_intervention**：唯一目标组件是`progress`。修改方式：在计算`delta_dist`后，乘入一个基于垂直速度的安全下降门控因子`gate`。当下降速度（-vy）超过`max_safe_vy=0.5`时，`gate`线性衰减至0，从而削弱高速下降时的progress奖励强度，其余组件保持不变。
6. **falsifiable_hypothesis**：通过抑制高速下降，代理将被迫学习在下降过程中减速或暂停，从而减少碰撞终止的概率，外部得分应出现明显改善（负分向0收敛或转正），episode可能因下降变慢而略延长，soft_landing的active_rate有望上升。
7. **expected_next_round**：外部score应至少升至-50以上；`progress`组件的episode_sum_mean可能从1.12下降至0.6~0.9，但`gate`的active_rate接近100%；`soft_landing`的active_rate可能从0.9%升至5%以上；terminated率可能仍为100%或略微下降，episode_length可能增加至80~90。
8. **main_risk**：`max_safe_vy=0.5`过于保守可能导致代理在空中悬停或下降极慢，造成episode长时间不终止且无法触及地面，若下一轮出现truncated/超时或proxy停滞，则需调高阈值或引入更精细的高度依赖门控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重和阈值
    w_progress = 1.0
    w_angle = 0.5
    w_angvel = 0.1
    w_soft_land = 2.0
    w_eff = 0.02

    angle_thresh = 0.3   # rad
    angvel_thresh = 1.0  # rad/s
    max_speed_land = 1.0 # 着陆容许最大合速度
    max_angle_land = 0.5 # 着陆容许最大倾角 rad
    max_safe_vy = 0.5    # 安全下降的垂直速度阈值（m/s）

    # 距离进展（步间距离减少量），加入安全下降门控
    old_dist = (obs[0]**2 + obs[1]**2)**0.5
    new_dist = (next_obs[0]**2 + next_obs[1]**2)**0.5
    delta_dist = old_dist - new_dist   # 正值表示向目标接近

    vy = next_obs[3]                   # 垂直速度
    downward_speed = -vy if vy < 0.0 else 0.0   # 向下速度
    if downward_speed > max_safe_vy:
        overshoot = downward_speed - max_safe_vy
        gate = max(0.0, 1.0 - overshoot / max_safe_vy)
    else:
        gate = 1.0
    progress = w_progress * delta_dist * gate

    # 姿态稳定性（hinge 惩罚）
    angle = next_obs[4]
    angvel = next_obs[5]
    angle_penalty = -w_angle * max(0.0, abs(angle) - angle_thresh)
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_thresh)

    # 软着陆奖励（仅在支撑腿接触时有效）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    any_contact = 1.0 if (left_contact + right_contact) > 0.5 else 0.0

    speed = (next_obs[2]**2 + next_obs[3]**2)**0.5
    speed_factor = 1.0 - min(1.0, speed / max_speed_land)
    angle_factor = 1.0 - min(1.0, abs(angle) / max_angle_land)
    soft_landing_score = speed_factor * angle_factor
    soft_landing = w_soft_land * soft_landing_score * any_contact

    # 发动机使用惩罚（离散动作每次非零动作）
    eff_penalty = -w_eff * (0.0 if action == 0 else 1.0)

    total_reward = progress + angle_penalty + angvel_penalty + soft_landing + eff_penalty

    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'soft_landing': soft_landing,
        'efficiency': eff_penalty
    }
    return float(total_reward), components
```