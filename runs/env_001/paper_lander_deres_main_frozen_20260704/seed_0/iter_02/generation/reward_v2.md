**evidence**: score=-111.76, 20/20 early_terminal at len≈68.5, all score<-50. progress episode_sum_mean=1.12, stability_penalty=-0.961, ratio |penalty/progress|=0.858>0.5。soft_landing_bonus active_rate=0.7%。首轮迭代，无历史修改。

**behavior_diagnosis**: agent在约68步内全部飞出边界或坠毁。每步progress≈0.016微弱为正，但stability_penalty≈-0.014几乎完全抵消进度信号（比值0.86），agent为避免惩罚不敢充分使用引擎，无法学会受控下降。

**signal_completeness**: progress（稠密Δ距离）和stability_penalty（稠密约束）职责均在位，soft_landing_bonus提供稀疏完成信号。缺乏失败惩罚和中间接近奖励，但基础信号结构对Level 1足够。

**selected_level**: Level 1——progress和stability_penalty同为逐步稠密信号，|penalty/progress|=0.858远超0.5经验触发器，且progress是唯一主引导信号。

**selected_intervention**: 仅降低stability_penalty三个系数各10倍：w_vel=0.001, w_angle=0.001, w_angvel=0.0005。其他组件不变。

**falsifiable_hypothesis**: 惩罚系数降低10倍后，|penalty/progress|应降至≈0.09，agent不再被过度约束压制引擎使用，从而能探索受控下降动作，episode延长、坠毁减少、得分改善。

**expected_next_round**: stability_penalty的episode_sum_mean应从-0.96降至约-0.10；episode_length应显著增加；score应改善（负值减小）；soft_landing_bonus的active_rate可能因存活更久而上扬。

**main_risk**: 约束过轻可能导致agent采取激进引擎策略，反而造成更严重坠毁；环境外部score可能惩罚燃料消耗，而我们的reward未对此建模。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # 目标位置（平台中心）定义为 (0,0)，因为 obs[0], obs[1] 是相对偏移
    # ------------------------------------------------------------
    # 1. 主学习信号：朝向目标的进度
    dist_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 1.0 * (dist_curr - dist_next)  # 正值表示接近目标

    # 2. 稳定性约束：大幅降低系数，使 |penalty/progress| ≈ 0.1
    linear_speed_cost = abs(next_obs[2]) + abs(next_obs[3])
    angle_cost = abs(next_obs[4])
    angular_cost = abs(next_obs[5])

    w_vel = 0.001      # 原 0.01 → 降10倍
    w_angle = 0.001    # 原 0.01 → 降10倍
    w_angvel = 0.0005  # 原 0.005 → 降10倍
    stability_penalty = - w_vel * linear_speed_cost - w_angle * angle_cost - w_angvel * angular_cost

    # 3. 软着陆近似奖励：多条件组合，引导飞行器低速、低角度、双足接触着陆
    near_target = dist_next < 0.5
    low_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5 < 0.5
    stable_angle = abs(next_obs[4]) < 0.1
    both_contacts = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contacts) else 0.0

    # 总奖励
    total_reward = progress + stability_penalty + soft_landing_bonus

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```