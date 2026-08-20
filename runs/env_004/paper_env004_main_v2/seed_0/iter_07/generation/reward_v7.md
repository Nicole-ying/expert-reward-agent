`evidence`：上一轮（iter 6）使用加速度主信号，episode长度仅71步、20/20全部terminated，acceleration_reward episode_sum_mean=6.50但步均≈0.09且伴随极弱惩罚（posture=-0.03、ang_vel=-0.04），agent无法存活。历史最佳（iter 1）使用`forward_vel * posture_factor`乘积结构达到2926分且1000步满存活，但同一骨架家族iter 2-5均未刷新best。乘积形式`forward_vel * exp(-10*θ²)`在θ=0.3时已将奖励衰减至41%，可能将agent锁在"慢速保姿态"的局部最优。

`behavior_diagnosis`：iter 6的agent在早期即摔倒终止，加速度信号的零均值噪声无法提供可学习梯度，惩罚太弱不足以挽救姿态。历史最佳agent虽能存活完整回合，但乘积耦合迫使它在前进速度与姿态之间做权衡，限制了更激进的跳跃式前进。

`signal_completeness`：前进速度信号和姿态约束信号均存在且可达。乘积耦合使两个必要信号互相抑制而非独立起作用，缺少"允许必要倾斜同时仍奖励前进"的解耦机制。不存在需要新增职责的证据缺口。

`selected_level`：Level 2。触发条件为`product_to_noncollapsing_joint`——乘积形式的`forward_vel * posture_factor`使两个目标互相压制，需转换为加性结构；同时采用`global_to_local_gate`，姿态惩罚仅在倾斜超出阈值后生效，允许跳跃必需的动态倾斜。

`selected_intervention`：将乘积耦合`speed_posture_reward = forward_vel * exp(-k*θ²)`替换为加性解耦结构：`forward_reward = forward_vel`（独立前进信号）+ `posture_penalty = w * max(0, |θ|-threshold)²`（仅超标时惩罚）。姿态从连续乘性调制器变为阈值门控的加性约束。

`falsifiable_hypothesis`：解耦后agent不再因前进加速伴随的倾斜而被乘性削弱奖励，应能学会更大幅度的前向跳跃；阈值门控允许0.3 rad以内的自然倾斜，仅在危险倾斜时施加惩罚，应同时保持存活率。

`expected_next_round`：episode长度应从71恢复至接近1000（terminated比例大幅下降），forward_reward的episode_sum_mean应显著高于iter 1的1.47（因不再被posture_factor衰减），posture_penalty在有节制倾斜下应保持低位，score应至少超过iter 1的2926。

`main_risk`：阈值门控可能在0.3 rad处产生不可导边界，若agent频繁在阈值附近振荡可能导致学习不稳定；若forward_vel原始尺度远大于预期，0.05的ang_vel_penalty可能不足以维持稳定。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    torso_angle = next_obs[1]
    forward_vel = next_obs[5]
    torso_ang_vel = next_obs[7]

    # 主信号：独立奖励前进速度，不再被姿态因子乘性压制
    forward_reward = forward_vel

    # 姿态约束：仅在倾斜超出阈值后才惩罚，允许跳跃必需的动态倾斜
    tilt_threshold = 0.3
    excess_tilt = max(0.0, abs(torso_angle) - tilt_threshold)
    posture_penalty = 1.0 * (excess_tilt ** 2)

    # 角速度约束：抑制快速翻滚，与历史最佳相同的线性形式
    ang_vel_penalty = 0.05 * abs(torso_ang_vel)

    total_reward = forward_reward - posture_penalty - ang_vel_penalty

    components = {
        'forward_reward': forward_reward,
        'posture_penalty': -posture_penalty,
        'ang_vel_penalty': -ang_vel_penalty
    }

    return float(total_reward), components
```