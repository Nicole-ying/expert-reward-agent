# Response Record

# 设计理由
本轮修改属于 **Level 2 结构变换**：删除僵尸组件 `vertical_penalty`（active_rate = 0%，阈值 0.5 过高导致 agent 垂直速度从未触发），替换为 **air_penalty**（基于腿部接触信号 `next_obs[12]` 与 `next_obs[13]`）。

## 为什么这样改
- 第 0 步审计发现 `vertical_penalty` 完全未激活（0%），是明确的僵尸组件，占据设计位但没有产生任何学习信号，必须移除。
- 未使用腿部接触信号是一个信号缺口，而历史记录（iter 3、6）证明 air_penalty 能有效提升 episode 长度（394/323 步）和得分。agent 当前在 ~373 步后摔倒，恰好需要这种空中惩罚来抑制危险的双脚离地行为。
- **数学形式**：连续因子 `air_factor = 1.0 - 0.5*(c1 + c2)`，双脚着地时 = 0（无惩罚），单脚 = 0.5，双脚离地 = 1.0。系数 `-0.1`，保证 per-step 惩罚 ≤ 0.05，远小于主信号 per-step （≈0.28），符合设计校准规则（≤0.3× 主信号）。
- `progress_gated` 保持不变：gate 使用 `hull_angle` 和 `hull_angular_velocity` 调制前进奖励，继续提供姿态约束。

## 本质差异
与上一轮相比，信号从无用的垂直速度阈值换成实际可触发的连续接触因子，重新引入历史验证有效的空中惩罚，直接针对摔倒前的步态不稳定问题。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extracting relevant observations
    hull_angle_abs = abs(next_obs[0])
    hull_ang_vel_abs = abs(next_obs[1])
    horizontal_speed = next_obs[2]
    leg_1_contact = next_obs[12]
    leg_2_contact = next_obs[13]

    # Core forward progress: only reward positive horizontal speed
    forward_speed = max(0.0, horizontal_speed)

    # Soft health gate: reduces forward reward when posture deteriorates
    k_angle = 5.0
    k_ang_vel = 0.5
    gate = 1.0 / (1.0 + k_angle * hull_angle_abs + k_ang_vel * hull_ang_vel_abs)

    # Gated forward progress (main learning signal)
    w_fwd = 1.0
    progress_gated = w_fwd * forward_speed * gate

    # Air penalty: discourages both feet leaving the ground simultaneously
    # air_factor = 0.0 when both feet on ground, 0.5 when one foot, 1.0 when airborne
    air_factor = 1.0 - 0.5 * (leg_1_contact + leg_2_contact)
    air_penalty = -0.1 * air_factor

    total_reward = progress_gated + air_penalty
    components = {
        'progress_gated': progress_gated,
        'air_penalty': air_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: vertical_penalty 是僵尸组件（active_rate 0%），未使用腿部接触信号，缺失空中状态预防；替换为 air_penalty 利用接触信号。
- **behavior**: agent 存活约 373 步最终摔倒，前进速度尚可，但缺乏空中惩罚导致步态不稳定，最终侧倾摔倒。
- **signal**: 缺失空中状态信号（双脚离地），历史 air_penalty 有效，垂直速度阈值过高导致无用。
- **level**: Level 2
- **hypothesis**: 加入空气惩罚将抑制双脚同时离地，延续 iter 3/6 的成功经验，进一步提高着地稳定性，延长存活并积累更多前进奖励。
- **risk**: 系数 0.1 较小，可能不足以显著改变行为；若效果不显，下一轮可微调系数或结合 angle hinge penalty。
