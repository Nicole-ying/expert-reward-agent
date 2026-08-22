# Response Record

# 设计理由
这是第一轮反思迭代。agent 平均存活 246 步后全部摔倒（20/20 terminated），forward_progress 每步约 0.316，说明有持续前进能力，但姿态稳定性不足最终导致跌倒。当前奖励只有 `forward_progress`（obs[2]）和 `balance_penalty`（obs[0], obs[1]），完全忽略了腿接地状态（obs[12-13]）和垂直速度（obs[3]）。
信号覆盖审计发现：leg_1/2_ground_contact 和 vertical_speed 未被使用，而这二者是摔倒前最常见的先兆信号——双脚同时离地并伴随向下的垂直速度往往意味着即将失去控制。因此本轮新增一个 `air_stability_penalty` 组件，使用双脚离地指示与垂直速度进行惩罚，促使 agent 保持至少单脚接地、避免弹跳/硬着陆，从而降低摔倒概率。

**数学形式**：
- `both_feet_off = max(0, 1 - (leg1 + leg2))`：双脚离地时为 1，单脚/双脚接地时为 0。
- `air_penalty = -0.3 * both_feet_off`：对所有腾空帧施加微弱惩罚，防止 agent 养成双脚同时起跳的习惯。
- `vertical_fall_penalty = -1.0 * both_feet_off * max(0, -vertical_speed)`：当双脚离地且向下运动时加重惩罚，抑制硬着陆。
- 总和为一个组件 `air_stability_penalty`，既不过细也不过于稀疏。

**系数校准**：
- 主信号 per‑step ≈ 0.316，目标惩罚负担 ≤ 0.095。
- 假设 `both_feet_off` 活跃率 ≈ 15–20%，则 `air_penalty` 平均 per‑step ≈ −0.05，`vertical_fall_penalty` 更稀疏，合计大概率 ≤ 0.07，安全。
- 原有 `balance_penalty` 平均仅 −0.011，几乎无负担。
- 新增惩罚不会导致 reward 塌缩或过度抑制行走。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # 1. Forward progress (unchanged)
    forward_progress = horizontal_speed

    # 2. Balance penalty (unchanged)
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angvel_excess = max(0.0, abs(hull_angvel) - angvel_threshold)
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    # 3. Air-stability penalty (NEW)
    #    Punish having both feet off the ground, especially when falling downward.
    #    leg contact is 0/1, so sum 0 => both off, 1 => one on, 2 => both on.
    both_feet_off = max(0.0, 1.0 - (leg1_contact + leg2_contact))
    # Base penalty for any airborne frame (small, to allow natural brief flight)
    air_penalty = -0.3 * both_feet_off
    # Extra penalty when airborne and descending (hard landing / falling)
    vertical_fall_penalty = -1.0 * both_feet_off * max(0.0, -vertical_speed)
    air_stability_penalty = air_penalty + vertical_fall_penalty

    total_reward = forward_progress + balance_penalty + air_stability_penalty

    components = {
        'forward_progress': forward_progress,
        'balance_penalty': balance_penalty,
        'air_stability_penalty': air_stability_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺失腿接触与垂直速度信号，这两者是摔倒前最直接的先兆，当前 reward 未覆盖。
- **behavior**: agent 能以 ~0.32/step 前进但平均 246 步后摔倒，终止全为 terminated（摔倒）。
- **signal**: 缺少“双脚离地”和“坠落速度”的惩罚信号，导致 agent 在跳跃/腾空时无负反馈。
- **level**: Level 2
- **hypothesis**: 加入双脚离地与垂直速度惩罚后，agent 将学会抑制不安全的腾空行为，保持至少单脚接地，从而减少摔倒、延长存活并最终提高前进总分。
- **risk**: 若 `air_penalty` 系数偏大，可能过度抑制自然步态中的短暂腾空，导致动作僵化、前进速度下降。可通过后续迭代调小或改为仅依赖垂直速度条件来缓解。
