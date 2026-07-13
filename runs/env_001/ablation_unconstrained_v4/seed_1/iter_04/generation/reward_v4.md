分析：

1. **agent行为诊断**：最终评估episode长度980，几乎全为truncation，仅1次terminated，意味着agent未学会成功着陆。主要奖励由step_penalty（-49）和contact_reward（+20）主导，progress很小（+4.1），landing_bonus从未触发（0.0）。这表明agent可能采取缓慢移动、偶尔触地但未满足软着陆条件的策略，步惩罚使总分为负。上一轮加入step_penalty恶化了分数（best -40.47 → -62.67），说明步惩罚加剧了负偏差，没有促进效率。二进制landing_bonus条件过于严格（距离<0.15，速度<0.04等），从未触发，缺乏梯度引导。

2. **最佳干预组件**：landing_bonus需要从二值改为连续松弛版本，提供渐进式成功反馈；同时去掉有害的step_penalty，避免其吞噬所有正奖励。progress奖励应保留但改为只奖励接近（不惩罚远离），避免因探索后退而受罚；contact_reward适度保留以鼓励触地。

3. **历史修改分析**：上一轮（iter 3）在best（未提供完整代码，但描述为分段角度+contact+progress，无步惩罚）基础上添加了step_penalty、speed_penalty和二进制landing_bonus，但得分下降。因此应回退步惩罚，并将landing_bonus改造为连续形式。

**修改方案**：
- 移除`step_penalty`和`speed_penalty`。
- `distance_progress`改为`2.0 * max(0, progress)`，避免惩罚远离。
- `contact_reward`减半至`2.0 * contact_count`，防止过度奖励无效触地。
- 保持姿态惩罚`c_angle`、`c_angvel`。
- 新增**连续软着陆奖励**：当双腿接触时，根据距离、速度、角度、角速度的接近程度线性插值给予最高200分的部分奖励，使agent能从小进步中获得正反馈。

该设计与历史所有版本有本质差异（无分段角度、无步惩罚、无二进制landing、无燃料惩罚），满足“materially different”要求。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # extract current and next state
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # distances from target
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # progress (positive if getting closer)
    progress = prev_dist - next_dist
    c_progress = 2.0 * max(0.0, progress)

    # contact reward (reduced to avoid exploit)
    contact_count = n_left_contact + n_right_contact
    c_contact = 2.0 * contact_count

    # posture penalties (global)
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.01 * (n_angvel**2)

    # continuous soft-landing bonus (triggered only when both legs contact)
    two_legs = (n_left_contact > 0.5 and n_right_contact > 0.5)
    if two_legs:
        # distance, speed, angle, angular speed magnitudes
        dist = next_dist
        speed_mag = (nvx**2 + nvy**2) ** 0.5
        angle_mag = abs(n_angle)
        angvel_mag = abs(n_angvel)

        # linear interpolation towards perfect landing (score=1 if thresholds met)
        score_dist = max(0.0, 1.0 - dist / 0.3)
        score_speed = max(0.0, 1.0 - speed_mag / 0.3)
        score_angle = max(0.0, 1.0 - angle_mag / 0.2)
        score_angvel = max(0.0, 1.0 - angvel_mag / 0.3)

        c_landing = 200.0 * score_dist * score_speed * score_angle * score_angvel
    else:
        c_landing = 0.0

    total_reward = c_progress + c_contact + c_angle + c_angvel + c_landing

    components = {
        'distance_progress': c_progress,
        'contact_reward': c_contact,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
        'landing_bonus': c_landing,
    }
    return (float(total_reward), components)
```