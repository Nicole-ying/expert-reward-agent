我们分析当前训练反馈：score -115.86，20/20 early terminal，agent全部失败，无法着陆。对比历史记忆，当前版本移除了上一轮best（iter 4，得分140）中的angle_penalty和vel_penalty，导致agent失稳、快速crash。因此必须恢复飞行约束，并在此基础上小幅改进以逼近200分目标。

诊断证据：
- contact_reward active_rate仅0.5%，且episode平均长度68.4，说明agent在接触垫板前即终止，飞行过程中因姿态/速度失控而crash或出界。
- progress和distance_penalty之和虽为正驱动，但缺少对速度和角度的持续约束，使得agent无法保持安全状态。
- best结构包含progress、vel_penalty、ang_penalty和contact_reward，成功实现了463步稳定着陆，得分140。因此，我们基于该骨架进行恢复和增强。

修改方案：
1. **恢复并强化稳定性约束**：加入对速度和大角度的hinge惩罚，仅当超过安全阈值时生效，避免惩罚正确低速飞行。
2. **增强前进驱动力**：将**距离缩减量（progress）**进行平方凸化，使中途减速区域仍保持显著梯度，避免线性 reward 在低速微调时近乎无信号——这是逼近目标时提升效率的关键。
3. **着陆接触奖励保持乘积形式**，但降低 speed/angle 阈值使着陆质量条件更严，鼓励更精准停泊。
4. 暂不加燃料惩罚，避免干扰探索；也不加x位置惩罚（之前iter 2导致下降）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- helper: distances ---
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    # --- progress (convexify for stronger gradient near target) ---
    dist_reduction = max(0.0, prev_dist - next_dist)
    progress = 2.0 * (dist_reduction ** 2)   # squared to amplify when reduction is small

    # --- speed penalty (hinge: penalize only above safe threshold) ---
    v_thresh = 0.5
    speed_penalty = -2.0 * max(0.0, speed - v_thresh)

    # --- angle penalty (hinge, keep upright) ---
    a_thresh = 0.15
    angle_penalty = -5.0 * max(0.0, angle - a_thresh)

    # --- contact quality reward (both legs touch) ---
    contact = next_obs[6] * next_obs[7]  # 0 or 1
    v_target = 0.2
    a_target = 0.1
    q_speed = max(0.0, 1.0 - speed / v_target)
    q_angle = max(0.0, 1.0 - angle / a_target)
    quality = q_speed * q_angle
    contact_reward = 20.0 * contact * quality

    # --- total ---
    total_reward = progress + speed_penalty + angle_penalty + contact_reward

    components = {
        'progress': progress,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward
    }
    return float(total_reward), components
```