1. `evidence`：上一轮在距离进展+倾斜惩罚基础上引入固定2.0的双腿接触持续奖励，得分由‑100急升到18.39，但landing_support占幅度份额97.5%、激活率仅18.1%，说明高分来源于少量接触步骤的固定收益，而非稳定着陆。
2. `behavior_diagnosis`：策略已学会触发双腿接触以获取持续奖励，但未能实现轻柔稳定的着陆；episode较长（均长321.6），最终总分偏低，部分episode可能在接触后因姿态或速度不满足终止条件而失败。
3. `signal_completeness`：现有信号缺少对“着陆瞬时或过程中的状态质量”的反馈，固定持续奖励导致exploit，需要将接触类奖励与真实任务完成程度对齐。
4. `selected_level`：Level 2 —— 固定每步奖励属于“占据好状态即可持续获奖”，应转换为与着陆状态改善（轻柔程度）挂钩的结构。
5. `selected_intervention`：只修改landing_support组件，由固定2.0持续奖励改为`landing_quality`，双腿接触时根据垂直速度大小和机体角度按`2.0/(1+|v_y|+|angle|)`递减，非接触时为0。
6. `falsifiable_hypothesis`：着陆奖励转为质量驱动后，agent将被迫降低触地速度和减小倾斜，才能获得更高回报，从而习得接近稳定着陆的行为并提升外部得分。
7. `expected_next_round`：landing_quality激活率可能上升，但单步奖励幅值下降；episode长度可能变长或变短，但真正稳定着陆的比例应增加，score的中位数和成功率有望改善；distance_progress和tilt_penalty保持不变。
8. `main_risk`：总正奖励减少，初期学习可能停滞或回到负分，需后续视情况加强progress等引导信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]

    # Distance progress toward target (0,0)
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_curr - dist_next

    # Stability: penalise large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    # Landing quality: reward based on vertical speed and body angle when both legs contact
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    if left_contact and right_contact:
        # Use vertical velocity (next_obs[3]) and body angle (next_obs[4])
        speed_angle_sum = abs(next_obs[3]) + abs(next_obs[4])
        landing_quality = 2.0 / (1.0 + speed_angle_sum)
    else:
        landing_quality = 0.0

    total_reward = progress + tilt_penalty + landing_quality

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'landing_quality': landing_quality
    }

    return float(total_reward), components
```