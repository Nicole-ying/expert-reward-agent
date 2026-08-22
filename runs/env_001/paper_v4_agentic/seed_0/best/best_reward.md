# 设计理由
本轮进行 Level 2 结构变换，修改 **approach_bonus** 组件。  
**当前问题**：虽然 agent 已学会靠近目标并减速，但 approach_bonus 仅依赖位置、速度和姿态，完全没有使用接触标志（`obs[6]`/`obs[7]`）。这使奖励在悬停未着地时仍能持续获得，可能导致部分 episode 未真正稳定着陆即终止（或延长悬停时间）。  
**修改**：在 approach_bonus 中引入双腿接触因子 `contact_factor = next_obs[6] * next_obs[7]`，将奖励形式改为  
`2.0 * prox * speed_factor * angle_factor * (0.5 + 0.5 * contact_factor)`。  
- **未接触时**：baseline 系数 0.5，保持与原来 *一半* 的密集引导，防止梯度消失；  
- **双腿接触时**：系数恢复为 1.0，达到原有最大奖励，鼓励 agent 尽快完成真实着陆。  
**系数校准**：保持 2.0 系数不变，最大单步奖励仍为 2.0，不突破约束；progress 和 safety 分量不变。  
**预期效果**：引导 agent 在接近目标后尽快放下双腿并接触平台，从而增加真正成功着陆的概率，减少无接触飘荡时间，可能进一步缩小 episode length。  
**风险**：未接触时奖励减半可能微妙地降低早期靠近的动力，但 progress_reward 仍为正向信号，历史表现已表明其足以驱动 agent 向目标移动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态（用于距离计算）
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    l_contact = next_obs[6]
    r_contact = next_obs[7]

    # ---------- 1. 进度奖励：向目标 (0,0) 靠近 ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. 着陆预备与接触奖励 ----------
    prox = 1.0 / (1.0 + 10.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 5.0 * (abs(nx_v) + abs(ny_v)))
    angle_factor = 1.0 / (1.0 + 3.0 * (abs(n_angle) + abs(n_ang_v)))

    # 双腿接触作为连续因子（二值乘积，0 或 1）
    contact_factor = l_contact * r_contact

    # 混合奖励：未接触时保留一半引导，接触时获得完整奖励
    approach_bonus = 2.0 * prox * speed_factor * angle_factor * (0.5 + 0.5 * contact_factor)

    # ---------- 3. 着陆安全性惩罚 ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate_safety = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.03 * vel_pen + 0.02 * ang_pen + 0.03 * tilt_pen) * gate_safety

    # ---------- 总奖励 ----------
    total_reward = progress_reward + approach_bonus - landing_safety_penalty

    components = {
        "progress_reward": float(progress_reward),
        "approach_bonus": float(approach_bonus),
        "landing_safety_penalty": float(landing_safety_penalty)
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有运动学信号已使用，但双腿接触标志（obs[6]/obs[7]）完全未参与奖励，构成信号缺口；当前成功率高但仍可强化真实着地要求。
- **behavior**: agent 已学会靠近目标并减速，但可能在某些 episode 中仅维持低空悬停而未稳定接触即终止。
- **signal**: approach_bonus 缺少接触约束，导致无需双腿着地即可获得大部分奖励。
- **level**: Level 2
- **hypothesis**: 在 approach_bonus 中引入双腿接触因子，使未接触时奖励减半、接触后完整发放，将鼓励 agent 主动完成真实着陆，进一步提升成功精度并减少无接触飘荡。
- **risk**: 未接触时奖励减半可能微幅减缓初始靠近速度，但 progress_reward 仍提供充足的前进动力，整体学习应平稳。