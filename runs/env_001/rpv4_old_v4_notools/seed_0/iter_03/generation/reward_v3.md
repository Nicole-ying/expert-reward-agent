**1. 诊断报告**

- **evidence**：  
  - iter_2 最终得分 -168.0，虽有大幅进步（从 cp10% 的 -758.2 提升至 cp100% 的 -166.4），但仍远低于目标 200。  
  - `contact_reward` ep_sum_mean = 0，active_rate = 0%，检查点趋势全程为 0，说明从未触发。  
  - `vel_damping` 同样全程为 0，表明高度门控阈值（1.0）过高，有效下降阶段未出现。  
  - `progress_reward` 虽活跃但 ep_sum_mean 仍为负值（-0.082），agent 整体未能持续接近目标；`orientation_penalty` 已很小（-0.01），姿态趋于稳定。  
  - 历史改善（假设 iter_1 → iter_2 提升超 30 分）主要来自 progress 权重从 1.0 降至 0.5，使优化更平滑，姿态和距离惩罚绝对值显著减小。

- **behavior_diagnosis**：  
  策略已学会部分稳定姿态并尝试靠近目标，但缺乏完成最终着陆的动力。agent 接近地面时可能因速度过快而坠毁，或因负向累积，学习到避免下降过低。因此 episode 多数以 terminate（坠毁）结束，未实现平稳着陆。

- **signal_completeness**：  
  缺少对“着陆”或“接触地面过程中的平稳性”的正反馈。当前 `contact_reward` 要求双脚同时触地，过于苛刻（从未激活），导致 agent 无动机完成下降并尝试接触。需要一个更易触发、且能引导柔和触地的着陆信号。

- **selected_level**：**L2**（改动数学形态）  
  现有框架方向基本正确，主要缺口在于着陆正反馈欠缺。上一轮未动过 `contact_reward`，本次改动不冲突。

- **selected_intervention**：  
  将 `contact_reward` 从仅双脚乘积改为 **左右脚接触之和**，并给予较大权重，同时放宽姿态/速度因子的容忍度（增大 sigma）。这样任何一条腿触地（即使不完全对称）都能获得正奖励，奖励大小随姿态与速度平滑衰减，引导 agent 主动降低高度并尝试轻柔触地。

- **falsifiable_hypothesis**：  
  引入可触发的着陆奖励后，agent 将开始学习下降并接触地面，利用姿态‑速度因子逐步趋向平稳着陆；预期下一轮 `contact_reward` 激活率上升，ep_sum_mean 变为正值，整体得分显著提升。

- **expected_next_round**：  
  策略会频繁下降到低空并尝试单脚或双脚触地，`contact_reward` 活跃，平均 episode 长度可能缩短（成功着陆终止），得分向正值大幅移动，但速度控制细节仍待后续优化（vel_damping 依旧死，下一轮可再处理）。

---

**2. 修改后的奖励函数代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: 重新设计 contact_reward，使其在任意腿触地时即可激活，
    放大权重以提供明确的着陆正激励，引导 agent 完成最终下降。
    progress/ orientation/ vel_damping 保持与 v2 相同。
    """
    # 提取信号
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward（保持 v2 权重 0.5）
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr
    w_progress = 0.5
    progress_reward = w_progress * progress

    # 2. Orientation penalty（保持不变）
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft contact reward（重写：左右脚之和，放大权重，放宽因子）
    contact_sum = left_contact + right_contact          # 0, 1, 或 2
    sigma_angle = 0.3                                   # 放宽角度容忍度
    sigma_vel = 0.5                                     # 放宽速度容忍度
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))
    vel_factor = 2.718281828 ** (- (vx ** 2 + vy ** 2) / (sigma_vel ** 2))
    soft_contact = contact_sum * angle_factor * vel_factor
    w_contact = 100.0                                    # 大幅提升权重，一次良好着陆约 200
    contact_reward = w_contact * soft_contact

    # 4. 接近地面速度惩罚（保持 v2，阈值仍为 1.0）
    altitude = y_curr
    alt_thresh = 1.0
    gate_alt = max(0.0, 1.0 - altitude / alt_thresh)
    w_vel_damp = 1.0
    vel_damping = -w_vel_damp * gate_alt * (abs(vy) + 0.3 * abs(vx))

    total_reward = progress_reward + orientation_penalty + contact_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```