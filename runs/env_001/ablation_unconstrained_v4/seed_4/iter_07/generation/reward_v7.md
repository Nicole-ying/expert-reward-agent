**分析**

1. **agent 发生了什么？**  
   当前所有 20 条评估轨迹全部以失败终止（terminated=20/20），且 13/20 在 150 步内提前崩溃（score<-50）。`contact_reward` 激活率仅 1.7%，说明几乎无法完成着陆；`progress` 激活率 72.8% 但均值只有 1.34 — 大部份步数虽在移动，但奖励被 speed/angle gate 严重压缩至接近 0，导致学习信号消失，策略退化。

2. **最值得干预的组件？**  
   `progress` 的 `gate_speed` 和 `gate_angle`（v_max=1.0, a_max=0.5）过于严格，使绝大多数步的 progress 为 0 甚至无法区分方向；同时缺少对超速和过度倾斜的有效约束，使得 crash 无法避免。应移除 gate，恢复无约束的 progress 作为基础前进信号，并用 hinge 惩罚替换全时惩罚来抑制危险姿态。

3. **我之前改了什么？**  
   上一轮（iter5）曾加入 fuel_penalty 导致崩溃，随后在 iter6 删除了所有独立惩罚，仅保留 `contact_reward + progress_gated`，结果更差。当前远差于 best（iter4: 140.27）。因此应回到 best 的核心结构（progress + contact + 速度/角度约束），但通过 hinge 惩罚和微量燃油代价寻求突破。

**修改方案**  
- 移除 progress 的速度/角度 gate，直接用 `w_progress * (prev_dist - next_dist)` 提供持续前进奖励。  
- 接触奖励调整为 `base + quality`，其中 `quality` 由 `speed` 与 `angle` 线性衰减因子相乘得到，鼓励稳定着陆。  
- 对 `speed` 和 `angle` 采用 **hinge 惩罚**：超过安全阈值才开始惩罚，避免扼杀正常飞行姿态。  
- 加入极微小 `fuel_penalty`（惩罚任何引擎点火）以激励节省燃料、缩短步数。  
- 所有权重参考 best 量级重新配平，保证主要激励来自 `contact_reward + progress`。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- weights & thresholds -----
    w_progress = 10.0
    w_contact_base = 5.0
    w_contact_quality = 40.0
    v_target = 0.5          # speed where quality reaches 0
    a_target = 0.3          # angle (rad) where quality reaches 0
    w_speed_penalty = 0.1
    w_angle_penalty = 0.1
    safe_speed = 0.8        # no penalty below this speed
    safe_angle = 0.3        # no penalty below this angle (rad)
    w_fuel = 0.02           # tiny fuel penalty per firing step

    # ----- 1. distance progress (no gate) -----
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_dist - next_dist
    progress_reward = w_progress * progress

    # ----- 2. contact reward (base + quality) -----
    contact = next_obs[6] * next_obs[7]  # 1.0 only when both legs on pad
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    # linear quality factors: 1 -> 0 as speed/angle approach v_target/a_target
    q_speed = max(0.0, 1.0 - speed / v_target)
    q_angle = max(0.0, 1.0 - angle / a_target)
    quality = q_speed * q_angle
    contact_reward = w_contact_base * contact + w_contact_quality * contact * quality

    # ----- 3. hinge penalties (protect against too fast / too tilted) -----
    speed_penalty = -w_speed_penalty * max(0.0, speed - safe_speed)
    angle_penalty = -w_angle_penalty * max(0.0, angle - safe_angle)

    # ----- 4. fuel penalty (discrete action) -----
    fuel_penalty = -w_fuel * (1.0 if action != 0 else 0.0)

    # ----- total -----
    total_reward = progress_reward + contact_reward + speed_penalty + angle_penalty + fuel_penalty

    components = {
        'progress': progress_reward,
        'contact_reward': contact_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components
```