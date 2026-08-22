# Response Record

# 设计理由
本轮从信号覆盖审计出发：  
- **僵尸组件清理**：`stability_angvel_penalty` 和 `vertical_speed_penalty` 的 `active_rate` 均为 0%，完全未触发，且它们涉及的角速度/垂直速度阈值设置未能捕捉 agent 的实际行为，属于无效惩罚，删除它们。  
- **信号缺口填补**：观测空间中的 `leg_1_ground_contact`（索引12）和 `leg_2_ground_contact`（索引13）可直接反映双脚接地状态，这些信号在上一轮代码中完全没有使用。双足机器人在粗糙地形上频繁摔倒（terminated 19/20），而双脚同时离地是摔倒前的典型前兆。新增“双脚离地惩罚”组件，直接对该状态施加惩罚，为 agent 提供立即的摔倒预警信号，而不依赖间接的角速度或倾角阈值。  

**数学形式**：  
- 保留原有的前进速度奖励（`forward_progress`）和倾角二次惩罚（`stability_angle_penalty`），它们具备一定作用（倾角惩罚 active_rate 3.3% 但仍有意义）。  
- 新增 `ground_penalty`：当 `obs[12] + obs[13] == 0` 时给予 `-0.3` 的常数惩罚，其他情况为 0。这避免了全时拖累，仅在危险姿态出现时才介入。  

**系数校准**：  
- 主信号 per‑step ≈ 0.24（基于 `forward_progress` 的 episode_sum_mean / len）。  
- 新增惩罚 per‑step 预计活跃率约 5‑10%，实际期望惩罚 ≤ 0.03，满足 ≤ 主信号 0.3× 的约束。  

**为什么这是更优方向**：  
- 过去连续多轮尝试通过增加角速度、垂直速度惩罚来稳定躯干，但 agent 始终无法关联这些信号与摔倒，导致惩罚从未激活或起效即为破坏（len 暴跌）。  
- 现在直接利用双脚触地这一物理现实信号，将摔倒前兆与奖励紧密耦合，学习将更直接地抑制引发摔倒的姿态，从而延长存活、提高外部评分。  

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测信号
    hull_angle = obs[0]
    horizontal_speed = obs[2]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # 组件 A：前进主奖励
    forward_progress = horizontal_speed  # 系数 1.0

    # 组件 B：身体倾角稳定性惩罚（保留）
    angle_threshold = 0.5
    w_angle = 5.0
    angle_error = max(0.0, abs(hull_angle) - angle_threshold)
    stability_angle_penalty = -w_angle * (angle_error ** 2)

    # 组件 C：双脚离地惩罚（新增，替代失效的角速度与垂直速度惩罚）
    w_ground = 0.3
    both_feet_off_ground = (leg1_contact == 0.0 and leg2_contact == 0.0)
    ground_penalty = -w_ground if both_feet_off_ground else 0.0

    total_reward = forward_progress + stability_angle_penalty + ground_penalty

    components = {
        "forward_progress": forward_progress,
        "stability_angle_penalty": stability_angle_penalty,
        "ground_penalty": ground_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 接地信号完全未被使用，摔倒相关反馈缺失；角速度与垂直速度惩罚为僵尸组件。
- **behavior**: agent 能保持一定前进速度，但频繁摔倒（terminated 19/20），外部得分极低（-61.4）。
- **signal**: 缺少对双脚离地这一直接摔倒前兆的反馈，导致无法学习规避摔倒。
- **level**: Level 2
- **hypothesis**: 给双脚离地状态施加立即惩罚，将使策略学习在危险姿态时尽快恢复接地，从而大幅减少摔倒，提升存活和外部评分。
- **risk**: 若系数过大可能使步伐过于保守、步速降低；当前系数（0.3）已限定，风险可控。
