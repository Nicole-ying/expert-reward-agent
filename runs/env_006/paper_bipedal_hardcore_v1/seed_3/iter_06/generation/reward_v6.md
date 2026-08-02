# 设计理由
本轮属于**正常模式**，修改一个组件。信号覆盖审计发现：当前代码完全未使用地面接触信号 `obs[12]` 和 `obs[13]`，而该信号可直接检测双脚离地状态（空中时间）。历史记录显示，第 3 轮加入的空中惩罚曾将 `len` 从 303 提升至 394、`score` 提升至 -59.20，是唯一有效改善；该组件在第 5 轮骨架替换时被移除，导致 `len` 暴跌至 190，score 退化到 -65。根据第 0 步审计结论，**信号缺口是当前失败的核心原因**，故添加基于接触信号的连续空中惩罚组件。

**数学形式**：`contact_sum = leg_1_contact + leg_2_contact`，取值为 0/1/2。  
`air_penalty = -0.2 × max(0, 1 - contact_sum)`。  
- 双脚离地 (`contact_sum = 0`) → 惩罚 -0.2  
- 单脚着地 (`contact_sum = 1`) → 惩罚 0  
- 双脚着地 (`contact_sum = 2`) → 惩罚 0  

**系数校准**：主信号 `progress_reward` 的每步均值 ≈ 130.96 / 190.3 ≈ 0.688。`air_penalty` 最大每步 0.2，比值 0.29，满足 ≤0.3× 主信号的设计校准要求。惩罚负担可控。

僵尸组件 `ang_vel_penalty`（几乎无贡献）暂时保留，本轮不调整。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度
    horizontal_speed = next_obs[2]
    progress = 2.0 * horizontal_speed

    # 稳定/安全约束：姿态角度超出健康范围时软惩罚（hinge）
    hull_angle = next_obs[0]
    max_allowed_angle = 0.3
    posture_penalty = -5.0 * max(0.0, abs(hull_angle) - max_allowed_angle)

    # 稳定/安全约束：角速度惩罚，抑制剧烈摇晃
    ang_vel = next_obs[1]
    ang_vel_penalty = -0.05 * (ang_vel ** 2)

    # 效率/动作代价：轻微二次惩罚
    action_cost = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # 新增：空中惩罚，基于地面接触信号，抑制双脚同时离地
    contact_sum = next_obs[12] + next_obs[13]  # 取值 0/1/2
    air_penalty = -0.2 * max(0.0, 1.0 - contact_sum)

    total_reward = progress + posture_penalty + ang_vel_penalty + action_cost + air_penalty
    components = {
        'progress_reward': progress,
        'posture_penalty': posture_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'action_cost': action_cost,
        'air_penalty': air_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺口：`obs[12]`/`obs[13]`（地面接触）未被利用，缺失空中惩罚，历史上加入该组件曾显著改善 `len` 与 `score`。
- **behavior**: agent 在多数 episode 中能向前行走，但稳定性差，约 35% 提前摔倒（len < 150），整体因摔倒终止导致负分。
- **signal**: 缺乏空中约束使 agent 不畏惧双脚离地的危险行为，导致摔倒风险升高。
- **level**: Level 2
- **hypothesis**: 加入基于接触信号的空中惩罚将抑制双脚同时离地，提高着地稳定性，减少摔倒终止，延长存活时间并积累更多前进奖励。
- **risk**: 过强惩罚可能抑制必要的跳跃动作，但 0.2 的小系数仅在双脚全离地时触发，不会影响正常单脚支撑的步态，副作用可控。