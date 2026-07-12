# Response Record

# 设计理由
**正常模式 — Level 2 结构变换：添加动作能量惩罚组件**

## 信号覆盖审计结论
- **终止→前兆**：只有倾斜摔倒前兆（hull_angle gate），但观察当前步态被截断的根本原因——高能耗步态会导致步数耗尽截断，但代码中完全没有动作效率信号。
- **目标→进度**：前向速度有门控奖励，但线性奖励不足以驱动策略加速。
- **效率缺口**：动作空间 4 维连续力矩，完全缺少动作惩罚。策略学会使用最大力矩、挥霍能量的步态，虽然维持了中等速度但效率极低，被步数截断。
- **僵尸组件**：无（active_rate 均正常）。
- **一句话结论**：当前奖励函数缺**动作效率信号**，导致策略挥霍能量、受步数截断，无法突破当前速度平台。

## 行为诊断
- **agent 在做什么**：策略以高功耗、相对稳定的步态维持 0.34 m/s 的前向速度，但步长从 911 降至 803，且终止率 100%（可能因步数截断而非摔倒）。这暗示动作成本过高，导致在有限的步数内未能覆盖足够距离。
- **干预哪个目标**：**动作效率**。当前主奖励（velocity_gated）和稳定性惩罚的组合已产生稳定步态，但缺少对能耗的约束，策略趋于使用高力矩、低效步态。
- **这个方向还值得继续吗**：稳定性惩罚提升 10 倍后分数反而下降 51 分，说明稳定性不是瓶颈——过度惩罚稳定性可能抑制了必要的动态平衡。现在应转攻能量效率，这是完全不同且尚未尝试的方向。

## 变换选择
根据算子切换表 — **add 新组件**：当前代码缺动作能量惩罚，且该缺失很可能是分数止步不前的原因。动作能量惩罚属于连续 bounded factor 型惩罚（力矩∈[-1,1]，平方和∈[0,4]），直接抑制高功耗模式。

## 系数校准
- 主信号 per-step：432 / 803 ≈ 0.54
- 新惩罚目标 per-step：≤ 0.54 × 0.3 = 0.16
- 动作平方和均值约 1.0（均匀随机动作下），用 w_action=0.3 时惩罚约 -0.3，偏高。用 w_action=0.1 时惩罚约 -0.1，加上稳定性惩罚 -0.001，总惩罚负担 0.101 < 0.54 × 0.5 = 0.27。满足约束。
- 同时降低稳定性惩罚系数回 0.1（从 0.5），因为它的实际效果只有 0.001/step，过度放大反而干扰探索。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角，越小越稳
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度，正值向前

    # ---------- 健康门控：当躯干倾斜过大时自动衰减前向奖励 ----------
    danger_angle = 0.8   # 接近摔倒的阈值（~45°）
    max_angle = 1.2      # 完全关闭主奖励的阈值（~69°）
    # 线性衰减门：在 [0, danger_angle] 恒为 1，在 [danger_angle, max_angle] 从 1 线性降到 0
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号：被门控的前向速度 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：温和地惩罚角速度，减少晃动 ----------
    w_ang_vel = 0.1  # 回退到合理范围，避免抑制必要动态
    stability_penalty = -w_ang_vel * (hull_ang_vel_next ** 2)

    # ---------- 动作效率：惩罚动作力矩平方和，降低能耗 ----------
    w_action = 0.1  # 约 0.1/step，不到主信号的 20%
    action_efficiency = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- 总奖励与组件字典 ----------
    total_reward = velocity_gated + stability_penalty + action_efficiency
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty,
        "action_efficiency": action_efficiency
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 漏了动作效率信号——4 维连续力矩完全无惩罚，导致高能耗步态被步数截断。
- **behavior**: agent 以稳定但能耗极高的步态维持中等速度，因步数截断无法走远。
- **signal**: 缺动作能量惩罚（动作平方和），该缺口是步长下降的根因。
- **level**: Level 2
- **hypothesis**: 加动作惩罚后，策略被迫学习低功耗、高续航的步态，在不牺牲速度的前提下延长行走距离 → 同时提升主奖励和步长 → 分数上升。
- **risk**: 动作惩罚系数 0.1 若过高，可能抑制探索性步态导致速度下降；需监控 velocity_gated 是否随迭代降低，若降则需适当回调系数。
