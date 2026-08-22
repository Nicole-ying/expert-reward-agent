# Response Record

# 设计理由

本轮为 **Level 2 结构变换**，只新增一个组件：**动作幅度惩罚**。保留了原有的 `forward_gated` 与 `height_reward`。

**为什么选这个变换**  
- 上一轮 `len` 从 201→463 大幅上升，但外部评估 `score` 从 +67 暴跌至 -271。这说明 agent 学会了在健康高度下**缓慢存活**，但真实性能（可能包含能耗指标）反而恶化。  
- 当前动作空间 8 维、无任何动作成本，agent 没有任何激励去降低力矩——这正是高能耗低效运动的典型症状。  
- 引入小幅度的 L2 动作惩罚可以压低多余力矩，让 agent 更倾向于用**节能的步态**前进，直接对齐真实任务中隐含的效率需求。  
- 惩罚系数 0.01，使单步惩罚约为主信号 `forward_gated` (≈0.677) 的 3–8%，远低于 0.5 倍警戒线，不会阻碍探索，也不会支配梯度。

**数学形式**  
- `action_penalty = -0.01 * sum(action ** 2)`（8 个力矩量的平方和），范围 [-0.08, 0]。  
- 保持 `forward_gated = body_x_vel * upright_gate`，避免改动已验证的正向信号。  
- 高度惩罚保持线性越界项，系数 5.0，因其 per‑step 均值仅 -0.019，无冲击。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z      = obs[0]
    quat_x      = obs[2]
    quat_y      = obs[3]
    body_x_vel  = obs[13]
    
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)
    
    # --- 前进奖励（保持原有） ---
    upright_gate = max(0.0, body_up_z)
    forward_reward = body_x_vel * upright_gate
    
    # --- 高度惩罚（保持原有） ---
    height_margin_low  = 0.3
    height_margin_high = 0.9
    below_margin = max(0.0, height_margin_low - body_z)
    above_margin = max(0.0, body_z - height_margin_high)
    height_penalty = below_margin + above_margin
    height_reward = -5.0 * height_penalty
    
    # --- 新增：动作幅度惩罚（抑制高能耗、抖动） ---
    action_penalty = -0.01 * sum(action ** 2)
    
    total_reward = forward_reward + height_reward + action_penalty
    
    components = {
        'forward_gated': forward_reward,
        'height_reward': height_reward,
        'action_penalty': action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少身体倾斜的明确惩罚（仅以 gate 削弱前进奖励，无直接负反馈），且存在高能耗漏洞（8 维动作无成本）。
- **behavior**: agent 学到了延长生存时间（len 463）但降低真实性能的策略，大概率依赖大幅力矩维持姿态而牺牲了能效。
- **signal**: 缺少动作效率信号；前进奖励 magnitude 不足以抵消生存延长带来的无效能量开销。
- **level**: Level 2
- **hypothesis**: 轻量动作惩罚会引导策略减少不必要的力矩幅度，提升步态能效，从而直接提高外部评价分数（score），同时保持生存长度。
- **risk**: 若惩罚系数过大，agent 可能走向“完全不行动”的极端；当前系数 0.01 极小，该风险低。
