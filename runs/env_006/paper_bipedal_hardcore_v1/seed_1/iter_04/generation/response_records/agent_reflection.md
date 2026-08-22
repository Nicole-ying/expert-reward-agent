# Response Record

# 设计理由
**改动组件**：新增 `vertical_velocity_penalty`（垂直速度惩罚），保持原有所有组件不变。  
**为什么**：  
- 第 0 步审计发现信号严重缺失——当前奖励只用了 3/24 的观测维度，完全忽略了垂直速度、脚触地等关键稳定性信号。  
- 训练反馈显示所有 episode 均为 terminated（非 truncation），平均长度仅 255 步，早期终止占 6/20，说明 agent 主要在摔倒中结束，而不是正常行走。  
- 累积记录中，上次引入角度门控后长度从 411 暴跌至 222，本轮仅微升至 255，预判连续 ❌，纯角度门控无法阻止摔倒。  
- **新增垂直速度惩罚**利用未使用的 `obs[3]`（vertical_speed），当机器人急速下坠（常见于摔倒前兆）时施加代价，将其与前进奖励形成对冲，迫使策略在起跳或失稳前主动调整姿态，而不是等摔倒后才接收失败信号。  

**数学形式**：  
`penalty = -0.15 * max(0.0, -0.7 - obs[3]) ** 2`  
- 阈值 −0.7 m/s：假设正常行走垂直速度波动在 ±0.5 左右，下坠超过 −0.7 表示异常加速下落（摔倒预兆）。  
- 二次型避免过度惩罚小波动，同时快速放大严重失稳的代价。  
- 系数 0.15 经校准：主信号 per‑step ≈ 105.2 / 255.7 ≈ 0.411，该惩罚预期 active_rate 低（仅下坠时触发），即使触发 per‑step 也远小于 0.12，满足 ≤0.3× 主信号的设计规则。

**代码**：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测提取
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]          # 新增：垂直速度

    # 基础前进奖励
    progress_base = 1.0 * horizontal_speed

    # 稳定性门控：abs(hull_angle) > 0.15 时开始削弱前进奖励
    angle_deviation = abs(hull_angle) - 0.15
    gate = 1.0 - 2.0 * max(0.0, angle_deviation)
    gate = max(0.0, gate)
    progress_reward = progress_base * gate

    # 躯干角速度二次惩罚（保留，轻微抑制剧烈旋转）
    angular_velocity_penalty = -0.1 * (hull_angular_velocity ** 2)

    # 动作效率惩罚
    action_efficiency_penalty = -0.01 * sum(a * a for a in action)

    # 新增：垂直速度惩罚 —— 阻止快速下坠（摔倒前兆）
    # 当 vertical_speed < -0.7 时触发，二次项使惩罚随下坠速度急剧上升
    vertical_velocity_penalty = -0.15 * (max(0.0, -0.7 - vertical_speed) ** 2)

    total_reward = (progress_reward +
                    angular_velocity_penalty +
                    action_efficiency_penalty +
                    vertical_velocity_penalty)
    components = {
        'progress_reward': progress_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_efficiency_penalty': action_efficiency_penalty,
        'vertical_velocity_penalty': vertical_velocity_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测空间严重利用不足（3/24），遗漏垂直速度、脚触地等关键稳定性信号；当前纯角度门控无法解释并防止摔倒，信号缺口明确。  
- **behavior**: agent 频繁摔倒，平均存活 255 步，主要因快速下坠导致倾倒，前进奖励虽高但被不稳定步态拉低总分。  
- **signal**: 缺乏对“坠落”的即时预警信号，导致策略无法在摔倒发生前修正动作。  
- **level**: Level 2（新增组件填补信号缺口）  
- **hypothesis**: 垂直速度惩罚会让上升或快速下坠的动作付出代价，迫使策略学习更平缓的垂直运动，从而减少摔倒，延长 episode 长度并提升最终得分。  
- **risk**: 若阈值或系数过强，可能使 agent 过分保守（如不敢迈步），导致前进速度下降。早期的 per‑step 惩罚负担已保持在主导奖励的 0.3 倍以下以降低该风险。若长度回升但速度停滞，后续可通过调整系数或引入脚触地 gate 进一步改善。
