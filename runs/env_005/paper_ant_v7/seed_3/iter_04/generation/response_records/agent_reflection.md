# Response Record

# 设计理由
本轮是正常模式下的 **Level 2 结构变换**。累积记录连续两轮预判 ❌，当前骨架（forward_gated + height_reward + action_penalty）虽然内部奖励很高，但外部评价得分仍为负（-112.4），并且评价终止率高达 85%。根据信号覆盖审计，当前最大的信号缺口是 **高度保护不足** ——高度独立惩罚（height_reward）激活率仅 21.8%，说明它只在边界附近施加负值，无法在健康区间内为高度调节提供连续梯度，导致机器人频繁超出高度边界而失败。同时，外部评分可能对无谓的倾倒有严厉惩罚，而我们的奖励结构没有把“生存”与“前进”强关联。

**变换**：将 `height_reward` 独立组件 **替换为** 一个 **软健康门控（soft health gate）**，乘入前进奖励中，形成 `forward_gated_with_height = body_x_vel * upright_gate * height_factor`。这样不论是否接近边界，高度变化都会直接影响核心前进信号的强度，从而迫使策略在“保持高度”与“前进”之间进行统一优化。保留轻量动作惩罚以控制震颤。

**数学形式**：
- `upright_gate = max(0.0, body_up_z)`
- `height_factor = low_factor * high_factor`
  - `low_factor = min(1.0, max(0.0, (body_z - 0.2) / 0.15))`（从 0.2 到 0.35 线性上升至 1）
  - `high_factor = min(1.0, max(0.0, (1.0 - body_z) / 0.15))`（从 1.0 到 0.85 线性下降至 1）
- 安全区 [0.35, 0.85] 内 `height_factor = 1`，完全保留原始前进强度。

**系数校准**：
- 动作惩罚 per‑step ≈ 0.043，主信号 per‑step ≈ 0.82，总惩罚负担远低于 0.3。
- 门控边界阈值为终止边界的 60%～80%（0.2→0.35，0.85→1.0），保证梯度平缓，不会塌缩。

**期望**：提高训练中对高度的持续关注，降低评价时的终止率，从而间接改善外部得分。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z      = obs[0]
    quat_x      = obs[2]
    quat_y      = obs[3]
    body_x_vel  = obs[13]
    
    # 直立因子（保持）
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)
    upright_gate = max(0.0, body_up_z)
    
    # 高度健康因子（替换独立惩罚）
    low_factor  = min(1.0, max(0.0, (body_z - 0.2) / 0.15))   # 0.2 -> 0.35
    high_factor = min(1.0, max(0.0, (1.0 - body_z) / 0.15))   # 1.0 -> 0.85
    height_factor = low_factor * high_factor
    
    # 前进奖励（推进 + 直立门控 + 高度门控）
    forward_reward = body_x_vel * upright_gate * height_factor
    
    # 动作惩罚（保持）
    action_penalty = -0.01 * sum(action ** 2)
    
    total_reward = forward_reward + action_penalty
    
    components = {
        'forward_gated_height': forward_reward,
        'action_penalty': action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺失高度保护的前兆软信号，独立惩罚无法提供持续梯度；可能缺少独立直立奖励以对齐未知外部分数。  
- **behavior**: 策略产生高内部前进量，但频繁跌落导致高终止率，外部得分仍为负。  
- **signal**: `height_reward` 效果微弱且不可微分地推迟失败，需替换为连续门控因子。  
- **level**: Level 2  
- **hypothesis**: 高度门控乘入主要奖励会使高度调节成为前进优化的必要条件，降低终止率，从而提升外部生存相关分数。  
- **risk**: 若初期探索高度频繁越界，门控整体 reward 过低可能拖慢学习；宽安全区间已对此做了缓冲。
