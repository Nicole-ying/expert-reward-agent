# 设计理由

## 为什么以前都失败了？
- **Iter 1**：原始五组件加权和（含 heavy upright_penalty），agent 存活极短（len≈11.8），因 upright_penalty 每步高达 -0.48，远大于 forward 收益，直接“压死”探索。
- **Iter 2**：移除 gate、大幅降低 upright 系数并改为温和二次型，len 暴涨至 503.5，说明 agent 首次能长存活。但 **forward (+329.7/ep) 与 upright_penalty (-367.1/ep) 近乎完全对消**，score 仍为负。核心矛盾是：agent 学会了“宁可倾斜也要快速前进”，因为 forward 在倾斜时仍全额发放，upright 惩罚力度不足以对抗 forward 的诱惑，形成冲消均衡——姿态信号始终未有效形塑行为。

## 新骨架的本质差异
我选择 **soft_health_gate × forward** 作为主骨架，而非再次调整独立惩罚系数。这不同于历史所试的任何结构：

- **之前**：forward 是无条件的正奖励，upright 只是附加的独立惩罚项；agent 可以在低姿态下肆意收割 forward 收益，事后才受罚。
- **现在**：前进奖励本身被 **upright gate** 相乘门控：直立好 → 全速奖励；开始倾斜 → forward 收益线性衰减至零；严重倾倒时前进奖励彻底消失。这样 agent 要获得任何前进好处，**必须先维持可接受的直立姿态**，从根本上消除“先冲后死”的冲消均衡。
- 额外加入一个极轻量的 **upright_bonus**（连续线性，w=0.2），提供直接的姿态改善梯度，但量级远小于 gated forward（约为主信号的 0.3x），不会取代走动动机。
- 其他组件（高度边界 hinge、侧向二次、动作二次）保持轻量且不越主信号比例（设计校准：所有惩罚 per-step 合计 < 0.5× forward per-step）。

## 组件选择与系数校准

### 主信号：gated_forward
- `forward = v_x`
- `gate = clamp((up_z - 0.3) / 0.7, 0.0, 1.0)`
- `gated_forward = forward * gate`
- up_z 从四元数计算：`1 - 2*(quat_x² + quat_y²)`。gate 在 up_z≥1 时=1，在 up_z≤0.3（≈70°倾斜）时=0。0.3 设在 **终止边界（body_z≤0.2）的 60-80% 前兆区间**，给 agent 充足信号在“危险前”纠正姿态。

### 辅助：upright_bonus
- `upright_bonus = 0.2 * up_z`（线性，正负均有梯度）
- 量级 per-step 最大 0.2，约为 forward 均值（0.65）的 0.3x，符合“不压主信号”原则。

### 保留约束（经削弱/保留）
- **height_penalty**：hinge 二次，安全区 [0.3, 0.8]，系数 w_h=0.5（历史 active_rate 仅 41%，几乎不触发，故降低权重以防占据）。
- **lateral_penalty**：`-0.2 * v_y²`（历史 -0.064/step，保持）。
- **action_penalty**：`-0.005 * mean(a²)`（几乎可忽略，保留作平滑正则）。

## 预判
- **正向**：以往 forward 被姿态惩罚抵消的困境会消失，因为 forward 只在姿态良好时有效；agent 必须“先学会站稳再跑”，upright_bonus 提供持续梯度辅助姿态收敛。len 已可长存活的条件下有望让 score 转正并持续提升。
- **风险**：初期 gate 可能抑制动作探索，若 agent 过早学习“小步慢走”以保持 up_z 始终完美，可能导致速度平淡（score 提升缓慢）。但 upright_bonus 权重很小，forward 本身有足够吸引力，此风险可控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]  # forward velocity
    v_y      = obs[14]  # lateral velocity

    # ---- upright measure (continuous, 1.0 when perfectly upright, -1.0 inverted) ----
    up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---- upright gate (soft multiplication to forward progress) ----
    # Gate starts to decay when up_z < 1.0, reaches 0.0 when up_z ≤ 0.3
    gate_lower = 0.3
    gate = (up_z - gate_lower) / (1.0 - gate_lower)
    gate = max(0.0, min(1.0, gate))

    # ---- gated forward progress ----
    w_fwd = 1.0
    gated_forward = w_fwd * v_x * gate

    # ---- upright bonus (continuous, provides direct posture gradient) ----
    w_up_bonus = 0.2
    upright_bonus = w_up_bonus * up_z

    # ---- body height safety (soft hinge quadratic, only near boundaries) ----
    z_low_safe  = 0.3
    z_high_safe = 0.8
    w_h = 0.5
    low_hinge  = max(0.0, z_low_safe - body_z)
    high_hinge = max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge**2 + high_hinge**2)

    # ---- lateral stability (quadratic penalty) ----
    w_lat = 0.2
    lateral_penalty = -w_lat * (v_y**2)

    # ---- action smoothing (light energy regularizer) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a**2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (gated_forward + upright_bonus +
                    height_penalty + lateral_penalty + action_penalty)

    components = {
        "gated_forward":      gated_forward,
        "upright_bonus":      upright_bonus,
        "height_penalty":     height_penalty,
        "lateral_penalty":    lateral_penalty,
        "action_penalty":     action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前奖励漏掉了“前进必须在姿态健康的前提下”这一因果结构——forward 与 upright 互为独立项导致冲消，需要用 gate 绑定。
- **behavior**: agent 学会长时间存活但始终在倾斜中高速前进，forward 与 upright_penalty 几乎完全对消。
- **signal**: 缺 soft_health_gate 将前进收益与直立姿态耦合；upright 信号虽活跃但形塑失败。
- **level**: Level 3（重建骨架）
- **hypothesis**: 将前进收益门控在直立姿态上，迫使 agent 必须先保持直立才能获利，能打破两大项对冲的局面，使 forward 真正转化为正面学习信号，同时 upright_bonus 给予直接姿态梯度。
- **risk**: 前期 gate 可能稍微抑制速度探索，但 upright_bonus 和已有存活能力可确保不会退回到短 episode 状态；若速度平淡，后续可微调 gate 形状或引入凸化 forward。