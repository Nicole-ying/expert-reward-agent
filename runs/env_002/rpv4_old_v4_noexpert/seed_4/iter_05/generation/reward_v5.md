## 1. 诊断报告

**evidence**
- `read_memory()` + `read_training_feedback(1/4)`：iter_1（`forward_progress=2v/(1+|v|)` + `upright_stability`）是历史最佳 **278.99**，`upright_stability ep_sum=-6.678`（近乎 0），len=1275.6，反推 `v≈0.376 m/s` —— 躯干非常直立、稳定，但速度偏低。
- iter_4（`1.5*clip(v,-1,1)`）：score=261.45，len=1061，`upright_stability ep_sum=-45.555`（是 iter_1 的 6.8 倍），`forward_progress ep_sum=718.594`，反推 `v≈0.451 m/s`。`score_range=[6.1, 292.7]` 跨种子极其脆弱。
- `read_checkpoint_trend(4)`：cp50% 时 score 骤升到 64.4、`forward_progress` 冲到 284，cp60% 立刻跌回 -74.4 —— 典型 exploit 中途崩坏特征（快而不稳）。
- `read_checkpoint_trend(4)` 的 `upright_stability` 最终 -38.2：按角度惩罚反推，倾角约 `θ≈0.172 rad ≈ 9.8°`，角速度惩罚几乎为 0 —— 说明 iter_4 学到的不是快速旋转摔倒，而是**维持恒定前倾姿态去换取更高水平速度**。
- `get_component_history(upright_stability)`：只有 iter_1 的 upright 惩罚绝对值很小（-6.7），其余 -43/-73/-45.6 的迭代分数全部更低 —— 分数与「直立清洁度」强相关。
- iter_3（把增益提到 2.5）也把 upright 恶化到 -73，说明「加性 速度奖励+直立惩罚」框架下，单纯加强速度激励必然诱导前倾。

**behavior_diagnosis**
当前 iter_4 策略学到的行为模式是**靠躯干前倾追速度**：躯干维持在约 9.8° 前倾，换取水平速度 0.376→0.451 m/s。这在部分种子上能跑到 292.7，但整体更脆弱（len 1061 < 1275，score_range 极宽，cp 中途崩坏）。这是不健康的 exploit：它用稳定性换速度，导致均分低于 iter_1 的稳定慢速步态。

**signal_completeness**
职责上「前进速度」与「直立稳定」都有，但二者是**加性解耦**的：agent 可以自由地把「直立预算」交易成「速度收益」。缺失的是一个**速度-姿态耦合信号**——速度奖励应当以保持直立为前提发放，而不是无条件发放。

**selected_level**
L2。理由：iter_2（加能量项）、iter_3（提增益）都是 L1 系数/权重方向，已实证失败；iter_4（clip 换形态）也未解决 speed-vs-lean 的权衡。问题本质是加性分解允许前倾套利，需要把速度奖励改成「乘性姿态门控」这一数学形态变化。不升 L3：主信号框架（前向速度+直立稳定）在 iter_1 已被验证有效，无需推翻。

**selected_intervention**
目标组件 `forward_progress`：保留 iter_1 验证过的饱和速度项 `2v/(1+|v|)`（上限 2.0，与直立惩罚上限 2.0 对齐的安全预算），但乘上一个直立门控 `upright_gate = 1/(1+3θ²)`。`upright_stability` 保持 iter_1 已验证形态不变。单一语义修改：**速度奖励以保持直立为前提**。

**falsifiable_hypothesis**
前倾套利的经济账会从「+0.113 速度收益 vs +0.038 姿态惩罚（净赚）」翻转为「门控后速度净收益 +0.03 vs 姿态惩罚 +0.038（净亏）」，因为 θ≈0.172 时门控把速度奖励压低约 8%，而 θ≈0.059（iter_1 正常步态）时门控仅衰减约 1%，几乎不影响已验证的好步态。因此策略会回到近直立步态、同时保留速度激励；预期 `upright_stability` 回到接近 iter_1 的小量级，方差收窄，score 至少不劣于 iter_1。

**expected_next_round**
策略应学回直立稳定步态：`upright_stability` 的 |ep_sum| 回落到 15 以内，len 恢复到 1200+，seed 方差收窄，score 应回到 278 附近或更稳，而不是 iter_4 的 261 且 range=[6, 293]。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度（保留 iter_1 验证的饱和形态），乘以上身直立门控。
    # 历史证据：iter_4 用 1.5*clip(v,-1,1) 提供恒定边际梯度，策略靠躯干前倾
    # （θ≈0.172 rad）把 v 从 0.376 提到 0.451，upright_stability 从 -6.7 恶化到 -45.6，
    # len 从 1275 掉到 1061，score 261.45 < iter_1 278.99；iter_3 提高增益也复现前倾 exploit。
    # 加性「速度奖励 + 直立惩罚」无法阻止这种套利，这里改为乘性门控：
    # 只有躯干保持直立时速度奖励才足额发放，使「前倾换速度」失去净收益。
    horizontal_velocity = next_obs[2]
    raw_forward = 2.0 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    torso_angle = next_obs[0]
    # 直立门控：θ≈0.059（iter_1 正常步态）时约 0.99，几乎不衰减；
    # θ≈0.172（iter_4 前倾 exploit）时约 0.92，显著压低速度奖励。
    upright_gate = 1.0 / (1.0 + 3.0 * torso_angle * torso_angle)

    forward_progress = raw_forward * upright_gate

    # 稳定性约束：保持 iter_1 已验证的形态（躯干倾角 + 躯干角速度）。
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    total_reward = forward_progress + upright_stability

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
    }

    return float(total_reward), components
```