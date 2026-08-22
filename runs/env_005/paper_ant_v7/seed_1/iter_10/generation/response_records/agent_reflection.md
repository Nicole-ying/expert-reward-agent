# Response Record

# 设计理由

## 第 0 步审计结论
- 终止条件（高度 <0.2 m、NaN）在代码中有前兆门控（height_factor 在 0.35→0.2 渐变），但 agent 在危险区得不到任何正向奖励，门控只削弱前进奖励，无法教会 agent 主动恢复安全姿态。
- 目标（稳定前进速度）有 `forward_speed_reward` 直接梯度，但它完全被门控因素乘积包裹，多次塌缩导致前进信号几乎消失。
- 效率信号尚可暂缓；僵尸组件 `height_penalty`、`upright_penalty` 始终为零，可移除。
- **一句话**：当前奖励缺少独立的**生存正强化组件**，导致 agent 在学站立之前无法利用前进奖励，频繁早期终止。

## 行为诊断
- agent 现在做的是：**快速跌倒**（65% 提前终止，len ≈474，terminated 率极高），它没有学会稳定站立，门控无法生效。
- 体征：`forward_speed_reward` 仅占 15% 的奖励量，但 `height_factor` 和 `upright_factor` 作为监控值占比巨大——它们不是奖励，只是 gate 因子，说明 gate 经常很低，导致真正奖励很小。
- 历史上干预多集中在改造门控或惩罚，均未改善站立能力，**连续多轮预判 ❌**，此时最紧迫的是补充**独立生存奖励**给站立本身以正反馈。

## Level 2 结构变换：添加生存奖励组件
- 用 `survival_reward = 0.03 * height_factor + 0.04 * upright_factor`
  - 直接用已有的连续因子（已从危险边界 0.2→0.35→1 映射至 0→1），不需要新 hinge。
  - 系数校准：当前前进奖励 per-step 约 0.247，站立奖励合计期望约 0.048（<0.3 倍），保持主导权在前向。
  - 即使 gate 完全切断前进奖励，agent 也能从维持高度和直立中获益，从而获得“站稳即奖励”的梯度，降低早期终止。
- 保留原有乘积门控前进奖励不变，仅增加生存项；不增加任何新惩罚。

## 设计校准检查
- 新组件 per-step 期望 < 前进 per-step 的 0.3 倍 ✓
- 门控无塌缩风险（生存组件独立于门控）✓
- 无总惩罚负担 ✓
- 移除僵尸组件，简化输出。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆解
    body_z = obs[0]
    quat_x, quat_y = obs[2], obs[3]
    body_x_velocity = obs[13]

    # 计算躯干对世界 z 轴的投影（直立度，1 为完全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 健康门控因子 ----------
    height_factor = max(0.0, (body_z - 0.2) / 0.15)      # 0.35→1, 0.2→0
    height_factor = min(1.0, height_factor)

    upright_factor = max(0.0, (body_up_z - 0.7) / 0.2)  # 0.9→1, 0.7→0
    upright_factor = min(1.0, upright_factor)

    # 门控乘积
    health_gate = height_factor * upright_factor

    # ---------- 主奖励：门控前进速度 ----------
    forward_speed_reward = 0.5 * (body_x_velocity ** 2)
    gated_forward_reward = forward_speed_reward * health_gate

    # ---------- 新增：生存奖励（直接鼓励保持高度与直立）----------
    survival_reward = 0.03 * height_factor + 0.04 * upright_factor

    total_reward = gated_forward_reward + survival_reward

    components = {
        "forward_speed_reward": gated_forward_reward,
        "survival_reward": survival_reward,
        "height_factor": height_factor,
        "upright_factor": upright_factor
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前奖励缺少独立生存正反馈，仅有门控切断前进奖励，导致 agent 在稳定站立前无法获得有效学习信号。
- **behavior**: agent 频繁跌倒，提前终止率高达 65%，前进奖励几乎失效。
- **signal**: 缺“站立即奖励”的生存梯度；前进奖励受门控乘积压制在危险区。
- **level**: Level 2
- **hypothesis**: 添加线性生存奖励（小系数高度+直立因子）提供持续正强化，引导 agent 先学会存活，门控前进奖励随后自然推动行走；预期 len 上升、terminated 下降、score 显著改善。
- **risk**: 可能诱导 agent 仅站立不前进，但系数远小于前进潜力（站立约 0.05/步 vs 前进约 0.25/步），整体仍以前进为主；若后期停滞可微调系数。
