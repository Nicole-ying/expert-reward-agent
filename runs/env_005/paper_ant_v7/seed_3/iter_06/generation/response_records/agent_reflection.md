# Response Record

# 设计理由

## 为什么以前都失败了

回顾全部 5 轮迭代：

| 结构 | 核心机制 | 结果 |
|---|---|---|
| iter 1（简单奖励和） | forward + height + upright 求和 | score=67.7, len=201 — **唯一正分** |
| iter 2（门控+独立高度） | forward×gate + height_reward | len=463 但 score=-271 |
| iter 3（+动作惩罚） | 同上加 action_penalty | len=369, score=-112 |
| iter 4（全门控+动作惩罚） | forward×height_gate + action_penalty | len=724, score=-55.5 |
| iter 5（独立高系数惩罚） | forward + 100×height² + 5×upright² + lateral | **len=39, score=-5** |

关键教训：

1. **独立高系数惩罚是毒药**。iter 5 把 iter 4 的门控改成独立惩罚（w_height=100），结果 len 从 724 暴跌到 39。agent 没有时间从惩罚中学习，因为在惩罚触发前就已经死了。active_rate=0% 证实：高度和姿态惩罚从未触发——agent 的死因不在 hinge 覆盖范围内。

2. **门控延长生存**。iter 4 的 `forward × height_gate` 让 agent 活了 724 步（最长），因为 gate 在接近危险时衰减前进奖励，给了 agent 一个"软刹车"而不是在越界后补一刀。

3. **iter 4 为什么得分还是负？** action_penalty 每步扣除约 0.043，724 步累积约 -31。gate 让有效前进奖励打了折扣。两者叠加导致净奖励不足以覆盖 baseline。另外 gate 只覆盖下界（body_z ≤ 0.2），如果 agent 因上界（body_z ≥ 1.0）死亡，下界 gate 完全无效——这正是 iter 5 的 audit 发现：**上界前兆缺失**。

4. **iter 1 的简单结构反而是最好的起点** — 正分 67.7 说明"前进+姿态+高度"的三元组足够驱动学习，只是 len=201 偏短。

## 新骨架设计：门控前进 + 双界保护 + 轻量辅助

**核心思路：** 融合 iter 1 的简洁信号结构和 iter 4 的门控机制，同时补上上界保护。

### 算子选择

| 组件 | 算子 | 为什么 |
|---|---|---|
| 前进奖励 | dense_state_signal（线性正奖励） | 持续前进是唯一任务目标，body_x_vel 直接可观测 |
| 高度门控 | soft_health_gate（双界线性衰减） | 上下界都有终止风险，gate 在越界前软切断前进奖励，比独立惩罚更早给出梯度 |
| 姿态奖励 | dense_state_signal（线性正奖励，轻量） | 保持直立是辅助目标，继承 iter 1 的有效做法，系数极小不主导 |
| 横向抑制 | quadratic_penalty（轻量） | 抑制侧向漂移，但不压制探索 |

### 与已尝试过的本质不同

- **vs iter 4（单界门控）**：增加了上界门控，防止 agent 跳跃过高触发 body_z ≥ 1.0 终止
- **vs iter 5（独立惩罚）**：放弃高系数二次惩罚，回归门控结构。gate 在 body_z ∈ [0.35, 0.85] 内=1.0，不干扰正常前进
- **vs iter 1（简单求和）**：用 gate 替代独立 height_reward，提供更智能的高度保护
- **无 action_penalty**：历史数据明确显示它降低 score（iter 3→-112, iter 4→-55），新骨架不加入

### 系数校准

- `w_fwd=1.5`：与 iter 5 相同，per-step 约 1.5 在 body_x_vel≈1 时，足够驱动前进
- gate 衰减区间：[0.2→0.35] 下界，[0.85→1.0] 上界。在 0.3 时 gate≈0.67，在 0.9 时 gate≈0.67，确保"不理想但安全"区域 gate≥0.3
- `w_up=0.1`：per-step 约 0.08-0.1，不到主信号的 10%
- `w_lat=0.3`：per-step 约 -0.05（body_y_vel≈0.4 时），不到主信号的 5%

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract observation indices ----------
    body_z_next = next_obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]

    # ---------- forward velocity reward (primary) ----------
    w_fwd = 1.5
    forward_reward = w_fwd * body_x_vel

    # ---------- height safety gate (dual-bound) ----------
    # termination: body_z <= 0.2 or body_z >= 1.0
    # safe zone: [0.35, 0.85] where gate = 1.0
    # gate decays linearly to 0 at boundaries [0.2, 1.0]
    z_low_safe = 0.35
    z_low_dead = 0.2
    z_high_safe = 0.85
    z_high_dead = 1.0

    gate_low = min(1.0, max(0.0, (body_z_next - z_low_dead) / (z_low_safe - z_low_dead)))
    gate_high = min(1.0, max(0.0, (z_high_dead - body_z_next) / (z_high_dead - z_high_safe)))
    height_gate = gate_low * gate_high

    # ---------- gated forward ----------
    gated_forward = forward_reward * height_gate

    # ---------- upright posture reward (light) ----------
    # body_up = 1 - 2*(qx^2 + qy^2), ranges [-1, 1], 1 = perfectly upright
    body_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    w_up = 0.1
    upright_reward = w_up * body_up

    # ---------- lateral drift penalty (light) ----------
    w_lat = 0.3
    lateral_penalty = -w_lat * (body_y_vel ** 2)

    # ---------- total reward ----------
    total_reward = gated_forward + upright_reward + lateral_penalty

    components = {
        "gated_forward": gated_forward,
        "height_gate": height_gate,
        "upright_reward": upright_reward,
        "lateral_penalty": lateral_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 上界终止（body_z ≥ 1.0）无前兆信号，下界 hinge 从未触发（active_rate=0%）说明死因不在低高度渐变而在其他路径
- **behavior**: agent 在约 39 步内突然死亡（terminated 100%），死前高度和姿态都在安全范围，极可能是跳跃过高触发 body_z ≥ 1.0 终止
- **signal**: 缺上界前兆保护；独立二次惩罚在终止前完全无法提供梯度（active_rate=0%）
- **level**: Level 3
- **hypothesis**: 门控结构（iter 4 验证可延长生存至 724 步）+ 双界保护（补上 iter 4 缺失的上界）+ 移除高系数惩罚和 action_penalty（iter 3-5 验证有害），应能同时恢复生存长度和正向前进奖励
- **risk**: gate 衰减区间如果设得太窄（safe zone 太小），agent 可能因频繁进入衰减区而前进奖励不足；若出现此情况，下一轮可微调 safe zone 参数而非更换骨架
