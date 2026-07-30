# Response Record

# 设计理由
## 审计结论
a) 终止条件：高度越界 z≤0.2 或 z≥1.0；当前代码有 `height_penalty`（hinge在 0.35~0.85，距边界有缓冲）和 `gate_z`（卡在 0.2~1.0），**覆盖**。
b) 目标信号：前向速度 `v_x` 被 `health_gate`（gate_z × gate_up）门控，且门控阈值 up_thr=0.7 很苛刻（训练反馈证实 active_rate=0.4%），**梯度被截断**。
c) 效率信号：动作维度 8 ≥ 6，已有 action_penalty，系数较小。
d) 僵尸组件：`upright_penalty` active_rate=0.4%，**几乎不触发但触发时惩罚量级是灾难性的**（episode_sum_mean=-5.673, 占比 5.8x 所有其他组件之和）。
e) 一句话结论：**health_gate 切断了唯一正向学习信号，upright_penalty 在不触发时无引导、触发时直接压死 reward，导致 agent 无梯度可用只能早期崩溃。**

## 行为诊断
- **agent 在做什么**：11.8 步即全部 early terminate。forward 仅 +0.222，upright_penalty 的 magnitude 占绝对主导。agent 完全没有机会学到有效行为——唯一的正向信号（forward）被 gate 切断 43% 步数，而惩罚项（尤其 upright_penalty）在少数触发步中就直接将总 reward 压到 -5.58 以下。
- **干预目标**：**恢复正向梯度**——移除 health_gate，让 forward 信号自由流动；**重构 upright 引导**——从灾难性间歇惩罚变为连续、温和的引导信号。
- **方向判据**：这是第一轮，无历史迭代记录。问题非常清晰：gate 阻断了信号，惩罚量级爆炸。

## Level 2 结构变换：三个改动合一
1. **移除 health_gate**（门控→直接正向）：forward 直接用 `v_x`，让 agent 的自由探索不再被 up_thr 和 body_z 约束锁死。对应 §5 信号覆盖审计——高度安全已有 `height_penalty` 通过 hinge 惩罚保障，不需要 gate 截断梯度。
2. **upright_penalty 从间歇灾难惩罚变为连续温和引导**：改二值 hinge（up_thr=0.7）为 `(1 - up_z)` 的二次/线性形式，系数从 5.0 降为 **0.5**。保证每步都有梯度引导，且量级可控。active_rate 将从 0.4% 升到 100%，但 per-step 惩罚不再是灾难。
3. **height_penalty 保留但不作为 gate**：hinge 形式保留，系数从 10.0 降至 **1.0**。高 active_rate 时期的过量大惩罚会导致 len 暴跌，需要温和化。

## 校准验证
- 当前 forward per-step ≈ 13.11 / 11.8 ≈ 1.11（健康时近 1.0），移除 gate 后期望 per-step ≈ 0.3~0.5。
- 新 upright_penalty per-step ≤ 0.5 × (1.0)^2 = 0.5，实际 agent 倾斜时会小于 0.2。
- 新 height_penalty per-step ≤ 1.0 × 0.15^2 ≈ 0.022。
- 总惩罚负担（含 lateral 0.02, action 0.003）≤ 0.6，约为 forward_per_step 的 0.5~1.2x，但 upright 同时是引导信号，不是纯粹惩罚。**符合设计校准**。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_w   = obs[1]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]  # forward velocity
    v_y      = obs[14]  # lateral velocity

    # ---- upright projection (continuous, always gradient) ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)  # 1.0 when perfectly upright, -1.0 when inverted

    # ---- forward progress (direct, NO gate) ----
    w_fwd   = 1.0
    forward = w_fwd * v_x

    # ---- body height safety (hinge quadratic penalty, softened) ----
    z_low_safe  = 0.35
    z_high_safe = 0.85
    w_h       = 1.0  # was 10.0
    low_hinge = max(0.0, z_low_safe - body_z)
    high_hinge= max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge ** 2 + high_hinge ** 2)

    # ---- upright guidance (continuous gentle quadratic penalty) ----
    # Guides uprightness at every step without gate-killing exploration
    w_up          = 0.5  # was 5.0 + hinge
    upright_error = (1.0 - up_z)  # 0.0 when upright, 2.0 when inverted
    upright_penalty = -w_up * (upright_error ** 2)  # quadratic: gentle near upright, steep near fall

    # ---- lateral stability (quadratic penalty, unchanged) ----
    w_lat          = 0.2
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action magnitude (light energy/smoothness proxy, unchanged) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a ** 2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (forward + height_penalty + upright_penalty +
                    lateral_penalty + action_penalty)

    components = {
        "forward":          forward,
        "height_penalty":   height_penalty,
        "upright_penalty":  upright_penalty,
        "lateral_penalty":  lateral_penalty,
        "action_penalty":   action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: health_gate 截断了仅有的正向梯度，upright_penalty 在 0.4% 步中制造灾难性惩罚导致所有 episode 崩溃
- **behavior**: agent 11.8 步全部 terminate，无法学到任何有效移动，forward=+0.222 被 upright_penalty=-5.673 完全压垮
- **signal**: 缺少自由的正向梯度流，upright 引导是间歇性灾难而非连续教学信号
- **level**: Level 2
- **hypothesis**: 移除 health_gate 释放 forward 信号 + 将 upright 改为连续温和二次惩罚 → agent 能获得持续梯度引导，len 上升，forward 积累，total score 突破性改善
- **risk**: height_penalty 也降低了系数，agent 可能暂时探索更多低高度区域，但 z_low_safe=0.35 距离硬终止边界 0.2 有缓冲，应安全
