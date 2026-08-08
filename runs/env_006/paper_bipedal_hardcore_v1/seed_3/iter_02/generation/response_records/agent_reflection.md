# Response Record

# 设计理由

## 审计结论
**终止模式**：训练反馈显示 20/20 episode 均为 terminated（非 truncated），平均 episode length 253 step。结合环境 §5，terminated 只有两种可能：摔倒 (body_fallen_over) 或到达终点 (reached_end_of_terrain)。给定 score = -61.55 且 len = 253，agent 不可能是到达终点（那会给出更高的正向奖励），**极大概率是反复摔倒**。len=253 表明 agent 在摔倒前还能徘徊一段时间，不是瞬间崩溃。

**信号覆盖**：当前代码使用了 obs[0-3] 四个信号，覆盖了身体倾角、角速度、前进速度、垂直速度。但未使用的观测中：
- **obs[12] leg_1_ground_contact 和 obs[13] leg_2_ground_contact**：双脚接地信号，可用于判断"无立足"状态——双腿同时离地是摔倒前兆。当前完全未使用。
- **obs[4-11] 关节角度/角速度**：可用于惩罚过度伸展或非自然步态，但属于次级需求。
- **obs[14:23] 雷达**：可检测前方障碍，但环境文档明确警告"不建议直接用作奖励信号"。

**僵尸组件**：`posture_penalty` 的 active_rate 仅 **3.9%** — 该组件设计意图（惩罚大倾角）几乎未实现。原因是阈值 `angle_threshold = 0.8` 设得过高，agent 在大多数 step 倾角低于此阈值，即使已经处于不稳定状态。

**信号缺口判断**：当前核心问题是 **信号校准问题**，而非信号缺失。倾角惩罚的 hinge 阈值过高导致它形同虚设，agent 没有收到足够的稳定性约束。同时 vertical_penalty active_rate=100% 但 per-step 贡献仅 -0.002，是持续性微弱的拖累。

## 行为诊断
agent 在以 **中速前进但缺乏稳定性约束的方式运行**。progress_reward 推动 agent 前进，但 posture_penalty 的阈值 0.8 远高于实际摔倒边界（~1.0-1.2 rad），导致在倾角 0.3-0.7 的"危险但不触发惩罚"区域，agent 无法区分安全和危险状态。**agent 最终摔倒是因为缺乏渐进的姿态反馈来纠偏。**

## 干预选择
**干预目标**：稳定性约束（posture_penalty 组件的重构）。
**干预层级**：Level 2 — 结构变换。posture_penalty 的 hinge + 高阈值设计导致 active_rate 仅 3.9%，应该改为 **连续的 soft health gate** 乘到 progress_reward 上。同时，vertical_penalty 作为 active_rate=100% 的常驻惩罚，其系数需要降低以匹配设计校准规则。

## 具体修改
**将 posture_penalty 的独立加法 hinge 惩罚改为 posture_gate 乘法门控**：用 `1.0 / (1.0 + 5.0 * abs(hull_angle))` 生成一个在倾角=0 时为 1.0、倾角=0.8 时约为 0.2 的连续衰减因子，乘到 progress_reward 上。这比独立惩罚更有效地塑造行为：当 agent 倾斜时前进奖励自动衰减，agent 必须恢复直立才能获得完整的前进激励。

**降低 vertical_penalty 系数**：从 0.15 降至 0.05，因为其 active_rate=100% 且 per-step 贡献虽小但持续拖累总分。新系数使其 per-step ≈ progress_reward per-step 的 0.1x 以内。

## 系数校准
- progress_reward per-step ≈ 24.03 / 253 ≈ 0.095
- posture_gate 在倾角=0 时不衰减，在倾角=0.5 时 gate≈0.29（progress_reward 降至 0.027），在倾角=0.8 时 gate≈0.2（progress_reward 降至 0.019）
- vertical_penalty per-step 预期 ≈ -0.05 × E[vz²]，从之前 active_rate 100% 但 episode_sum=-0.48 推算 E[vz²]≈0.013，新 per-step ≈ -0.0006，远小于 progress_reward per-step

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- 主学习信号：水平前进速度 × 姿态门控 -----
    horizontal_speed = obs[2]
    hull_angle = obs[0]

    # posture_gate: 连续衰减因子，倾角越大前进奖励越小
    # 倾角=0 → gate≈1.0; 倾角=0.5 → gate≈0.29; 倾角=0.8 → gate≈0.20
    posture_gate = 1.0 / (1.0 + 5.0 * abs(hull_angle))
    progress_reward = 0.3 * horizontal_speed * posture_gate

    # ----- 稳定/安全约束 -----
    # 1. 身体角速度惩罚 (quadratic，持续抑制旋转)
    hull_angular_vel = obs[1]
    angular_penalty = -0.06 * (hull_angular_vel ** 2)

    # 2. 垂直速度异常惩罚 (quadratic，弱抑制弹跳，系数降低至原 1/3)
    vertical_speed = obs[3]
    vertical_penalty = -0.05 * (vertical_speed ** 2)

    # ----- 汇总奖励 -----
    total_reward = progress_reward + angular_penalty + vertical_penalty

    components = {
        "progress_reward": progress_reward,
        "posture_gate": posture_gate,
        "angular_penalty": angular_penalty,
        "vertical_penalty": vertical_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: posture_penalty active_rate=3.9% 因 hinge 阈值过高形同虚设，agent 缺乏渐进姿态反馈；未使用的接触信号可后续引入但非当前根因。
- **behavior**: agent 中速前进但最终摔倒，因为在倾角 0.3-0.7 的危险区域无区分性反馈。
- **signal**: 姿态约束信号存在（hull_angle 已使用）但数学形式失败——hinge 阈值 0.8 远高于实际需要纠偏的区间，应改为连续门控。
- **level**: Level 2
- **hypothesis**: posture_gate 乘到 progress_reward 上将迫使 agent 在倾斜时自动减速/纠姿以获得完整前进激励，从而降低摔倒率、提高有效前进距离。
- **risk**: 如果 posture_gate 衰减过强（在 0.3-0.5 倾角就显著衰减），agent 可能变得过于保守，牺牲前进速度换取低倾角。需在下一轮观察 len 和 terminated/truncated 分布来判断是否矫枉过正。
