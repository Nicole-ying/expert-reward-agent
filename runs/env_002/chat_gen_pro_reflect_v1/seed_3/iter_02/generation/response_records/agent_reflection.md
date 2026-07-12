# Response Record

# 设计理由

**audit 结论**：  
训练奖励函数中 `alternating_bonus` 作为独立正奖励组件，每步约贡献 0.09，远超 `forward_progress_reward` 的 0.005。机器人学会原地交替摆腿即可获得足够奖励，无需前进。这是导致 score 为负（环境可能还会施加其他生存惩罚）的根本原因——前进信号太弱，被步态奖励掩盖。

**行为诊断**：  
Agent 并未摔倒（terminated=0），但始终在原地踏步（truncated=20/20, len=1600），因为单纯保持两腿交替接触就能获得大量正向反馈，缺乏动机产生实际前进速度。

**信号缺口**：  
强前进信号缺失。当前 `forward_progress_reward` 用 `horizontal_vel**2` 且未过滤负速度，量级极低，且允许后退获得正奖励。必须将前进激励放大，并建立“步态质量只为前进服务”的依赖关系。

**干预层级**：**Level 2 – 结构变换**  
将 `alternating_bonus` 从独立的累加组件改为乘性门控因子，乘到前进奖励上，使原地踏步的行为不再独立获得奖励，同时基础前进奖励不受完全抑制。仅修改一个组件（交替 ↔ 前进的交互方式），保留 `balance_penalty` 和 `both_off_ground_penalty` 的职责。

**数学形式**：  
- 前进主信号：`forward_progress = 2.0 * max(0, horizontal_vel) ** 2`  
  采用正速度凸化，系数从 1.0 提升至 2.0 以增强主梯度。
- 步态门控：`gait_quality = 1.0 - abs(leg1_contact - leg2_contact)`（值域 [0,1]，1 为完美交替）
- 总前进奖励：`forward_with_gait = forward_progress * (0.5 + 0.5 * gait_quality)`
  使每一步至少有 0.5 倍基础前进奖励，交替越好奖励越丰厚，但原地踏步无前进则贡献为零。

**系数校准**：  
- `forward_progress` per‑step 预估值（速度 0.5 → 0.5）约 0.5，按 0.75 平均门控 = 0.375，远大于原来 forward 的 0.005，有望成为主导信号。
- `balance_penalty` 与 `both_off_ground_penalty` 的总惩罚负担较低，主奖励应能轻松覆盖。

**风险**：如果机器人一开始无法产生明显速度，前进奖励可能长期接近零，学习困难。但 `gate` 的 0.5 基底保留基础梯度，避免信号完全消失。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号（只奖励正向水平速度）
    #    w_up=2.0，凸化平方，拒绝负速度
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress = 2.0 * (max(0.0, horizontal_vel) ** 2)

    # ============================================================
    # 2. gait_quality gate: 步态交替质量作为乘性门控
    #    使用 leg1_contact，leg2_contact 计算交替程度
    #    值域 [0,1]，1 表示两脚接触状态完全相反
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)

    # 将步态质量作为 0.5~1.0 的乘性因子乘到前进奖励上
    forward_with_gait = forward_progress * (0.5 + 0.5 * gait_quality)

    # ============================================================
    # 3. balance_maintenance: 防摔倒软约束（保持原样）
    #    轻微惩罚倾斜角与角速度
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    balance_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_vel ** 2)

    # ============================================================
    # 4. both_off_ground_penalty: 防止跳跃/双腿腾空（保持原样）
    # ============================================================
    contact_sum = leg1_contact + leg2_contact
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0

    # 组合总奖励（不再包含独立的交替奖励）
    total_reward = forward_with_gait + balance_penalty + both_off_ground_penalty

    components = {
        "forward_with_gait": forward_with_gait,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "gait_quality": gait_quality,           # 纯诊断，不影响奖励
        "forward_progress_raw": forward_progress # 纯诊断
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 交替奖励喧宾夺主，独立存在导致原地踏步刷分，前进信号微弱且可奖励后退。
- **behavior**: 原地交替摆腿存活至最大步数，无摔倒，无前进。
- **signal**: 缺乏足够量级的正向速度激励，交替与前进解耦。
- **level**: Level 2
- **hypothesis**: 将交替信号转为前进的门控乘性因子，消除原地踏步独立收益，迫使机器人通过实际前进获取奖励，同时保持平衡约束。
- **risk**: 早期探索可能难以产生正速度，前进奖励近似为零，导致学习变慢；但 0.5 基底门控保留基本梯度，可缓解。
