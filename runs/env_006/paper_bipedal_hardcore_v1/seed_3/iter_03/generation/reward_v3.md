# 设计理由

## 审计发现（信号覆盖审计）
- **终止模式**：95% 的 episode 以 `terminated` 结束（推测为摔倒），平均存活 303 步，说明 agent 能移动一段距离但最终失控。无成功到达终点的迹象，因为 score 为负且无正向完成信号。
- **观测使用扫描**：当前仅使用了 `obs[0:4]`（倾角、角速度、水平/垂直速度），**完全未使用** 足部接触传感器 `obs[12]`、`obs[13]`。这是一项关键缺失——双足机器人的稳定性与足部接触状态强相关，双脚同时离地（“腾空”）往往是跳跃或失衡摔倒的前兆。
- **信号缺口**：存在明确的信号缺失。添加基于 `obs[12]/[13]` 的惩罚组件可以直接填补这一缺口，在摔倒发生前给予抑制信号。

## 诊断与修改决策
- **行为诊断**：agent 学会了一种在崎岖地形上勉强前进的策略（水平速度平均 ≈0.28），但缺乏稳定性，频繁摔倒。现有惩罚（角速度、垂直速度）系数极弱（episode_sum_mean 约 -0.02 和 -0.11），几乎形同虚设，无法抑制不稳定行为。
- **累积记录分析**：前两轮使用相同的骨架（前进速度 + 姿态门控 + 两个弱惩罚），len 略有增加但 score 几乎不变且极低（-61.5）。第二轮引入的 `posture_gate` 乘到进度奖励并未改善摔倒问题，说明仅靠姿态门控不足以教会稳健步态。
- **修改层级选择**：Level 2 结构变换，**新增一个组件**：足部接触浮空惩罚 `air_penalty`。
- **数学形式**：当 `obs[12]`（leg_1_ground_contact）与 `obs[13]`（leg_2_ground_contact）均为 0 时，施加负奖励 `-0.015`。系数设定依据：
  - 主信号每步均值 ≈ 0.056，惩罚每步负担 ≤ 0.3× 主信号 → ≤0.0168。
  - 取 `0.015`，在浮空期间施加适中压力，不破坏正常单脚支撑阶段的探索。
- **副作用检查**：该惩罚仅在双脚完全离地时激活，正常行走、单脚支撑均不受影响，不会塌缩主奖励。预计 active_rate 不高，总惩罚负担可控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- 主学习信号：水平前进速度 × 姿态门控 -----
    horizontal_speed = obs[2]
    hull_angle = obs[0]

    # posture_gate: 连续衰减因子，倾角越大前进奖励越小
    posture_gate = 1.0 / (1.0 + 5.0 * abs(hull_angle))
    progress_reward = 0.3 * horizontal_speed * posture_gate

    # ----- 稳定/安全约束 -----
    # 1. 身体角速度惩罚 (quadratic，持续抑制旋转)
    hull_angular_vel = obs[1]
    angular_penalty = -0.06 * (hull_angular_vel ** 2)

    # 2. 垂直速度异常惩罚 (quadratic，弱抑制弹跳)
    vertical_speed = obs[3]
    vertical_penalty = -0.05 * (vertical_speed ** 2)

    # 3. 足部浮空惩罚：双脚同时离地常伴随跳跃或即将摔倒
    leg1_contact = obs[12]
    leg2_contact = obs[13]
    both_airborne = 1.0 if (leg1_contact + leg2_contact == 0.0) else 0.0
    air_penalty = -0.015 * both_airborne

    # ----- 汇总奖励 -----
    total_reward = progress_reward + angular_penalty + vertical_penalty + air_penalty

    components = {
        "progress_reward": progress_reward,
        "posture_gate": posture_gate,
        "angular_penalty": angular_penalty,
        "vertical_penalty": vertical_penalty,
        "air_penalty": air_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 足部接触传感器完全未使用，双脚同时离地（跳跃/失衡前兆）无惩罚，信号缺失直接关联摔倒高发。
- **behavior**: agent 能低效前进（均速 ~0.28），但频繁摔倒导致 terminated（19/20）。
- **signal**: 缺少抑制不稳定腾空阶段的惩罚，现有角速度/垂直速度惩罚过弱无法引导稳定性。
- **level**: Level 2
- **hypothesis**: 引入空中惩罚将迫使 agent 减少双脚同时离地的危险行为，提高落足稳定性，从而降低摔倒率，延长存活时间并为正确步态学习创造空间。
- **risk**: 若浮空也是正常跑步的必要组成部分（如高速双足跑），过强惩罚可能抑制速度上限；但当前双足崎岖地形任务中，跑步极少发生，风险可控。