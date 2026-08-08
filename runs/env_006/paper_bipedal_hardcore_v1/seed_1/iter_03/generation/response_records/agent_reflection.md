# Response Record

# 设计理由

本轮改一个组件：将原先的 **加性稳定性惩罚** 改为 **乘性健康门控（soft_health_gate）**。  
证据：训练反馈显示 agent 全部 20 条 episode 以终止（推断为摔倒）结束，episode 长度从 411 骤降至 222，且 stability_penalty 的 active_rate 仅 15.8%，说明在摔倒前的“临界不稳定期”惩罚出现太晚、力度太弱；同时 angular_velocity_penalty 近乎零贡献，没有起到早期拦截作用。因此需要一种**更前置、更强耦合**的稳定性信号。

数学形式：  
- 原方案：`reward -= 0.5 * max(0, |angle| - 0.3)` —— 独立加性项，在失调初期数值微小，无法有效抑制 agent 在危险姿态下继续前进。  
- 新方案：`gate = 1.0 - 2.0 * max(0, |angle| - 0.15)`，乘到 progress_reward 上。将前进奖励与稳定程度强绑定：姿态偏离一旦超过 0.15 rad，前进奖励就开始线性衰减；到 0.65 rad 时彻底归零。这迫使 agent 在学会快速行走前必须先学会维持躯干直立，从而延长存活时间，提高总体得分。

系数校准：  
- 门控阈值下移至 0.15 rad，在仍属“安全但不够完美”的区域（0.2 rad）gate≈0.9，满足“不理想但安全区域 gate≥0.3”的要求；  
- 整体惩罚负担（progress 的损失）已隐含在奖励中，不再有额外的加性惩罚，满足总惩罚负担 ≤ 0.5× 主信号 per-step 的约束。  

angular_velocity_penalty 与 action_efficiency_penalty 保持不变，避免一次改动过多变量。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测提取
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    horizontal_speed = obs[2]

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

    total_reward = progress_reward + angular_velocity_penalty + action_efficiency_penalty
    components = {
        'progress_reward': progress_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_efficiency_penalty': action_efficiency_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号基本齐全但校准失误——稳定性惩罚触发过晚、力度过小，且角速度惩罚几无影响；缺失对“摔倒前急剧恶化”的有效前置拦截。
- **behavior**: agent 以 100% 终止（摔倒）结束，平均存活仅 222 步，姿态控制极弱。
- **signal**: 原稳定性惩罚（hinge, threshold 0.3）在临界不稳定期基本不激活，无法阻止 agent 在危险姿态下仍追求前进速度。
- **level**: Level 2 — 结构变换（加性约束 → 乘性门控）。
- **hypothesis**: 把稳定性信号从前置松散惩罚改为强耦合的进度缩放门，会迫使 agent 在学到高速行走前先学会保持躯干稳定，从而延长 episode 长度并提升真实得分。
- **risk**: 门控过早介入可能使 agent 过于保守，导致 horizontal_speed 下降，短期内平均奖励可能仍偏低；但存活时间提升应能扭转整体得分。
