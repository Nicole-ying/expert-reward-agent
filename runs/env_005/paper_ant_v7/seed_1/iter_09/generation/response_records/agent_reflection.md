# Response Record

# 设计理由

## 0. 信号覆盖审计
- **终止 → 前兆**：环境终止条件为身体高度超出 (0.2, 1.0)（跌倒/失控）。当前奖励数的 `height_penalty` 阈值设在 0.25~0.95，active_rate 仅 1.8%，说明绝大部分步数 body_z 处在 “安全幻觉” 区（0.25~0.95），但 episode 仍快速终止（len=138, terminated=20/20）。这意味着 agent 从安全区到死亡 <0.2 的滑落极快，现有的事后二次惩罚来不及提供有效梯度 → **缺少接近危险边界的早期预警信号**。
- **目标 → 进度**：主目标前进速度已有 `forward_speed_reward`（active 100%），但没有与安全姿态耦合 → agent 可以不顾姿态全力前冲然后跌倒。
- **效率信号**：动作维度 8 ≥ 6，当前无 action penalty，但历史中 action penalty 未能改观，暂不作为第一优先级。
- **僵尸组件**：`height_penalty` active_rate=1.8%，实质上未参与学习 → 应当被更有效的机制替代。
- **一句话结论**：当前 reward 漏掉了将前进速度奖励与持续健康姿态强耦合的门控机制，导致 agent 为刷前进速度牺牲稳定性、快速跌倒。

## 1. 行为诊断
- **agent 在做什么**：以高速前向运动（`forward_speed_reward` 均值 207.6）但牺牲直立和高度稳定性，迅速跌倒终止（episode 均值仅 138 步）。
- **干预哪个目标**：将前进速度奖励与健康姿态（身体高度 + 直立度）进行强挂钩，使 agent 必须同时保持稳定才能获得前进信号。
- **方向是否值得继续**：累积记录中多次尝试独立约束（action penalty、横向惩罚、高度二次惩罚）均未解决快速跌倒，且 len 从此前 400~740 暴跌至 138，说明单纯“惩罚”无效。需要 Level 2 结构变换：将约束从独立惩罚改为**门控主奖励**。

## 2. 干预层级：Level 2 — 结构变换
**证据**：`height_penalty` active<2%，形态塌缩；agent 快速死亡，独立惩罚未能阻止。  
**变换**：移除失效的独立高度/直立惩罚，将前进速度奖励乘以健康门 `gate = height_factor * upright_factor`，其中：
- `height_factor`：body_z 从 0.35（安全）到 0.2（死亡）线性衰减到 0，提供提前 75% 终止边界的早期预警。
- `upright_factor`：body_up_z 从 0.9 到 0.7 线性衰减，防止严重前倾但允许合理的前倾以产生速度。
- 乘积形式确保任一维度严重恶化时主奖励快速归零，驱动 agent 恢复姿态。

**系数校准**：
- 原每步 forward 约 1.49；门控后预期健康步均因子约 0.5~0.7，有效奖励下降但可接受，不缩放主系数。
- 不再有独立高度/直立惩罚，避免惩罚负担叠加。依据设计准则 #5，惩罚总负担 ≈ 0，远低于主信号。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆解
    body_z = obs[0]
    quat_w, quat_x, quat_y, quat_z = obs[1], obs[2], obs[3], obs[4]
    body_x_velocity = obs[13]

    # 计算躯干对世界 z 轴的投影（直立度，1 为完全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 健康门控因子 ----------
    # 高度因子：body_z 安全区 [0.35, +∞)，危险区 (0.2, 0.35)
    height_factor = max(0.0, (body_z - 0.2) / 0.15)      # 0.35→1, 0.2→0
    height_factor = min(1.0, height_factor)              # 裁剪 >1 的情况

    # 直立因子：body_up_z 理想 ≥0.9，宽容至 0.7 开始惩罚
    upright_factor = max(0.0, (body_up_z - 0.7) / 0.2)  # 0.9→1, 0.7→0
    upright_factor = min(1.0, upright_factor)

    # 综合门控：乘积形式，任一因子恶化到 0 直接切断主奖励
    health_gate = height_factor * upright_factor

    # ---------- 主学习信号：前向速度奖励（凸化形式，门控） ----------
    forward_speed_reward = 0.5 * (body_x_velocity ** 2)
    gated_forward_reward = forward_speed_reward * health_gate

    # ---------- 组件输出（旧独立惩罚归零，保留字段供监控） ----------
    total_reward = gated_forward_reward

    components = {
        "forward_speed_reward": gated_forward_reward,   # 门控后的有效前进奖励
        "height_factor": height_factor,                 # 高度门控因子（监控用）
        "upright_factor": upright_factor,               # 直立门控因子（监控用）
        "height_penalty": 0.0,                          # 已被门控替代
        "upright_penalty": 0.0                          # 已被门控替代
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少高度持续恶化的早期预警信号，导致 agent 可以在全身高速状态下瞬间跌倒而无梯度矫正。
- **behavior**: agent 为追求高前进速度而牺牲稳定性，平均 138 步即因高度越界终止。
- **signal**: 缺了将前进奖励与持续健康姿态耦合的门控，原独立高度/直立惩罚已塌缩且无效。
- **level**: Level 2
- **hypothesis**: 将主奖励与健康门控强挂钩后，agent 若想获得前进速度就必须主动维持身体高度和直立度，从而延长存活并提升外部得分。
- **risk**: 门控可能抑制早期探索时的偶然高速步态，导致前进速度提升变慢；但鉴于当前快速死亡的根本问题，存活率优先。
