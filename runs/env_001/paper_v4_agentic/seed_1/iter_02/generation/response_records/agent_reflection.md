# Response Record

# 设计理由
这是第一轮反思。当前奖励函数的主体是 `progress`（距离缩小给予正向奖励）和 `landing_proxy`（接近、双腿接触、低速的综合代理）。但从训练反馈看，`landing_proxy` 的 episode 总奖励贡献高达 98.5%，步均贡献远超 `progress`，并且 active_rate 高达 58.3%，说明 agent 在大量步数内保持双腿接触并持续获得这一奖励。结合 episode 长度接近 600 步（多为 truncated），agent 可能学会了一种 reward hacking 行为：着陆后在垫子上低速移动（仍保持双腿接触、低速），以此源源不断获取 `landing_proxy` 奖励，而不是尽快进入 `body_not_awake_or_settled` 终止（即尽快稳定停靠）。这违背了“尽快着陆并稳定”的任务目标。

因此，对 `landing_proxy` 组件进行 **Level 1 尺度修复**：将其权重 `w_proxy` 从 0.3 降低至 0.05，减少其每步奖励的支配性，让 `progress` 在整体奖励中的相对重要性回升，抑制 agent 在着陆后无谓停留的动机。

# 代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ================== main progress：减少到目标垫的欧氏距离 ==================
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_old - dist_new                    # 距离缩小为正奖励
    w_progress = 1.0

    # ================== 姿态/角速度稳定约束（hinge 软惩罚）===================
    body_angle = abs(obs[4])
    ang_vel = abs(obs[5]) if abs(obs[5]) > 1e-6 else 0.0

    angle_penalty = max(0.0, body_angle - 0.3)        # 角度超过0.3 rad 才惩罚
    ang_vel_penalty = max(0.0, ang_vel - 1.0)         # 角速度超过1.0 rad/s 才惩罚

    w_angle = 0.05
    w_ang_vel = 0.02

    # ================== 登陆完成软代理（joint_condition_proxy）=================
    # 用 next_obs 判断着陆条件
    proximity = 1.0 / (1.0 + 5.0 * dist_new)           # 越近越接近1
    leg_contact = next_obs[6] * next_obs[7]            # 双腿都接触=1.0，否则0
    speed = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * speed)           # 低速接近1

    # 几何平均缓解乘积塌缩
    proxy = (proximity * leg_contact * speed_factor) ** (1.0/3.0)
    w_proxy = 0.05   # <--- 从 0.3 降低至 0.05

    # ================== 总奖励 ==================
    total_reward = (w_progress * progress
                    - w_angle * angle_penalty
                    - w_ang_vel * ang_vel_penalty
                    + w_proxy * proxy)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": -w_angle * angle_penalty,
        "ang_vel_penalty": -w_ang_vel * ang_vel_penalty,
        "landing_proxy": w_proxy * proxy
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度均被使用，信号覆盖齐全；但 `landing_proxy` 步均奖励过高，导致 agent 在着陆后滞留以持续获取奖励。
- **behavior**: agent 可能在着陆后低速移动、保持双腿接触以积累 `landing_proxy` 奖励，而非尽快稳定停靠。
- **signal**: `landing_proxy` 强度（per‑step）过大，压倒了 `progress`，且无任何“尽快结束”的压力信号。
- **level**: Level 1
- **hypothesis**: 降低 `landing_proxy` 权重后，agent 将更注重中距离的快速接近，着陆后停留获得的奖励显著减少，从而倾向于更快触发 `body_not_awake_or_settled` 终止，缩短 episode 长度并改善整体效率。
- **risk**: 权重下降可能暂时减慢早期训练时的着陆学习速度，但 `progress` 和姿态惩罚仍能提供足够的引导；已超出目标分数较多，性能回落至目标附近仍属可接受。
