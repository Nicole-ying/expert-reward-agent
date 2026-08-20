# Response Record

Now I have all the evidence I need. Let me compile the diagnosis and produce the fix.

---

## 1. 诊断报告

### evidence

**跨轮全局视图：**

| iter | score | len | 关键信号 | 核心问题 |
|------|-------|-----|---------|---------|
| 1 | -203.53 | 185.8 | landing dead (0%) | 乘积式门控太苛刻 |
| 2 | **-37.03** | 962.5 | landing 93.8% share, ep_sum=697 | **proximity 门控被 exploit**：agent 悬停 962 步收集奖励不着陆 |
| 3 | -222.85 | 209.6 | landing dead again (0%) | contact-gated 太稀疏，信号消失 |
| 4 | -295.33 | 138.9 | velocity_guidance 100% active, **始终为负** | 方向引导只惩罚不对齐，无正向梯度；11/20 early_terminal |

**Iter 4 深度诊断：**

- `velocity_guidance`：ep_sum=-1.596，**100% active**，**signed_share=-18.9%**，**magnitude_share=53.9%**。该组件始终输出负值——agent 从未因速度方向正确而获得正奖励。这是一个纯惩罚信号，不提供"什么是对的"的梯度。
- `progress`：ep_sum=-0.026，几乎为零。agent 在水平方向上没有净进展。
- `speed_penalty`：active_rate 从 iter 3 的 14.8% 跳升至 63.2%，ep_sum 从 -0.735 恶化到 -2.097。agent 更多时间处于近目标速度惩罚区但未学会减速。
- **Checkpoint 趋势**：velocity_guidance 从 cp10%=-15.862 改善至 cp100%=-1.395，说明 agent 确实在学习对齐，但改善速度不足以在 100% 训练完成前达到正反馈区域。
- **early_terminal 11/20**：agent 频繁 crash（可能是高速撞击或飞出视口）。

**根本原因**：`velocity_guidance` 是一个"只罚不奖"的方向信号——`alignment ∈ [-1, 1]`，agent 大部分时间 alignment < 0 被惩罚。离散推力（4 个离散动作）使精确速度方向控制极其困难，agent 无法稳定获得正反馈。同时 landing 信号完全缺失，agent 在接近目标后没有任何着陆引导。

### behavior_diagnosis

Agent 在 iter 4 的行为：在空间中移动但无法有效导航至目标（progress≈0），velocity_guidance 持续惩罚速度方向不对齐。agent 在 138.9 步内 crash（11/20 early_terminal），说明它可能在目标区域附近高速撞击或飞出视口。**关键缺失：接近目标后的着陆/减速引导信号。**

### signal_completeness

- `approach_target`（接近目标）：progress 提供距离变化梯度，但信号很弱（ep_sum≈-0.026）。velocity_guidance 试图提供方向引导但失败。
- `stable_landing`（稳定着陆）：**完全缺失**。iter 2 证明 proximity-gated landing 能有效引导 agent 接近目标（progress=+0.665），但设计缺陷导致 exploit。iter 3 证明纯 contact-gated 太稀疏。iter 4 完全放弃 landing。
- `action_penalty`（燃料节省）：缺失，但当前不是主要矛盾。

**缺失信号：一个既不被 exploit、又能被发现的着陆引导信号。**

### selected_level

**L2** — 替换一个组件的数学形态。将 `velocity_guidance`（方向对齐）替换为 `landing`（proximity-gated + contact-amplified 着陆引导）。这不是调参（L1），也不是换整个框架（L3）。这是将"方向引导"的语义改为"目标接近+着陆质量引导"。

### selected_intervention

**移除 `velocity_guidance`，新增 `landing` 组件**：
- **门控**：Manhattan 距离 < 1.0（比 iter 2 的 1.5 更紧，防止远距离悬停 exploit）
- **无接触时**：proximity × quality_factors 提供接近目标的连续梯度
- **有接触时**：contact_mult = 1 + 4×contact（双脚接触时 5× 放大），强力激励着陆
- **权重**：w_land=0.5（远低于 iter 2 的 3.0，防止 exploit）
- **保留** progress、speed_penalty、angular_vel_penalty 不变

### falsifiable_hypothesis

proximity-gated landing 在 iter 2 中成功引导 agent 接近目标（progress=+0.665），只是权重过大 + 无接触要求导致 exploit。本设计通过：(a) 收紧 proximity 阈值（1.0 vs 1.5），(b) 大幅降低权重（0.5 vs 3.0），(c) contact 5× 放大提供着陆激励——应保留 iter 2 的导航引导能力，同时消除悬停 exploit。**预期：landing 组件 active_rate > 0%（可发现），且 agent 在接近目标后因 contact 放大而试图着陆，而非悬停。**

### expected_next_round

- `landing` active_rate > 5%（比 iter 3 的 0% 显著改善）
- `landing` ep_sum 为正值且 magnitude_share < 40%
- `progress` ep_sum 趋向正值（agent 被吸引向目标）
- `score` 应改善至 > -200（至少优于 iter 3 的 -222.85）
- `len` 可能增加（减少 early crash）
- `speed_penalty` active_rate 可能下降（agent 在接近目标时学会减速）

### main_risk

landing 组件可能再次被 exploit——如果 proximity 阈值 1.0 仍允许 agent 在目标附近徘徊而不着陆。缓解措施：权重 0.5 使单步最大无接触奖励仅 0.5，远低于 iter 2 的 3.0。如果下一轮 active_rate 高但 contact 始终为零且 ep_sum 异常大，则需要进一步收紧门控或添加基于 episode 长度的衰减。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：proximity门控 + contact放大
    #    设计要点：
    #    - proximity 门控保证可发现性（进入目标附近即有信号）
    #    - contact 放大提供着陆激励（双脚接触时 5x 奖励）
    #    - 低权重防止悬停 exploit（无接触时 max=0.5/步，远低于 iter2 的 3.0/步）
    proximity = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 1.0)  # Manhattan < 1.0

    if proximity > 0.0:
        # 着陆质量因子（每个在 [0,1]）
        pos_quality = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 0.3)
        vel_quality = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 0.5)
        angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.3)

        # 接触信号：0 / 0.5 / 1.0
        contact = (next_left_contact + next_right_contact) / 2.0

        # contact 放大倍数：无接触=1x，单脚=3x，双脚=5x
        contact_mult = 1.0 + 4.0 * contact

        # 最终着陆分数：proximity × contact_mult × 平均质量
        landing_score = proximity * contact_mult * (
            pos_quality + vel_quality + angle_quality
        ) / 3.0
    else:
        landing_score = 0.0

    w_land = 0.5
    landing_reward = w_land * landing_score

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```
