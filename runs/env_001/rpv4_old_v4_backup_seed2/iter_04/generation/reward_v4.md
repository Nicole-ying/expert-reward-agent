现在我已经掌握了所有关键信息。让我整理诊断并制定修改方案。

---

## 1. 诊断报告

### evidence

**跨轮全局视图：**

| iter | score | len | landing 状态 | 核心问题 |
|------|-------|-----|-------------|---------|
| 1 | -203.53 | 185.8 | dead (0%) | 乘积式门控太苛刻，agent 崩溃 |
| 2 | **-37.03** | 962.5 | **exploited (93.8% share, ep_sum=697)** | proximity 门控被悬停利用，agent 接近目标但不着陆 |
| 3 | -222.85 | 209.6 | **dead again (0%)** | 切回 contact 门控，梯度消失，agent 崩溃 |

**Iter 3 详细证据：**
- 全部 20/20 terminated（崩溃终止），7/20 early_terminal（<150 步）
- 四个组件中 **三个是纯惩罚**（progress=-0.299, speed_penalty=-0.735, angular_vel_penalty=-0.368），**一个 dead**（landing=0.000）
- **全局无任何正向信号**：agent 的最优策略是尽快崩溃以最小化累积惩罚
- Checkpoint 趋势：cp10%=-758 → cp70%=-218（+540 改善），但 cp70%→cp100% 在 -218~-229 之间停滞——agent 学会了减少早期崩溃，但无法突破到正向行为

**Landing 语义已走完三个来回：**
- v1: contact-gated product（死）→ v2: proximity-gated sum（exploited, 93.8%）→ v3: contact-gated sum（死）
- 在「太稀疏」和「太容易被利用」之间振荡，两轮修改均未成功

**Iter 2 (best) 的真相：**
- landing ep_sum=697, magnitude_share=93.8% —— 这是严重的 exploit，不是健康的引导
- agent 在目标附近悬停收集 landing 奖励，但 16/20 是 truncated（活满步数），从未真正着陆
- 虽然得分 -37 是最高，但它不代表正确的引导方式

### behavior_diagnosis

Agent 当前行为：**快速崩溃策略**。因为所有四个组件都是零或负贡献，总奖励恒为负，agent 学会了"尽快终止以停止损失"。它可能在训练早期尝试过接近目标，但没有任何正向梯度告诉它"你在做正确的事"。

### signal_completeness

对照环境卡片的必要职责：

| 职责 | 当前状态 | 缺失 |
|------|---------|------|
| **approach_target**（引导向目标移动） | progress 存在但为负值（agent 在远离） | 缺少正向导航梯度。progress 是结果信号（距离变化），太稀疏；需要意图信号（速度方向是否指向目标） |
| **stable_landing**（低速双脚接触） | landing 已死三回合 | 该语义已穷尽（sparse↔exploit 振荡），本轮不碰 |
| **action_penalty**（节能） | 无 | 暂不需要 |
| **time_efficiency** | 无 | 暂不需要 |

**关键缺失：正向导航信号。** agent 需要知道"朝目标移动"是正确的。progress（Δ距离）是弱结果信号，且当前为负——agent 在后退。

### selected_level

**L2**：替换一个组件的数学形态/语义。不是调系数（L1），也不是全框架重构（L3）。

具体：将 dead 组件 `landing` 替换为全新的 `velocity_guidance` 组件，语义从「着陆质量」切换为「导航方向引导」。这是从未尝试过的语义方向。

### selected_intervention

**目标组件：** 移除 `landing`，新增 `velocity_guidance`。

**具体修改：**
- 计算当前位置指向目标 (0,0) 的单位方向向量
- 计算下一步速度与该方向的余弦相似度（alignment ∈ [-1, 1]）
- 用距离做权重衰减：远处时引导强（需要导航），近处时引导弱（让 speed_penalty 和 angular_vel_penalty 接管精细控制）
- 权重适中（~0.5），避免像 iter_2 那样主导全局

这是全新的语义：**「你的速度方向对吗？」** 而非「你着陆了吗？」或「你离目标近吗？」。与 progress（结果导向）互补——guidance 提供即时梯度，progress 提供累积验证。

### falsifiable_hypothesis

**假设：** 引入正向速度方向引导后，agent 将获得指向目标的连续梯度，从而：
1. 不再崩溃逃离（有了正向激励去接近目标）
2. 逐步学会朝目标移动
3. 当距离缩小到 speed_penalty 激活范围（<1.5）时，自然会减速

**可证伪：** 如果下一轮 `velocity_guidance` 的 ep_sum 仍为负或 active_rate 异常，说明方向引导不足以克服其他惩罚。

### expected_next_round

| 指标 | 当前值 (iter_3) | 预期方向 |
|------|----------------|---------|
| score | -222.85 | 显著上升（目标 > -100） |
| len | 209.6 | 上升（不再快速崩溃） |
| terminated | 20/20 | 下降（部分 episode 能存活更久） |
| velocity_guidance ep_sum | N/A（新组件） | 正值，active_rate ~100% |
| progress ep_sum | -0.299 | 从负转正（agent 向目标移动） |
| speed_penalty active_rate | 14.8% | 上升（agent 更频繁进入近目标区域） |

### main_risk

**最大风险：velocity_guidance 与 progress 冗余。** 两者都鼓励向目标移动：progress 奖励结果（距离减小），guidance 奖励意图（速度方向正确）。如果 guidance 太强，agent 可能学会"冲向目标然后 overshoot"——因为 guidance 在 overshoot 后速度方向背离目标时变为负值，但 progress 在 overshoot 时也为负。这可能导致 agent 在目标附近震荡。缓解措施：距离衰减权重（近处 guidance 减弱，让 speed_penalty 主导减速）。

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

    # 4. 速度方向引导（新组件，替代 landing）
    #    奖励「速度方向指向目标」，提供即时正向导航梯度
    #    距离衰减：远处需要导航引导，近处让减速信号接管
    if old_dist > 0.05:
        # 单位方向向量：从当前位置指向目标 (0,0)
        dir_x = -x / old_dist
        dir_y = -y / old_dist

        # 速度方向与目标方向的余弦相似度
        spd = (next_vx ** 2 + next_vy ** 2) ** 0.5 + 0.01
        alignment = (next_vx * dir_x + next_vy * dir_y) / spd  # [-1, 1]

        # 距离衰减权重：远处=1.0，距离0→0（线性衰减，5单位外满权重）
        distance_weight = min(1.0, old_dist / 5.0)

        w_guidance = 0.5
        guidance_reward = w_guidance * distance_weight * alignment
    else:
        guidance_reward = 0.0

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + guidance_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "velocity_guidance": guidance_reward
    }
    return float(total_reward), components
```