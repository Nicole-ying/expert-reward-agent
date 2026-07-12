# Response Record

# 设计理由

## 信号覆盖审计
- **终止→前兆**：§5 声明终止条件为高度越界（<0.2 或 >1.0）和 NaN/inf。代码中已通过 `height_factor` 门控提供软信号，但这个门控设计有缺陷：它在安全区线性降权而非鼓励探索，导致 agent 被迫在高度门控下行走，无法自然学习步态。
- **目标→进度**：§1 声明核心目标是向前行走。代码有 `forward_reward = body_x_velocity * height_factor` 提供直接前进梯度，但乘性门控使信号异化——agent 不是在学习"走得好"，而是在学习"在门控允许的范围内走"，导致 survival 被门控扭曲而非被前进目标驱动。
- **效率信号**：8 维连续动作空间 `action_cost = -0.01 * torque_penalty`，系数过小，对步态质量无实质约束。
- **僵尸组件**：`upright_penalty` 的 `active_rate = 99.7%`，每步都在惩罚姿态，但 `terminated = 11/20` 表明姿态问题仍未解决。这是一个**失效的惩罚**——它不是保护性约束，而是无差别的负奖励流，每步 -1.04（从 episode_sum_mean / len 可得），占主导地位。
- **一句话结论**：当前 reward 缺少有效的姿态约束（现有惩罚每步都触发但无效），且高度门控扭曲了前进信号。

## 行为诊断

- **agent 在做什么**？慢速徘徊。len=731.50 表明它能存活，但姿态崩塌（upright_penalty 每步 -1.04）。agent 为了最小化 upright_penalty 而缓慢移动、减少动作幅度，但这反而让姿态更差——因为四足机器人必须有足够的力矩驱动才能维持动态稳定。二次惩罚压制了所有大动作，导致 agent 无法学习有效步态，陷入低力矩-低速度-差姿态的恶性循环。

- **干预什么目标**？修复姿态约束。这是核心瓶颈：当前惩罚在正常步态中就已经很高（每步 -1.04），而前进奖励只有每步 +0.0031，惩罚比奖励强 335 倍。agent 的主要学习信号是"不要动以减少惩罚"，而不是"前进"。

- **这个方向还值得继续吗**？连续 2 轮预判 ❌，但主骨架（前进速度 + 姿态惩罚 + 力矩效率）在四足任务中是经典有效框架。问题不在于骨架，而在于 upright_penalty 的**数学形态**：二次惩罚在所有状态都惩罚，形成了"抑制一切动作"的误导向。改为 hinge 形式可以只在姿态超出安全范围时介入。

## 选择干预层级：Level 2 — 结构变换

**证据**：`upright_penalty` active_rate = 99.7%，但 terminated 率仍高 → **全时二次惩罚 → hinge**

### 变换要点
- **当前形态**：`-2.0 * (1.0 - body_up_z)**2`，body_up_z 在 0.9 时（轻微倾斜）惩罚 = -0.02，但 body_up_z 在 0.7 时惩罚 = -0.18。这个惩罚在正常行走姿态中就已经很高。
- **目标形态**：`hinge(0.7 - body_up_z)`，当 body_up_z > 0.7 时不惩罚，< 0.7 时线性惩罚。0.7 对应约 45° 倾斜，是终止前兆的 60-80% 处。
- **系数校准**：目标 per-step ≤ forward_reward per-step 的 0.3x。forward_reward per-step ≈ 2258.69 / 731.50 ≈ 3.09 → 目标 upright per-step ≤ 0.93。hinge 系数设为 -1.0，当 body_up_z = 0.6（接近跌倒）时 penalty ≈ -0.1，在典型步态中（body_up_z ≈ 0.9）penalty = 0。
- **同时调整 action_cost**：提升到 -0.05，从 magnitude_share=1.0% 提升到约 5%，鼓励更高效的步态。

### 设计校准检查
1. **新惩罚系数**：upright_penalty 在 hinge 下，正常步态中 body_up_z ≈ 0.8-0.9，penalty = 0；只有当姿态恶化时才激活 → 目标 per-step 约 0.03（间歇性激活），远小于 forward_reward 的 3.09。
2. **hinge 阈值**：body_up_z > 0.7 对应倾斜 < 45°，这约为终止边界（完全倒地 body_up_z≈0.0）的 70% 处。
3. **gate 不塌缩**：height_factor 改用不乘入前进速度，而是作为正向生存奖励，避免扭曲前进信号。
4. **单组件 ≤ 2x 主信号**：upright_penalty 在极端倾斜时 max ≈ -0.7（body_up_z=0），远小于 forward_reward 的 3.09。
5. **总惩罚负担**：height_factor 改为正向奖励约 +0.1（撤销旧门控的压迫），upright_penalty per-step ≈ 0.03，action_cost per-step ≈ -0.04，合计约 -0.01，远小于 forward_reward 的 3.09。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    # body_up_z = 1 表示完全直立，= 0 表示完全倒地
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控（作为独立正向生存奖励，不扭曲前进信号） ----------
    # [0.2, 0.3] 线性 0→1, [0.9, 1.0] 线性 1→0
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)  # 安全区=1, 越危险→0
    height_reward = 0.1 * height_factor  # 独立的生存奖励，不乘入前进信号

    # ---------- 主学习信号：前进速度 ----------
    forward_reward = 1.0 * body_x_velocity

    # ---------- 直立姿态约束（hinge 惩罚，只在危险时激活） ----------
    # body_up_z < 0.7 时开始惩罚，0.7 对应约 45° 倾斜
    upright_error = max(0.0, 0.7 - body_up_z)
    upright_penalty = -1.0 * upright_error

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = forward_reward + height_reward + upright_penalty + action_cost

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

---

# 诊断摘要
- **audit**: 当前 reward 的姿态惩罚是全时二次惩罚，每步都抑制动作幅度，导致 agent 无法学习有效步态（越慢越容易保持直立但越慢越不稳定）。高度门控乘入前进信号扭曲了学习目标。
- **behavior**: agent 慢速徘徊，为了最小化 upright_penalty 而减少动作幅度，但这反而使姿态更差，陷入"低力矩→差姿态→更多惩罚"的恶性循环。
- **signal**: 缺少只在危险状态介入的姿态hinge，现有的全时惩罚是误导向的负信号。
- **level**: Level 2
- **hypothesis**: 将全时二次姿态惩罚替换为 hinge（只在大倾斜时激活），agent 在正常步态中不再受罚，可以自由探索力矩空间学习动态步态；前进信号不再被高度门控扭曲；更强但合理的力矩惩罚引导高效步态。预期 agent 能学会中等速度的稳定行走，terminated 率下降。
- **risk**: 如果 hinge 阈值 0.7 太松，agent 可能长期在倾斜状态下行走而不受罚，导致步态不自然。但如果发生过早终止增加，可在下一轮收紧 hinge 斜率。
