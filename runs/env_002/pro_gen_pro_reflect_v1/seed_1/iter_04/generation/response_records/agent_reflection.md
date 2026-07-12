# Response Record

# 设计理由

## 0. 信号覆盖审计与诊断

### a) 终止 → 前兆信号
环境事实声明终止主要由 `body_fallen_over` 导致。"前兆"信号 `hull_angle` 已被有效使用：
- `gate` 在 `hull_angle` 过大时（>0.8）衰减前向奖励，阻止接近摔倒时仍追逐速度。
- `stability_penalty` 惩罚 `hull_angular_velocity` 的高频晃动。 **覆盖合格。**

### b) 目标 → 进度信号
核心目标：走得远、走得快、最小化能耗。
- “走得快”：`velocity_gated` 直接关联，但加权太低（实际 per-step 仅 ~0.234）。**回报强度不足，需提升。**
- “走得远”：**无直接信号**。当前奖励鼓励速度，但不奖励持续前进。严格来说，速度积分=距离，但 vel 低时积分也低。可以后续考虑添加生存/distance 奖励，但本轮优先修复当前性能悬崖。
- “最小化能耗”：`action_efficiency` 覆盖。

### c) 效率信号
动作维度=4 (≥6? 否)，但 action_efficiency 已存在且 active_rate=100%。

### d) 僵尸组件
所有组件 active_rate=100%，无僵尸。

### e) 一句话结论
**当前 reward 漏了持续前进（生存/distance）的直接激励，且动作惩罚过重抑制了速度探索，导致策略退化为低速维持平衡。**

---

## 1. 行为诊断

### agent 在做什么？
从反馈数据看：
- `len=461.85`：大幅低于 iter1 (911) 和 iter2 (803)，说明 agent 更快终止（摔倒或不再前进）。
- `velocity_gated per-step = 0.234` (181.58 / 461.85 * 2? 不对，episode_sum_mean 是 total，per-step = 181.58/461.85 ≈ 0.393)。**速度偏低。**
- `action_efficiency per-step = -0.041` (-18.89/461.85)，看起来不大，但对于低速度信号已是其 10%+ (0.041/0.393≈10.4%)。
- 累计记录：iter3 引入 action_efficiency 后 **len 暴跌（803→462），score 断崖（243→28.7）**。

**结论：agent 学会了极低能耗、极低速度的平衡策略。动作惩罚让任何有意义的运动都带巨大代价，agent 选择“不动或少动”以避免惩罚。**

### 干预哪个目标？
**提升“走得快”（前向速度奖励强度）。** 当前速度水平太低 (0.393/step)，需要更强的吸引力把 agent 从低能耗低速度的 local optimum 拉出来。

### 方向还值得继续吗？
累积记录显示：
- iter2 提高稳定性惩罚 → 预判 ❌，实际 score 下降。
- iter3 加动作惩罚 → 预判 ❌，实际断崖下降。
- **连续 2 轮预判 ❌（接近 3 轮阈值），但属于 scale 问题（惩罚过重）而非结构方向错误。**
骨架本身（gated velocity + stability + efficiency）在第 1 轮表现优异 (294.82)，说明结构可行。问题在于 **iter3 添加惩罚时打破了 balance**。不再加新东西，应修复平衡。

---

## 2. 选择干预层级

**Level 1 — 尺度修复**：
证据：
- `action_efficiency per-step (0.041)` / `velocity_gated per-step (0.393)` ≈ 10%，看起来不大，但考虑到 agent 最优策略可能大幅提升速度，动作惩罚绝对值将同步增加（速度需要力矩支持），**惩罚的动态抑制效应被低估。**
- 为鼓励 agent 探索更快步态，**需大幅降低动作惩罚系数**，让“高速度高力矩”组合的收益显著大于“低速度低力矩”。
- `stability_penalty per-step (0.0002)` 可忽略，但保留。

**动作：降低 `w_action` 从 0.1 → 0.02，降低 `w_ang_vel` 从 0.1 → 0.05。**
不添加新组件，恢复 iter1 的良好平衡，并更温和地引入效率约束。

---

## 3. 设计校准

- **主信号 per-step**：约 0.393（当前），优化后预期 0.5+。
- **新动作惩罚 per-step 目标**：≤ 0.15x 主信号 = 0.06。当前 0.041 已在此范围，但需防止优化后主信号提升时惩罚同比增大。降低至 0.02 系数，预计 per-step ~0.008 (假设 action mag 类似)，更安全。
- **新稳定性惩罚 per-step 目标**：可忽略，0.05 仍安全。
- **总惩罚负担**：0.008 + 0.0001 ≈ 0.008，为主信号的 ~1-2%，远 < 50%。
- Hinge 阈值：gate 参数 danger_angle=0.8, max_angle=1.2 已合适（终止边界 60-80%）。
- Gate 不塌缩：当前 hull_angle 常态 <0.5，gate≈1.0。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角，越小越稳
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度，正值向前

    # ---------- 健康门控：当躯干倾斜过大时自动衰减前向奖励 ----------
    danger_angle = 0.8   # 接近摔倒的阈值（~45°）
    max_angle = 1.2      # 完全关闭主奖励的阈值（~69°）
    # 线性衰减门：在 [0, danger_angle] 恒为 1，在 [danger_angle, max_angle] 从 1 线性降到 0
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号：被门控的前向速度 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：温和地惩罚角速度，减少晃动 ----------
    w_ang_vel = 0.05  # 从0.1降至0.05，避免过度约束动态探索
    stability_penalty = -w_ang_vel * (hull_ang_vel_next ** 2)

    # ---------- 动作效率：极轻惩罚动作力矩平方和，鼓励探索更高效步态 ----------
    w_action = 0.02  # 从0.1大幅降至0.02，避免抑制速度探索
    action_efficiency = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- 总奖励与组件字典 ----------
    total_reward = velocity_gated + stability_penalty + action_efficiency
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty,
        "action_efficiency": action_efficiency
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少直接“走得远”信号，动作惩罚过重压制速度探索，导致迭代3表现断崖。
- **behavior**: agent 陷入低速度、低能耗的局部最优，维持平衡但不再前进。
- **signal**: 主信号 velocity_gated 太弱 (per-step~0.39)，动作惩罚 (per-step~0.04) 动态抑制效应被低估。
- **level**: Level 1
- **hypothesis**: 大幅降低动作惩罚系数 (0.1→0.02) 将恢复 iter1 的良好探索动力，让 agent 更愿产生有效力矩来提升前向速度，从而跳出局部最优，恢复 score→300。
- **risk**: 动作惩罚过低可能导致效率下降或步态夸张，但当前约束仍是够轻，主要风险是 score 提升不足而非过度。必要时可后续微调系数。
