# Response Record

# 设计理由

## 信号覆盖审计结论
上一轮`health_gate`同时混合了高度因子和直立因子，相乘在`forward_gated`上。当agent中度倾斜时(`body_up_z`在0.2~0.6区间)，`upright_penalty`在惩罚，`health_gate`压低前进奖励乘数，形成**双重压制**。高度边界(0.2, 1.0)附近`health_gate`衰减过慢(在0.2时仍有0.5)，不够紧迫。`z_velocity_penalty`触发率4.9%，已成僵尸。

## 行为诊断
agent在16/20 episode中跌倒终止，`upright_penalty`贡献-141平均每轮，`forward_gated`受gate钳制仅有+122。agent学会了苟活(len=319)但无法同时保持直立和前进，一旦倾斜就陷入无梯度可恢复的死亡地带。iter 1简单线性得分远优于当前框架，说明复杂的联合gate/penalty结构适得其反。

## 干预策略 — Level 2 结构变换
**分离高度与直立的关注点**：
1. **高度gate仅乘前进奖励**：`forward = body_x_velocity × height_factor`。`height_factor`在`(0.3, 0.9)`为1.0，在`0.2`和`1.0`处精确降至0，紧贴终止边界。这样agent为获得任何前进奖励必须维持安全高度。
2. **直立改为温和正向bonus**：`upright_bonus = 0.1 × body_up_z`。无惩罚、无gate，给全梯度引导恢复直立，不与前进奖励冲突。
3. **移除所有惩罚组件**：删除`upright_penalty`和`z_velocity_penalty`，消除死亡螺旋。

**数学形式与校准**：
- 主信号per-step约0.5，upright_bonus per-step ≤ 0.1 (20%以内)
- 高度gate无惩罚负担，仅在边界外将前进奖励归零
- 总惩罚=0，符合轻负担原则

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]

    # ---------- Derived signals ----------
    # Uprightness: 1.0 = perfectly upright, -1.0 = upside-down
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Height gate (tightly coupled to termination bounds) ----------
    # Healthy zone (0.30, 0.90) -> factor=1.0
    # Linear decay to 0.0 at termination boundaries (0.20, 1.00)
    height_low_danger = max(0.0, (0.30 - body_z) / 0.10)   # 0.30→0.20
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    height_high_danger = max(0.0, (body_z - 0.90) / 0.10)  # 0.90→1.00
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # ---------- Reward components ----------
    # Forward velocity — only available when height is safe
    forward_reward = float(body_x_velocity) * height_factor

    # Upright bonus — gentle positive gradient, no penalty
    upright_bonus = 0.1 * body_up_z

    # ---------- Total reward ----------
    total_reward = forward_reward + upright_bonus

    components = {
        "forward": forward_reward,
        "upright_bonus": upright_bonus,
        "height_factor": height_factor,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止边界缺紧迫前兆，直立信号引发死亡螺旋（gate+penalty双重压制中度倾斜区间）。
- **behavior**: agent苟活但不断跌倒，无法同时维持直立与前进（terminated 16/20，len=319，score=-101）。
- **signal**: 缺失中度倾斜区的恢复梯度，直立惩罚与gate相互耦合无出路。
- **level**: Level 2 — 结构变换：分离高度与直立关注点，gate仅限制前进，直立改为纯正向bonus，移除所有惩罚。
- **hypothesis**: 给前进始终保留梯度（仅受高度门控），直立用温和奖励引导而非惩罚，消除死亡螺旋，恢复iter 1的前进势头同时延长存活。
- **risk**: 无直立惩罚可能导致agent不优先恢复直立，需观察；必要时可后续加极低阈值的软hinge penalty（如body_up_z<-0.3）。
