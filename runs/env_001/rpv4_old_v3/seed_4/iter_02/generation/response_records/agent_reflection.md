# Response Record

---

## 诊断与干预决策

### 1. evidence

| 数据点 | 值 | 解读 |
|---|---|---|
| score | 26.74 (目标 200) | 远未达标 |
| len | 341.75 | 回合偏长，接近时间上限 |
| terminated | 15/20 | 多数能着垫，但 5 个被截断 |
| score_range | [-78, 258] | 巨大方差：部分坠毁、部分成功 |
| checkpoint 40%/80% | 均为 0.00 | 训练中期策略仍无进步信号 |
| **contact** ep_sum | +555.99 (85.4% mag) | 绝对主导，但 active 仅 31.9% — 稀疏高额奖励 |
| **pos** ep_sum | -88.13 (13.5% mag) | 始终激活 (100%)，始终为负，拖累总分 |
| **vel** ep_sum | -5.30 (0.8% mag) | 门控惩罚过于微弱，形同虚设 |
| **angle** ep_sum | -0.44 (0.1% mag) | 淹没状态，几乎无训练信号 |
| **angvel** ep_sum | -1.24 (0.2% mag) | 淹没状态，active 92.7% 但贡献可忽略 |

组件历史无 5× 跳变（仅一轮数据），无 "刚激活" 警告。

### 2. behavior_diagnosis

策略当前行为：**缓慢漂移靠近着陆垫 → 双腿触垫 → 在垫上停留数十步获得 contact 奖励 → 被截断或被 pos 惩罚逼离**。

三个关键失败模式：

- **无进度驱动**：`pos = -dist_sq` 只说「远离不好」，不说「靠近好」。agent 没有正向激励去加速接近，导致平均 341 步才完成。
- **着陆后惩罚矛盾**：agent 成功着陆后，pos 组件仍然每步罚款（因永远不可能精确停在原点），与 contact 奖励形成对冲，削弱「保持着陆」的学习信号。
- **稳定性信号太弱**：vel/angle/angvel 被门控 + 小权重双重压制，量级仅为 contact 的约 1/500，agent 学不到减速/调姿。

### 3. signal_completeness

| 职责 | 当前承担者 | 状态 |
|---|---|---|
| 接近/进度 | pos（纯惩罚，无正向梯度） | ❌ 信号方向错误 |
| 着陆检测 | contact（稀疏乘积） | ✅ 有效但过强 |
| 软着陆/减速 | vel, angle, angvel（门控惩罚） | ❌ 淹没，量级不足 |
| 坠毁惩罚 | 无 | ❌ 缺失 |
| 时间压力 | 无 | ❌ 缺失 |

最紧迫的缺口：**进度信号**。pos 是唯一覆盖全程的密集信号，但它只有惩罚没有奖励，agent 无法区分「正在变好」和「正在变差」。

### 4. selected_level

**L2** — 修改 `pos` 组件的数学形态（稀疏→连续、状态值→改善量）。

触发条件：这是首轮干预，`get_component_history` 无警告，过去无同方向失败记录。pos 的「始终为负」特性是设计层面问题，纯系数调优（L1）无法改变信号方向，必须改数学表达式。

### 5. selected_intervention

**目标组件**：`pos`

**修改内容**：从「当前位置的平方距离惩罚」改为「步间距离改善量的线性奖励」。

- 旧：`pos_reward = -w_pos * (nx² + ny²)` — 始终 ≤0，远处惩罚极大，近处惩罚小但不归零
- 新：`pos_reward = w_pos * (prev_dist - next_dist)` — 靠近时为正、远离时为负，着陆后距离不变则 ≈0

同时上调配重 `w_pos: 1.0 → 3.0` 以保持合理的每步量级（预估：从初始距离 ~8 到着垫 ~0.1，全程累积约 +24；若远离则累积为负）。

### 6. falsifiable_hypothesis

> 将 pos 从状态惩罚改为进度奖励后，agent 会获得「靠近即奖励」的密集正向梯度，从而学会更直接、更快速的下降轨迹。预计 episode 长度下降，pos 组件的 signed_share 从负转正（或接近零），且 score 提升。

如果下一轮 pos 的 active_rate 仍然 100% 但 ep_sum_mean 仍为负 → 说明 agent 在远离 pad，需要检查是否其他组件（如 contact）的稀疏性导致探索不足。

### 7. expected_next_round

| 指标 | 当前 | 预期 | 理由 |
|---|---|---|---|
| score | 26.74 | ↑ 60–120 | 进度奖励加速接近，contact 累积增加 |
| len | 341.75 | ↓ 200–280 | 靠近有奖励，agent 不再拖延 |
| pos ep_sum | -88.13 | 正或接近零 | 进度奖励在靠近时为正 |
| pos signed_share | -13.5% | 正或小负 | 不再拖累总分 |
| contact ep_sum | 555.99 | 持平或↑ | 更快着陆 → 更多着垫步数 |
| vel/angle/angvel | 淹没 | 仍淹没 | 本次未动稳定性组件 |

### 8. main_risk

**过度摆动**：进度奖励可能让 agent 学会「靠近-远离-再靠近」的循环来刷分。缓解：着垫后的 contact 奖励（需双腿+近中心）会锚定最终位置，且摆动幅度受初始距离约束（总分有上限）。若下一轮出现 len 上升 + score 不升，则需给 progress 加单调性约束。

---

## 新奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs indices:
    # 0: x_position (relative to pad center)
    # 1: y_position (relative to pad height)
    # 2: x_velocity
    # 3: y_velocity
    # 4: body_angle
    # 5: angular_velocity
    # 6: left_support_contact (0.0 or 1.0)
    # 7: right_support_contact (0.0 or 1.0)

    ox = obs[0]
    oy = obs[1]
    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft = next_obs[6]
    nright = next_obs[7]

    # Distances
    prev_dist = (ox**2 + oy**2 + 1e-8) ** 0.5
    next_dist = (nx**2 + ny**2 + 1e-8) ** 0.5

    # -- 1) approach progress: reward reduction in distance to pad --
    w_pos = 3.0
    pos_reward = w_pos * (prev_dist - next_dist)   # >0 when approaching, <0 when retreating

    # -- 2) soft landing: speed & attitude penalty gated by distance --
    k_gate = 10.0
    gate = 1.0 / (1.0 + k_gate * next_dist)         # near pad -> gate ~1, far away -> gate ~0

    w_vel = 0.5
    vel_penalty = -w_vel * (nvx**2 + nvy**2) * gate

    w_angle = 0.5
    angle_penalty = -w_angle * (nangle**2) * gate

    w_angvel = 0.5
    angvel_penalty = -w_angvel * (nangvel**2) * gate

    # -- 3) safe contact: reward both legs on the pad near center --
    contact_proxy = nleft * nright                    # both legs touching -> 1.0, else 0.0
    contact_threshold = 0.3
    contact_factor = max(0.0, 1.0 - next_dist / contact_threshold)
    w_contact = 10.0
    contact_reward = w_contact * contact_proxy * contact_factor

    total_reward = pos_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        'pos': pos_reward,
        'vel': vel_penalty,
        'angle': angle_penalty,
        'angvel': angvel_penalty,
        'contact': contact_reward
    }

    return total_reward, components
```

**与上一轮的差异（仅 pos 组件）**：

| 方面 | iter_01 | iter_02 |
|---|---|---|
| pos 公式 | `-1.0 × (nx²+ny²)` | `3.0 × (prev_dist − next_dist)` |
| 信号性质 | 始终 ≤0（纯惩罚） | 可正可负（进度奖励） |
| 着陆后行为 | 继续惩罚微小偏移 | 距离不变→约 0，不惩罚 |
| prev_dist 计算 | 无（不需 obs 位置） | 新增 `ox, oy = obs[0], obs[1]` |
