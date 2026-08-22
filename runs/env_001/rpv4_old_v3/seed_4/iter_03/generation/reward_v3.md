---

## 诊断与干预决策

### 1. evidence

| 数据点 | iter_01 | iter_02 | iter_03(当前) | 解读 |
|---|---:|---:|---:|---|
| score | 26.74 | 64.24 | 64.24 | 迭代停滞，iter_02→iter_03 零改善 |
| len | 341.75 | 924.05 | 924.05 | 回合长度翻倍但不再增长 |
| terminated | 15/20 | 10/20 | 10/20 | 成功率反降，一半到不了垫子 |
| checkpoint 40% | 0.00 | 0.00 | 0.00 | 三轮训练中期均无学习信号 |
| checkpoint 80% | 0.00 | 0.00 | 0.00 | 同上，训练过程完全平坦 |
| **contact** mag_share | 85.4% | 97.8% | 97.8% | 稀疏着陆奖励绝对主导 |
| **contact** active_rate | 31.9% | 12.0% | 12.0% | 越来越稀疏——只有恰好着陆的几步触发 |
| **pos** mag_share | 13.5% | 1.4% | 1.4% | 接近信号被淹没，几乎不可见 |
| **pos** per-step | -0.258 | 0.004 | 0.004 | 切换为 delta 后转正，但量级微不足道 |

**量级对比（每步均值的数量级差距）**：
- contact 活跃时: ~2.75/step（w_contact=10 × contact_factor≈0.3-1.0）
- pos: ~0.004/step（全episode均值）
- **接触信号约是接近信号的 ~700 倍**

### 2. behavior_diagnosis

Agent 学到的策略是**「碰运气着陆」**：

- **10/20 成功**：偶然飘到垫子上，触发 contact 巨额奖励（10.0×contact_factor/step），然后停在垫上直到 settle，累计大量 contact 奖励。
- **10/20 失败**：全程未抵达垫子，随机漂移至 truncation。因为 pos 信号太弱（每步 0.004），agent 无法从「靠近一步」和「远离一步」之间分辨出有意义的梯度。
- **训练中 checkpoint 均为 0.00**：训练时 agent 甚至还没学会靠近垫子——因为稀疏的 contact 在训练早期几乎从不触发，而 pos 太弱无法提供有效梯度。整个训练过程是「随机游走直到偶然着陆」。

核心矛盾：**contact 是唯一有效的奖励源，但它太稀疏；pos 是唯一稠密的引导信号，但它太弱。**

### 3. signal_completeness

| 职责 | 现状 | 评价 |
|---|---|---|
| 接近引导 | pos (delta-dist, w=3.0) | ✅ 形态正确（改善量），❌ 量级不足 |
| 着陆判定 | contact (both legs × dist_factor, w=10.0) | ✅ 形态合理，❌ 权重过大淹没其他信号 |
| 速度约束 | vel penalty (gated) | ✅ 有，但太弱（mag 0.5%） |
| 姿态约束 | angle + angvel penalty (gated) | ✅ 有，但太弱（mag 0.3%） |
| 燃料效率 | 无 | ❌ 缺失——任务要求节省燃料 |
| 终端奖励 | 无（不可用 explicit flag） | ⚠️ 不可用，但 contact 在 settle 期间持续累加已部分充当 |

当前最紧急缺口：**接近引导信号强度**。燃料效率是次要目标，可在接近问题解决后再引入。

### 4. selected_level

**L1** — 仅调整 `pos` 组件的系数 `w_pos`。

触发条件：
- `get_component_history('pos')` 警告：pos 在 iter_01→iter_02 跳变 3554x（从 -88→+3.6），标记为 "recently DISCOVERED"。规则要求**只做 L1 系数调优，不改数学形态**。
- iter_02 已将 pos 从 `state-value 罚分` 切换为 `delta 改善奖励`，数学形态正确，不需要再改。
- iter_02→iter_03 分数完全停滞，说明单纯重复训练无济于事，必须调整信号强度。

### 5. selected_intervention

**目标组件**：`pos`

**修改**：`w_pos` 从 `3.0` → `50.0`（约 16.7× 提升）

**理由**：
- 当前 pos ep_sum=3.55。线性放大到 w_pos=50 后约为 ~59，占总 magnitude 约 16%（vs 当前 1.4%）。
- 接近信号从「噪声级别」提升到「可感知级别」，agent 在长距离接近阶段能获得有效梯度。
- 保留 contact 的主导地位（~80%）——着陆仍然是主要目标，只是接近路径不再完全黑暗。

### 6. falsifiable_hypothesis

> 如果 pos 信号足够强（magnitude_share 从 1.4% → 10-20%），agent 应能在训练过程中学会系统性接近垫子，表现为：(a) checkpoint 分数不再全是 0.00；(b) terminated 率上升（更多 episode 抵达垫子）；(c) 总 score 提升。

若下一轮 pos 的 magnitude_share 上升但 score 不变，说明 agent 在「振荡套利」（来回移动刷 delta），需要引入 max(0, prev_dist - next_dist) 截断或切换为其他形态——但那属于 L2，本轮不做。

### 7. expected_next_round

| 指标 | 当前 | 预期 | 原因 |
|---|---|---|---|
| score | 64.24 | 80-120 | 更强的接近梯度 → 更多 episode 抵达垫子 |
| terminated | 10/20 | 12-16/20 | 接近信号引导更多 episode 成功 |
| pos ep_sum | 3.55 | 40-80 | w_pos 放大 16.7×，但接近行为也会改善 |
| pos magnitude_share | 1.4% | 10-25% | 权重提升 + 更好的接近行为 |
| contact magnitude_share | 97.8% | 70-85% | 被 pos 稀释，但仍主导 |
| checkpoint 40%/80% | 0.00 | >0 | 训练中应有非零学习信号 |

### 8. main_risk

**振荡套利**：agent 学会「远离→靠近→远离→靠近」来回刷 delta 奖励，而不是持续接近。当前 truncation 机制（episode 长度上限）提供了一定的自然惩罚（浪费时间降低接触机会），但如果出现此问题，下一轮需对 pos 做 `max(0, prev_dist - next_dist)` 截断（L2）。

---

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
    w_pos = 50.0
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