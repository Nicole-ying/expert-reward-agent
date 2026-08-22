## 诊断与干预决策

---

### 1. evidence

| 证据来源 | 数据点 | 值 |
|---|---|---|
| 本轮 feedback | score | -21.40（范围 [-48.68, 18.11]） |
| 本轮 feedback | len | 1000.00（满步截断 20/20，terminated=0） |
| 本轮 feedback | contact active_rate | **0.0%（死组件）** |
| 本轮 feedback | pos magnitude_share | **97.9%（绝对主导）** |
| 本轮 feedback | pos ep_sum | **59.09** |
| get_component_history | pos 跨轮变化 | iter1: -88.13 → iter2: +3.55 → iter3: **+59.09（16x 跳变⚠）** |
| get_component_history | contact 跨轮变化 | iter1: +555.99 (31.9%) → iter2: +305.31 (12.0%) → iter3: **0.00 (0.0%)** |
| read_reward_code | iter2→iter3 唯一变化 | `w_pos`: 3.0 → **50.0** |
| read_reward_code | iter1→iter2 唯一变化 | pos 数学形态：状态值 `-dist_sq` → 改善量 `prev_dist - next_dist`，w_pos=3.0 |
| memory | 历史最佳 | iter2, score=**64.24**, len=924, terminated=10/20 |

---

### 2. behavior_diagnosis

Agent 学到的策略是**「在着陆垫附近无限徘徊，永不降落」**。

根因：`w_pos=50.0` 使得每步接近垫子的奖励（~0.06/step）在 1000 步内累积到 +59，远超任何可能的着陆奖励。同时，gate 机制在靠近垫子时激活速度/姿态惩罚，agent 学会了停在 gate≈0 的中等距离——既拿接近奖励，又避开惩罚。contact 奖励（最大 10/step，且需要双腿同时触垫）相比之下稀疏且难以发现，agent 根本不探索着陆行为。

**本质是奖励尺度失衡**：接近奖励过于丰厚，agent 无需完成着陆即可获得可观总分。

---

### 3. signal_completeness

| 职责 | 信号 | 状态 |
|---|---|---|
| 接近引导 | `pos`（delta 距离） | ✅ 存在但权重过大（50→应降至 5-10） |
| 着陆激励 | `contact`（双腿触垫 × 距离因子） | ❌ 死组件 —— 太稀疏、相对权重太低 |
| 稳定性约束 | `vel/angle/angvel`（门控惩罚） | ✅ 正常，但被接近信号淹没 |

缺失的职责：**无**。所有必要信号都存在，问题纯粹是权重失衡。

---

### 4. selected_level

**L1（系数调优）**

触发条件：
- `get_component_history('pos')` 明确警告：ep_sum 从 3.55→59.09（16x 跳变），禁止修改数学形态，只允许 L1。
- 本轮唯一改动是系数（iter2→iter3 `w_pos` 3.0→50.0），结构未变，回退/调整系数即可。

---

### 5. selected_intervention

**目标组件**：`pos`

**修改**：`w_pos`: 50.0 → **8.0**

选择 8.0 的理由：
- iter2 的 `w_pos=3.0` 偏弱（pos ep_sum 仅 3.55，接近信号不足，仅 10/20 着陆）
- iter3 的 `w_pos=50.0` 过强（pos ep_sum 59.09，contact 死亡，0/20 着陆）
- 8.0 = 3.0 × 2.67，给予比 iter2 更强的接近引导，同时远低于 50.0 的爆炸区间
- 着陆时 contact_reward ≈ 10 × 双腿 × (1-dist/0.3)，在 dist≈0 时 = 10.0/step，而接近奖励约 8 × 0.01~0.05 = 0.08~0.4/step。着陆激励在近距离远强于接近激励。

---

### 6. falsifiable_hypothesis

> 降低 `w_pos` 到 8.0 后，每步接近奖励从 ~0.059 降至 ~0.01，contact 的相对吸引力从「可忽略」变为「值得探索」。Agent 将重新发现着陆行为，contact active_rate 从 0% 回升至 5-15%，terminated 比率从 0/20 回升至 ≥5/20，score 应回到正值区间（目标 >30）。

---

### 7. expected_next_round

| 指标 | 当前 (iter3) | 预期 (iter4) |
|---|---|---|
| score | -21.40 | +20 ~ +80 |
| len | 1000 | 600~950（部分着陆截断） |
| terminated | 0/20 | 5~12/20 |
| contact active_rate | 0.0% | 5%~15% |
| pos ep_sum | 59.09 | 8~20 |
| pos magnitude_share | 97.9% | 40%~70% |
| contact magnitude_share | 0.0% | 20%~55% |

---

### 8. main_risk

`w_pos=8.0` 可能仍然偏高（如果 agent 已在 iter3 深陷「徘徊」局部最优，8.0 仍足以维持该策略）。若下轮 contact 仍 <2%，需考虑 L2 改造 contact（从 `nleft*nright` 的纯稀疏 AND 改为 `(nleft+nright)/2` 的渐进信号，或在 gate 区增加单腿触垫小奖励）。

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
    w_pos = 8.0
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