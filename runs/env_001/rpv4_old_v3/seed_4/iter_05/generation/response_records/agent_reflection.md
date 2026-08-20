# Response Record

---

## 1. 诊断报告

### evidence

| 证据 | 数据 |
|---|---|
| 本轮 score | 24.69（目标 200，差距 8x） |
| Checkpoint 40% / 80% | 均为 **0.00** — 训练过程完全无学习信号 |
| terminated | 10/20（一半无法着陆） |
| contact active_rate | **0.5%**（iter_4），iter_3 为 **0%（死组件）** |
| contact 历史轨迹 | iter_1: 31.9% → iter_2: 12.0% → iter_3: 0.0% → iter_4: 0.5% |
| pos magnitude_share | iter_3: 97.9%（exploit），iter_4: 58.1% |
| 跨轮 contact 形态 | **从未改变**：始终是 `nleft * nright`（二元：双腿同时触地才 1.0） |
| iter_3→iter_4 干预 | 仅将 w_pos 从 50→8，未改变 contact 数学形态 |

### behavior_diagnosis

策略当前处于「能靠近但不能着陆」的半失败状态：
- pos 信号密集且主导（58.1% magnitude），驱动 agent 朝垫子移动
- contact 信号极其稀疏（0.5% active），agent 几乎从不触发双腿着陆奖励，因此**没有梯度教会它如何着陆**
- 稳定性惩罚（vel/angle/angvel）总计 ep_sum ≈ -2.45，在 gate 区域内力度微弱，不足以引导减速和姿态控制
- iter_3 中 w_pos=50 导致 contact 完全死亡，说明 **pos 信号稍强就会淹没 contact** — 这是二元 contact 的结构性脆弱

### signal_completeness

- ✅ 接近信号（pos）：密集、有梯度、正常工作
- ❌ 着陆信号（contact）：**稀疏到不可学习**。二元 `nleft * nright` 要求双腿同时接触才给奖励，agent 无法获得「先放一只脚」的中间梯度
- ⚠️ 稳定性信号（vel/angle/angvel）：权重过低（各 0.5），在 gate 区域内每步惩罚约 0.1-0.5，远弱于 pos 的每步约 0.01-0.05 和 contact 的每步约 0-10

### selected_level

**L2** — 改变 contact 组件的数学形态（二元→渐进），同一语义方向（着陆信号）。这是该语义功能的**首次形态变更**（历史 4 轮从未改过 contact 公式）。

### selected_intervention

**目标组件：`contact`**

修改内容：
1. **`contact_proxy`**：从 `nleft * nright`（二元，0 或 1）改为 `nleft + nright`（渐进，0/1/2）— agent 可以获得单腿触地的中间奖励，形成"先放一只脚→再放另一只"的学习梯度
2. **`contact_threshold`**：从 0.3 扩大到 0.5 — 在更远处就给 contact 信号，让 agent 在接近阶段就能感知着陆 incentive
3. **`w_contact`**：从 10.0 调到 6.0 — 旧最大值 10×1=10，新最大值 6×2=12，天花板相近但分配更渐进

### falsifiable_hypothesis

**如果 contact 从二元改为渐进，agent 将获得可学习的着陆梯度，contact active_rate 应从 0.5% 显著上升（预期 5-20%），terminated 率应上升，score 应超过 iter_2 的 64.24。**

理由是：iter_2（best=64.24）证明了 contact 信号可以驱动着陆行为，但 12% active_rate 的二元信号太脆弱，被 w_pos 从 3→50 就彻底杀死。渐进化使 contact 信号对 pos 权重不那么敏感，且提供中间梯度。

### expected_next_round

- `contact active_rate`：从 0.5% → 预计 5-15%
- `contact ep_sum`：上升（更频繁触发 + 更宽阈值）
- `terminated`：从 10/20 → 预计 12-16/20
- `score`：预计 40-80（超过 iter_2 的 64.24）
- `pos magnitude_share`：从 58.1% → 下降（contact 分摊更多）
- `checkpoint`：应从 0.00 开始出现非零值（梯度可学习）

### main_risk

contact 渐进化后可能重新成为 exploit 目标：agent 学会「反复单腿触地再抬起」来刷 contact 奖励而不真正着陆。如果下轮 contact magnitude_share > 85%，需要加触发条件（如每 episode 只奖励首次接触、或要求持续性）。

---

## 2. 完整 Python 代码

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

    # -- 3) safe contact: reward each leg touching near center (gradual) --
    contact_proxy = nleft + nright                    # 0, 1, or 2: gradual (was binary nleft*nright)
    contact_threshold = 0.5                           # wider reach (was 0.3)
    contact_factor = max(0.0, 1.0 - next_dist / contact_threshold)
    w_contact = 6.0                                   # adjusted: old max=10, new max=12
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
