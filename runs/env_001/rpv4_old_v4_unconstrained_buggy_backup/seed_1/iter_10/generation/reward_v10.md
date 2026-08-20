## 诊断报告

### Evidence

| 来源 | 关键数据 |
|---|---|
| **v4 (BEST)** | score=135.22, len=285, 19/20 terminated, proximity=40, binary contact, landing ep_sum=161.4, active_rate=23.5% |
| **v8** | proximity=30, binary contact → hovering (len=760, score=14.44), landing active_rate=3.4% — 接近梯度太弱 |
| **v9 (当前)** | proximity=35, graduated contact (0.5× single-leg) → crash diving (len=69, score=-102), landing active_rate 仍为 3.4%, ep_sum=2.3 — graduated contact 完全无效 |
| **v9 checkpoint** | score trend: -122→-107→-107→-78→-78→-94→-105→-103→-102→-101，mid-training 短暂改善后崩塌 |
| **v9 soft_landing_bonus 趋势** | flat at ~2.3 across ALL checkpoints — 死信号，agent 在整个训练过程中从未真正发现着陆 |
| **soft_landing_bonus 历史** | v4: 161.4 (23.5%) → v5: 1.0 (1.5%) → v6: 0.0 → v7: 1.5 → v8: 39.1 (3.4%) → v9: 2.3 (3.4%) — 自 v4 后从未恢复 |
| **proximity_progress 历史** | v4: 44.4 → v5: 55.9 (crash) → v8: 24.9 (hover) → v9: 39.2 (crash) — proximity 在 30-50 之间震荡，临界点极其敏感 |

### Behavior Diagnosis

当前策略 **(v9)** 的行为模式：**crash diving — agent 学到的唯一有效行为是"向下冲"**。全部 20 集在平均 68 步内早期终止，得分 -102。soft_landing_bonus active_rate 仅 3.4%（在已 crash 的轨迹中偶然出现单腿擦碰），ep_sum 微不足道 2.3。graduated contact（0.5× 单腿接触）完全未能解决稀疏性问题，因为 agent 在到达 pad **之前**就已坠毁。

**根本原因**：binary contact gate 和 graduated contact gate 都是稀疏信号——它们只在 agent 实际接触 pad 时才激活。但 contact 是终端事件：agent 必须先学会"靠近 pad 并减速"才能触发 contact。目前的奖励结构缺少一个**连续、随高度衰减的"接近地面"信号**来桥接 proximity_progress（全程有效）和 contact（仅在着陆瞬间有效）。

v4 之所以成功，恰恰因为 proximity=40 的强梯度**暴力推动** agent 到达 pad，然后 binary contact landing bonus 接管。但这种配置极其脆弱——proximity 稍低就悬停，稍高就 crash diving。

### 干预方向（L2 单一语义修改）

**核心改动**：将 `soft_landing_bonus` 的门控从 **接触信号（contact）→ 高度信号（height）**。将组件重命名为 `landing_readiness`。

**为什么这从根本上不同**：
- contact 是二值的、稀疏的、终端事件——只有在 agent 已经着陆后才激活
- height 是连续的、全程可观测的——`near_ground = 1/(1+5·height)` 在任何高度都提供梯度
- 新信号在 agent 靠近地面时逐渐增强，鼓励"减速 + 保持姿态"——即使 agent 尚未接触 pad
- 这创建了从 proximity_progress（全程导航）到 soft landing（终端着陆）之间的**连续信号桥**

**参数选择**：
- proximity_progress = 30.0：中等梯度，由新的 height-gated 信号补充下行压力，不会单独导致 crash diving（v8 验证）
- near_ground 陡度 = 5.0：足够陡峭以防止中空悬停 exploit，但在 height<1 区域提供有用梯度
- landing_readiness 系数 = 3.0：低于 v4 的 4.0（因为新信号的 active_rate 会显著更高）

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v10 — Replace contact-gated soft_landing_bonus with height-gated landing_readiness.

    Diagnosis from v9:
    - soft_landing_bonus active_rate=3.4%, ep_sum=2.3 — dead signal across all checkpoints.
      The agent never discovers landing because the contact gate is a terminal event:
      the agent must already have reached the pad (and touched it) to receive any signal.
    - Graduated contact (0.5x single-leg) failed to fix this: agent crashes before reaching
      the pad, so single-leg contact is just as sparse as dual-leg contact in practice.
    - The fundamental gap: there is NO continuous signal bridging proximity_progress
      (all-phase navigation) and soft_landing_bonus (contact-only terminal event).

    Single semantic change (L2):
    - Replace contact gate with continuous height gate: `near_ground = 1/(1+5*height)`.
      Height is always observable and provides gradient at any altitude, unlike contact
      which is only informative at touchdown. This creates a "slow down near the ground"
      pressure that the agent can discover long before it ever touches the pad.
    - Proximity kept at 30.0 (v8-level, proven non-crash-diving) because the new
      height-gated signal now provides additional downward motivation.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- landing_readiness: height-gated continuous quality signal ----------
    # Key change: binary contact gate → continuous height gate.
    # near_ground provides gradient at any altitude, bridging approach → landing.
    height = next_obs[1]                                 # vertical distance to pad surface
    near_ground = 1.0 / (1.0 + 5.0 * height)             # 1.0 at h=0, 0.5 at h=0.2, 0.17 at h=1

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)        # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))   # → 1.0 at θ=0, → 0.17 at |θ|=1

    landing_readiness = 3.0 * near_ground * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + landing_readiness

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "landing_readiness": landing_readiness
    }
    return float(total_reward), components
```