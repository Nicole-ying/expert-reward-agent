---

## 诊断报告

### evidence

| 证据来源 | 关键数据 |
|---|---|
| 最终评分 | score = **-2.98**（目标 200），严重失败 |
| 轨迹长度与终止 | len = 813，truncated = 15/20，terminated = 5/20 → **agent 长期悬停不降落** |
| soft_landing_bonus 最终 | active_rate = **0.3%**，ep_sum_mean = 8.05 → 几乎从不触发，是死组件 |
| proximity_progress | magnitude_share = **82.9%** → 绝对主导，agent 只被"靠近目标"这一个信号驱动 |
| cp 趋势：soft_landing_bonus | cp80% 时 ep_sum 飙到 **256.5**（active 23.4%），cp100% 崩到 108.9（active 4.7%）→ 训练中学会"弹跳刷分"然后过拟合 |
| cp 趋势：score | cp80% 时达到 164.9，cp100% 回落到 122.7，最终评估 -2.98 → **严重过拟合 + 泛化崩溃** |
| 缺失信号 | 无燃料惩罚、无时间压力、无下降激励 |

### behavior_diagnosis

当前策略学习到的是一个**不健康的悬停+弹跳刷分模式**：

1. **主导行为**：proximity_progress 驱动 agent 快速飞到 (0,0) 附近，然后因为没有下降激励而悬停。attitude_penalty 仅微调姿态。
2. **训练中的 exploit**：agent 发现偶尔双腿着地（soft_landing_bonus）能获得巨额奖励（256 ep_sum），于是学会「弹跳」—— 反复轻触垫面刷 bonus 再拉起。这个策略高度依赖特定种子，换种子就失效（final eval 0.3% active）。
3. **根本缺陷**：奖励函数缺少三个关键信号：(a) 持续下降激励（让 agent 愿意落下去），(b) 燃料/时间惩罚（让悬停变贵），(c) 平滑的着陆梯度（而非全有全无的 binary bonus）。

### 干预方案

**完全重写奖励结构**，保留 proximity 作为导航骨架但大幅降权，新增：
- `altitude_reward`：连续奖励低高度，提供下降梯度
- `speed_penalty`：惩罚高速，尤其是近地时
- `contact_reward`：连续版本（0/1/2 腿），配合速度+姿态质量因子，小步奖励
- `fuel_penalty`：点火即罚，抑制引擎滥用
- `step_penalty`：每步小额惩罚，激励尽快完成着陆

设计原则：**悬停在 pad 附近时净奖励为负**，迫使 agent 完成着陆结束 episode。

---

## 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — restructured to fix hovering and landing-sparsity problems.

    Core changes from v1:
    - Added altitude_reward: continuous incentive to descend (fixes hovering).
    - Added fuel_penalty + step_penalty: make lingering near the pad costly.
    - contact_reward now uses continuous n_contact (0/1/2) not binary product,
      so partial contact provides gradient. Per-step value is modest to prevent
      bounce-exploit (fuel+step costs outweigh repeated tiny contact gains).
    - Reduced proximity_progress weight — was 82.9% magnitude share, too dominant.
    - All components balanced so net reward near pad w/o landing is negative.
    """
    # ── current state ──────────────────────────────────────────────
    x  = next_obs[0]          # horizontal position relative to pad center
    y  = next_obs[1]          # vertical position (height above pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle      = next_obs[4]  # body tilt angle
    angular_vel = next_obs[5] # angular velocity
    left_contact  = next_obs[6]  # 1.0 if left leg on ground
    right_contact = next_obs[7]  # 1.0 if right leg on ground

    ox = obs[0]               # previous x
    oy = obs[1]               # previous y

    # ── 1. proximity_progress ──────────────────────────────────────
    # Reward closing distance to the pad centre. Reduced weight vs v1
    # (was 30.0, now 5.0) so it doesn't drown out landing signals.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    proximity_progress = 5.0 * (old_dist - new_dist)

    # ── 2. altitude_reward ─────────────────────────────────────────
    # Continuous reward for being at low height.  Creates the *gradient*
    # to descend that v1 was missing.  Max 0.15/step at y=0.
    altitude_reward = 0.15 / (1.0 + 5.0 * (y ** 2) ** 0.5)

    # ── 3. speed_penalty ───────────────────────────────────────────
    # Discourage high speed generally; extra weight when close to ground
    # (ground_proximity factor) so the agent slows down for landing.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_proximity = 1.0 / (1.0 + 5.0 * (y ** 2) ** 0.5)
    speed_penalty = -0.12 * speed * (1.0 + ground_proximity)

    # ── 4. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Same form as v1, slightly
    # reduced angular-vel coefficient.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 5. contact_reward ──────────────────────────────────────────
    # Reward leg-ground contact (0, 1, or 2 legs), scaled by quality:
    # low speed and upright attitude.  *Not* binary — partial credit
    # for single-leg contact provides a learning gradient.
    n_contact = left_contact + right_contact          # 0.0, 1.0, or 2.0
    contact_quality = 1.0 / ((1.0 + 3.0 * speed) * (1.0 + 2.0 * (angle ** 2) ** 0.5))
    contact_reward = 1.5 * n_contact * contact_quality

    # ── 6. fuel_penalty ────────────────────────────────────────────
    # Action 0 = no engine; actions 1,2,3 fire at least one engine.
    # Discourages unnecessary thrust and makes hovering expensive.
    fuel_penalty = -0.25 if action != 0 else 0.0

    # ── 7. step_penalty ────────────────────────────────────────────
    # Slight time pressure — makes indefinite hovering net-negative.
    step_penalty = -0.02

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        proximity_progress
        + altitude_reward
        + speed_penalty
        + attitude_penalty
        + contact_reward
        + fuel_penalty
        + step_penalty
    )

    components = {
        "proximity_progress": proximity_progress,
        "altitude_reward":    altitude_reward,
        "speed_penalty":      speed_penalty,
        "attitude_penalty":   attitude_penalty,
        "contact_reward":     contact_reward,
        "fuel_penalty":       fuel_penalty,
        "step_penalty":       step_penalty,
    }

    return float(total_reward), components
```