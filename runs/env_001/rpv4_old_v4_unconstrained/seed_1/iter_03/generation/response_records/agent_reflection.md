# Response Record

现在我已拥有完整的图景。以下是诊断和 v3 奖励函数。

---

## 诊断报告

### evidence

| 来源 | 关键数据 |
|---|---|
| iter_1 得分 | -2.98，长度 813 → agent 悬停，从不降落。soft_landing_bonus 活跃率 0.3%（死组件），proximity_progress 幅度占比 82.9% |
| iter_2 得分 | **-125.32**，长度 68.35，20/20 提前终止（score < -50）→ agent 快速坠毁 |
| iter_2 检查点趋势 | 得分从 -116 单调恶化到 **-125**：agent 在**学习更快坠毁**以最小化负奖励累积 |
| iter_2 speed_penalty | ep_sum_mean = **-11.35**，幅度占比 **46.3%**——在奖励中绝对主导地位 |
| iter_2 contact_reward | 活跃率仅 **3.7%**——着陆几乎从未发生 |
| iter_2 每步净值 | 大致为：+0.081（接近） + 0.035（高度） - 0.166（速度） - 0.031（姿态） - 0.02（步数） + … ≈ **每步约 -0.10**。最优策略：立即终止。 |

### behavior_diagnosis

v2 奖励函数存在根本性缺陷：**速度惩罚应用于全局**（`-0.12 * speed * (1.0 + ground_proximity)`），使得整条轨迹的每步净奖励为负。agent 学到的是：存活越久，累积的负奖励越多。因此最优策略变为**尽快坠毁**——这也正是检查点趋势所显示的：得分从 cp10% 的 -116 下降到 cp100% 的 -125，说明 agent 在训练过程中越来越擅长更快地终止。

v1（悬停 813 步）与 v2（68 步内坠毁）之间的"中间地带"完全缺失。v3 必须创造一种结构，使得：

1. **接近阶段净奖励为正**——agent 想要存活并抵达平台
2. **速度惩罚仅在地面附近生效**——高速下降是高效的，但需要为着陆减速
3. **着陆奖励可达成且具有吸引力**——一个清晰的终端吸引盆地

### 干预策略

**移除**：全局速度惩罚、步数惩罚（鼓励提前终止）。

**重构速度惩罚**：仅在地面附近（`near_ground` 因子）生效，系数大幅降低。高空时速度不受惩罚——快速下降是高效的。

**增加下降激励**：显式奖励向下的每一步（`descent_incentive`），作为对接近项（potential-based）的补充，确保即使在下降初期也能获得正奖励。

**强化着陆奖励**：在着陆质量基础上增加系数，使其成为强劲的终端吸引子。使用连续 `n_contact`（0/1/2）提供部分梯度。

**降低燃油惩罚**：从 -0.25 降至 -0.05——agent 需要自由使用引擎进行控制，尤其在制动阶段。

---

## v3 奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3 — fixed net-negative-per-step bug from v2.

    Core changes from v2:
    - speed_penalty is now conditional on near_ground proximity (was global).
      Fast descent at high altitude is efficient, not penalised.
    - Added descent_incentive: explicit per-step reward for moving downward,
      supplementing the potential-based approach_reward.
    - Reduced fuel_penalty from -0.25 to -0.05 so the agent dares to use engines.
    - Removed step_penalty entirely — it incentivised early termination.
    - Boosted landing_reward coefficient to create a strong terminal attractor.
    - All coefficients rebalanced so the approach phase is net-positive.
    """

    # ── current state ──────────────────────────────────────────────
    x  = next_obs[0]          # horizontal position relative to pad centre
    y  = next_obs[1]          # vertical position (height above pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle      = next_obs[4]  # body tilt angle
    angular_vel = next_obs[5] # angular velocity
    left_contact  = next_obs[6]  # 1.0 if left leg on ground
    right_contact = next_obs[7]  # 1.0 if right leg on ground

    ox = obs[0]               # previous x
    oy = obs[1]               # previous y

    # ── 1. approach_reward ─────────────────────────────────────────
    # Potential-based shaping: reward closing Euclidean distance to pad.
    # Modest coefficient (was 30.0 in v1, 5.0 in v2) — enough to guide,
    # not so large it drowns out landing signals.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    approach_reward = 3.0 * (old_dist - new_dist)

    # ── 2. descent_incentive ───────────────────────────────────────
    # Explicit reward for each unit of downward progress.
    # Supplements approach_reward: even if horizontal drift dilutes the
    # distance signal, descending is always rewarded.
    descent_incentive = 0.6 * max(0.0, oy - y)

    # ── 3. altitude_bonus ──────────────────────────────────────────
    # Small bonus for staying at low altitude.  Creates a gentle
    # "basin" near the pad without being large enough to cause hovering.
    altitude_bonus = 0.02 / (1.0 + 3.0 * (y ** 2) ** 0.5)

    # ── 4. speed_penalty ───────────────────────────────────────────
    # Penalise high speed ONLY when close to the ground, where it risks
    # a crash.  At high altitude, speed is unpenalised — fast efficient
    # descent is actually desirable.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    near_ground = 1.0 / (1.0 + 8.0 * (y ** 2) ** 0.5)   # ~1 at y=0, ~0 when high
    speed_penalty = -0.08 * speed * near_ground

    # ── 5. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating throughout the trajectory.
    # Moderate coefficients — strong enough to matter, not so strong
    # that the agent fears using orientation engines.
    attitude_penalty = -0.2 * (angle ** 2) - 0.05 * (angular_vel ** 2)

    # ── 6. landing_reward ──────────────────────────────────────────
    # Reward leg-ground contact (0, 1, or 2 legs), scaled by quality:
    # low speed, upright attitude, and proximity to pad centre.
    # Continuous n_contact (not binary product) gives partial credit
    # for single-leg contact, providing a learning gradient.
    n_contact = left_contact + right_contact          # 0.0, 1.0, or 2.0
    landing_quality = 1.0 / (
        (1.0 + 4.0 * speed)
        * (1.0 + 3.0 * (angle ** 2) ** 0.5)
        * (1.0 + 2.0 * (x ** 2) ** 0.5)
    )
    landing_reward = 0.8 * n_contact * landing_quality

    # ── 7. fuel_penalty ────────────────────────────────────────────
    # Small penalty for engine use (actions 1,2,3).
    # Kept intentionally light so the agent is willing to use engines
    # for braking and attitude control.  -0.05 vs v2's -0.25.
    fuel_penalty = -0.05 if action != 0 else 0.0

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        approach_reward
        + descent_incentive
        + altitude_bonus
        + speed_penalty
        + attitude_penalty
        + landing_reward
        + fuel_penalty
    )

    components = {
        "approach_reward":   approach_reward,
        "descent_incentive": descent_incentive,
        "altitude_bonus":    altitude_bonus,
        "speed_penalty":     speed_penalty,
        "attitude_penalty":  attitude_penalty,
        "landing_reward":    landing_reward,
        "fuel_penalty":      fuel_penalty,
    }

    return float(total_reward), components
```
