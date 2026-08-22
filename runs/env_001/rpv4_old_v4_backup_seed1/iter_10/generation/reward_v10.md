现在我已经掌握了全部历史。让我写诊断报告。

---

## 1. 诊断报告

### evidence

**核心数据（iter 9 vs best iter 2）：**

| metric | iter 9 (当前) | iter 2 (best) |
|---|---|---|
| score | 3.04 | 63.72 |
| terminated | 2/20 | 17/20 |
| 主导组件 | soft_landing_envelope (98.8% mag) | contact_bonus (98.2% mag) |
| 主导组件 active_rate | 100.0% | 5.2% |
| 主导组件 ep_sum | 1670.9 | 1075.6 |

**soft_landing_envelope 的 exploit 证据（checkpoint 轨迹）：**
- cp10%: 24 → cp40%: 710 → cp100%: 1581（持续增长，agent 在学习最大化它）
- 同期 score: -209 → -33 → 14（几乎不动）
- **结论：agent 在学习最大化 envelope，而不是学习着陆。100% active_rate + 98.8% magnitude_share = 完全主导训练信号。**

**contact_bonus 的 9 轮生死循环：**
- 在 iter 2/3 触发时（5-8% active），score=64/21，terminated=17/9
- 在其他 5 轮完全 dead（0% active），score 为负
- 稀疏二元信号不可靠：要么死、要么主导，agent 无法获得指向它的梯度

**iter 6 的 landing_approach（delta 型）回顾：**
- `phi = -h * exp(-5x²)`，仅含高度+居中，不含速度/姿态
- ep_sum=0.97，score=-81.80，0/20 terminated
- 失败原因：信号太弱（scale=2.0）、不包括速度和姿态、允许负 delta 惩罚探索

### behavior_diagnosis

Agent 当前策略：**在近地高度（h≈0-0.5）低速徘徊，收集 soft_landing_envelope 的持续奖励（~1.73/step），但不实际着陆。** 18/20 episode 因 truncation（步数上限）终止，仅 2 次偶然成功。着陆信号（envelope）提供了"滞空即奖励"的错误引导——只要保持低高度、低速度、居中和直立，就能每步收取 ~1.7 奖励，无需双腿接触地面。

### signal_completeness

四个必要职责的覆盖情况：
- `goal_proximity`：✅ 存在但被淹没（magnitude_share=0.4%），delta-distance 提供均匀梯度
- `soft_landing_velocity_penalty`：✅ 存在但被淹没（0.5%），高度门控合理（k=1.5）
- `upright_orientation`：✅ 存在但被淹没（0.3%）
- `contact_success`：❌ **缺失。** soft_landing_envelope 不要求任何腿接触——它完全是状态评估，不是着陆信号。这是本次需要修复的核心缺口。

### selected_level

**L2** — 修改一个组件的数学形态。将 `soft_landing_envelope` 从"绝对状态奖励"改为"状态改善量奖励（delta quality）"。不引入新组件，不调整其他组件权重。

### selected_intervention

**将 `soft_landing_envelope` 替换为 `landing_approach`（delta 型质量改善奖励）。**

具体变化：
- **旧公式**（iter 9）：`soft_landing_envelope = 3.0 * exp(-h/0.3) * exp(-5x²) * exp(-(vx²+vy²))` — 绝对状态值，每步发放
- **新公式**：`landing_approach = 20.0 * max(0, quality(next) - quality(prev))` — 改善量，只在状态变好时发放
- **quality 函数** = `exp(-h/0.3) * exp(-5x²) * exp(-(vx²+vy²)) * exp(-|angle|)` — 四维全面评估（比 iter 6 多了速度+姿态因子）

为什么与 iter 6 的 landing_approach 不同：
- iter 6：`phi = -h*exp(-5x²)`，仅高度+居中，scale=2.0，允许负delta
- 本轮：四维 quality（含速度、姿态），scale=20.0，`max(0, delta)` 只奖不罚

### falsifiable_hypothesis

delta 型奖励的每 episode 总收益有理论上限（≈20），agent 无法通过滞空来无限累积。稳态（悬停）时 delta=0 无奖励，agent 必须持续改善状态才获得正反馈。这迫使 agent 完成下降→接触→稳定的完整着陆序列，而非在中途悬停。着陆瞬间的 quality 跃升（h→0, v→0, contact 使状态骤变）会提供集中的正 delta，形成可学习的着陆梯度。

### expected_next_round

| 指标 | 当前 (iter 9) | 预期 (iter 10) |
|---|---|---|
| landing_approach ep_sum | (soft_landing_envelope: 1670) | 5-20 |
| landing_approach active_rate | 100% | 60-90%（只在改善时激活） |
| landing_approach magnitude_share | 98.8% | < 30% |
| goal_proximity magnitude_share | 0.4% | 10-30% |
| velocity_penalty magnitude_share | 0.5% | 10-25% |
| score | 3 | 20-60 |
| terminated | 2/20 | 5-12/20 |

### main_risk

**改善信号可能太弱。** quality 从起始≈0 到着陆≈0.7，总改善=0.7×20=14，分摊到 200+ 步中每步 ≈0.07。若 agent 无法感知如此小的梯度，可能退化为无着陆信号的随机游走（类似 iter 4/5/8 的 contact_bonus dead 状态）。缓解：quality 的四维结构确保下降、居中、减速、扶正四个动作各自产生独立的正 delta 贡献，合计梯度密度应足够。若下一轮 score 仍 < 10 且 terminated=0，应考虑 L3（更换整体框架，如加入 action-based fuel penalty 改变探索分布）。

---

## 2. 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 10): soft_landing_envelope (absolute per-step reward,
    exploited via hovering at 98.8% magnitude share) replaced with
    landing_approach — a delta-quality reward that only pays when the
    landing state actually improves.

    landing_approach = 20.0 * max(0, quality(next) - quality(prev))
    where quality = exp(-h/0.3) * exp(-5x^2) * exp(-(vx^2+vy^2)) * exp(-|angle|)

    This is bounded (~20 max per episode), cannot be exploited by hovering
    (steady state gives zero delta), and provides gradient across all four
    landing dimensions (height, centering, speed, uprightness).

    Unlike iter 6's landing_approach (which used only height+centering in a
    linear potential with scale 2.0 and negative deltas), this version:
      - includes velocity and angle in the quality function
      - uses scale 20.0 for stronger per-step signal
      - uses max(0, delta) to avoid penalizing exploration regression
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 at ground, decays smoothly with height (k=1.5)
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 1.5)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- landing approach: delta-quality reward ---
    # Quality function: multiplicative envelope peaking at 1.0 for the
    # ideal landing state (h=0, x=0, v=0, angle=0). Each factor in (0,1].
    def landing_quality(h, x, vx, vy, angle):
        h_safe = max(h, 0.0)
        h_factor = 2.718281828 ** (-h_safe / 0.3)
        x_factor = 2.718281828 ** (-5.0 * x**2)
        v_factor = 2.718281828 ** (-(vx**2 + vy**2))
        a_factor = 2.718281828 ** (-abs(angle))
        return h_factor * x_factor * v_factor * a_factor

    curr_quality = landing_quality(
        next_obs[1], next_obs[0], next_obs[2], next_obs[3], next_obs[4]
    )
    prev_quality = landing_quality(
        obs[1], obs[0], obs[2], obs[3], obs[4]
    )

    # Reward only improvement (delta > 0). Steady-state hovering gives zero.
    # Scale 20.0: max episode total ~20 (quality range ~0 -> ~1).
    landing_approach = 20.0 * max(0.0, curr_quality - prev_quality)

    total_reward = goal_proximity + vel_penalty + upright_penalty + landing_approach

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "landing_approach": landing_approach
    }
    return float(total_reward), components
```