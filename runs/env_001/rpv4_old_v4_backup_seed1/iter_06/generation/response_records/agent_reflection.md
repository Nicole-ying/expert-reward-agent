# Response Record

现在我掌握了完整画面。让我进行诊断。

---

## 1. 诊断报告

### evidence

**跨轮核心数据（contact_bonus 语义的生死循环）：**

| iter | score | term | CB ep_sum | CB active | CB mag_share | 关键变化 |
|------|-------|------|-----------|-----------|-------------|----------|
| 1 | -76.56 | 1/20 | 0 | 0% | 0% | CB=50，死亡 |
| 2 | **63.72** | **17/20** | 1075.63 | 5.2% | 98.2% | VP高度门控，GP=2.0，CB=50 |
| 3 | 21.45 | 9/20 | 1011.47 | 8.2% | 97.7% | CB加速率门控（试图修复"exploit"） |
| 4 | -10.11 | 0/20 | 0 | 0% | 0% | GP改逆距离，CB降至8 |
| 5 | -33.04 | 0/20 | 0 | 0% | 0% | GP回线性，CB保持8 |

**iter 5 checkpoint 轨迹的独特信号：**
- cp70% 得分飙至 **201.1**，contact_bonus=571（14.7% active）——agent 确实学会了着陆！
- cp80% 降至 25.1，contact_bonus=326（4.8%）
- cp90% 崩溃至 -32.1，contact_bonus 归零
- 这说明：**CB=8 在训练中途足以引导出着陆行为，但不足以稳定该行为**。一旦 exploration noise 使策略偏离着陆区，微弱的 +8 无法提供足够的梯度把策略拉回来——连续惩罚（VP≈-10, UP≈-4）主导了后续学习，驱使 agent 远离地面。

**语义打转证据：**
- 「稀疏 contact_bonus」（50→速度门控→8）已经改了 5 轮，反复在"死亡"和"主导/exploit"之间振荡。iter 2 的 CB=50 虽得分最高，但 98.2% magnitude_share 使得连续信号被淹没。后续降低 CB 量级又导致完全死亡。**这个语义已经穷尽——必须换方向。**

**当前 iter 5 信号结构问题：**
- 三个连续信号全覆盖（100% active）：GP=+4.29，VP=-10.12，UP=-3.60
- 连续净值为负：+4.29 - 10.12 - 3.60 = **-9.43/episode**
- 唯一的正终端信号（CB）死亡——agent 没有任何正向激励去接近地面
- 结果：agent 学会了在空中徘徊 1000 步，避免靠近地面触发 VP，最终被 truncate

### behavior_diagnosis

Agent 当前策略：**高空悬停/徘徊**。连续信号净值为负（约 -0.009/step），agent 通过最小化移动来最小化 VP 和 UP 惩罚，导致 len=1000 全部 truncation。GP 虽然提供接近信号，但不足以克服"接近地面 = 触发 VP"的恐惧。稀疏 CB=8 太弱，无法作为终端吸引子。

### signal_completeness

对照环境卡片中的 mandatory_roles：
- `goal_proximity` ✅ 存在，但被负信号淹没
- `soft_landing_velocity_penalty` ✅ 存在
- `upright_orientation` ✅ 存在
- `contact_success` ❌ **名义上存在但实际死亡（0% active）**——着陆完成信号缺失

**缺失的关键信号：从"接近目标"到"接触地面"之间的连续梯度。** 当前 GP 奖励任何 Euclidean 距离缩短，但 agent 在 (x≈0, y=small) 时，进一步下降的 GP 增量极小（距离对高度的偏导在低空趋于零），而 VP 在低空因 gate≈1 达到最大。这制造了一个"近地惩罚墙"：接近地面时边际收益递减、边际惩罚递增。

### selected_level

**L2** — 修改一个组件的数学形态。不再使用稀疏二值 contact_bonus（已穷尽），而是引入一个**连续的"着陆区接近"势能奖励**，替代 contact_bonus 的语义角色。

### selected_intervention

**将 `contact_bonus`（稀疏二值）替换为 `landing_approach`（连续势能差）。**

公式：
```
phi(x, y) = -y * exp(-5 * x²)
landing_approach = w * (phi(next) - phi(prev))
```

核心特性：
- 当 agent **水平居中且正在下降**时给予最强正奖励（x≈0, Δy<0 → 大正值）
- 当 agent 偏离中心时，下降奖励被指数衰减抑制（防止在错误位置着陆）
- 当 agent 水平居中但上升时给予负奖励（Δy>0 → 负值，抑制逃离行为）
- **连续梯度**：从任意位置都能感知到"向(0,0)移动"的奖励梯度，不存在死区
- **不会 exploit**：每步奖励严格等于势能差，累计值受限于初始和最终势能之差

辅助调整：velocity_penalty 权重从 0.5→0.3，降低"近地惩罚墙"，让新的 landing_approach 信号能有效引导最终下降。

### falsifiable_hypothesis

将稀疏终端奖励替换为连续势能差后，agent 在训练的每一时刻都能感知到"下降+居中"的梯度。与 iter 2 的稀疏 CB=50 不同，landing_approach 不会在着陆瞬间产生量级爆炸（98% share），因为它均匀分布在所有下降步骤中。与 iter 5 的 CB=8 不同，landing_approach 不会死亡，因为它在任何位置都能提供非零梯度。cp70% 曾出现的"学会着陆后崩溃"现象应被消除，因为即使策略偏离着陆区，landing_approach 的连续梯度也能将其拉回。

### expected_next_round

- `landing_approach` active_rate 应接近 100%（任何时候移动都有信号）
- `landing_approach` ep_sum 应为正值且 magnitude_share 在 20-40%（不会主导）
- `velocity_penalty` magnitude_share 应降至 35-45%
- `goal_proximity` 保持活跃
- terminated 应 > 0（至少有几集成功着陆）
- score 应显著改善（目标是正值，理想 > 0）
- checkpoint 趋势：不应再出现 cp70% 峰值后崩溃的模式

### main_risk

landing_approach 的势能形态可能与 goal_proximity 产生冗余——两者都奖励向 (0,0) 移动。如果权重配比不当，可能导致 agent 过度关注垂直下降而忽略水平纠正（或反之）。通过 centering_factor `exp(-5x²)` 解耦：goal_proximity 主导远距离水平接近，landing_approach 主导近地垂直下降。权重 2.0 相对于 GP 的 4.0 是合理的从属关系。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 6): contact_bonus replaced with landing_approach —
    a continuous potential-based reward that provides gradient all the way
    to the surface. Unlike the sparse binary contact_bonus (which oscillated
    between dead and 98% magnitude-share across iters 1-5), this signal:
      - rewards descending when horizontally centered (x≈0)
      - suppresses descent reward when far from center (via exp(-5x²))
      - provides uniform per-step gradient at all altitudes (no dead zones)
      - cannot explode: total per-episode reward is bounded by the potential
        difference between start and goal states.

    Velocity penalty weight reduced 0.5→0.3 to soften the "near-ground
    penalty wall" and allow the new landing_approach signal to effectively
    guide the final descent phase.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.3 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- landing approach: continuous surface-proximity potential ---
    # phi(x, y) = -y * exp(-alpha * x²)
    #   High potential when low (small y) AND centered (small |x|).
    #   The exp(-5x²) centering gate suppresses descent reward when the
    #   agent is far from the pad, preventing it from diving at wrong spots.
    # Reward = phi(next) - phi(prev):
    #   + when descending while centered (both factors improve)
    #   + when centering while low (x-improvement matters more near surface)
    #   - when rising or moving away from center
    prev_h = max(obs[1], 0.0)
    next_h = max(next_obs[1], 0.0)
    alpha = 5.0
    prev_phi = -prev_h * (2.718281828 ** (-alpha * obs[0]**2))
    next_phi = -next_h * (2.718281828 ** (-alpha * next_obs[0]**2))
    landing_approach = 2.0 * (next_phi - prev_phi)

    total_reward = goal_proximity + vel_penalty + upright_penalty + landing_approach

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "landing_approach": landing_approach
    }
    return total_reward, components
```
