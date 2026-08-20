Now I have the full picture. Let me write the diagnosis.

---

## 1. 诊断报告

### evidence

**跨轮核心数据（contact 语义的生死循环）：**

| iter | score | term | contact 组件 | ep_sum | active | mag_share | 关键特征 |
|------|-------|------|-------------|--------|--------|-----------|---------|
| 1 | -76.56 | 1/20 | contact_bonus=50 | 0 | 0% | 0% | 死亡 |
| 2 | **63.72** | **17/20** | contact_bonus=50 | 1075.63 | 5.2% | 98.2% | **主导/exploit** |
| 3 | 21.45 | 9/20 | contact_bonus=50+速度门 | 1011.47 | 8.2% | 97.7% | 主导，分数下降 |
| 4 | -10.11 | 0/20 | contact_bonus=8 | 0 | 0% | 0% | 死亡 |
| 5 | -33.04 | 0/20 | contact_bonus=8 | 0 | 0% | 0% | 死亡 |
| 6 | **-81.80** | **0/20** | landing_approach | 0.974 | 98.3% | 22.1% | **完全移除contact语义** |

**Iter 6 的关键问题：**

- **Checkpoint 趋势**：cp1=-172.8 → cp7=**138.5（峰值）** → cp10=-67.1。训练中期发现好策略，随后崩溃。cp7→cp10 期间，goal_proximity 从 5.4 跌至 2.2，landing_approach 从 2.8 跌至 1.2。agent 学会了减少惩罚（速度从-3.3→-2.4）但牺牲了进展——经典的「多做多错，不如不做」。
- **所有 episode truncate**（1000 步跑满），0 次成功着陆，无终止。
- **每步净奖励为负**：goal_proximity=0.0018 + landing_approach=0.0010 + velocity_penalty=-0.0024 + upright_penalty=-0.0031 = **-0.0027/步**。agent 每走一步都受罚，唯一正收益路径（着陆）的引导信号太弱（landing_approach 全程只有 0.974）。
- **landing_approach 与 goal_proximity 信号重叠**：两者都在奖励「靠近目标垫」，但 landing_approach 是势能差形式（ep_sum 被 bounded 在 ~20），在 1000 步中平均每步只有 0.001，提供不了有效梯度。

**环境卡片明确声明 contact_success 为强制角色**：*"这是唯一明确的『着陆完成』信号"*。Iter 6 完全移除了这一语义，违反了强制角色要求。

**历史模式**：contact_bonus=50 主导（iter 2-3），contact_bonus=8 死亡（iter 4-5）。从未尝试过中间值（15-25），也从未在 velocity_penalty=0.3（iter 6 才降低）的软约束环境下测试 contact_bonus。

### behavior_diagnosis

Agent 陷入「消极生存」策略：由于每步净奖励为负，且着陆信号太弱（landing_approach 全程贡献仅 0.974），agent 学到的最优策略是减少移动以降低速度惩罚和姿态惩罚，宁可悬停/漂移 1000 步也不冒险下降着陆。Checkpoint 趋势完美印证：cp7→cp10 期间，速度惩罚改善但进展信号恶化——agent 在「学会不犯错」而非「学会成功」。

### signal_completeness

**缺失：contact_success（强制角色）**。Iter 6 的 landing_approach 是「下降势能」信号，不是「着陆完成」信号。它不区分「降落到垫子上双腿触地」和「降落到垫子旁边」。环境卡片明确将其列为 mandatory_roles 之一：*"当双腿同时接触垫面时，给予大规模正奖励，标记着陆成功。"*

### selected_level

**L2** — 替换 landing_approach 为 contact_bonus（不同的数学形态：连续势能差 → 稀疏条件触发）。这不是回到 iter 2/5 的旧代码，因为：(a) 使用新值 20（历史上未尝试过），(b) 在 velocity_penalty=0.3 的新软约束环境下运行。

### selected_intervention

**将 `landing_approach` 替换为 `contact_bonus`**：
- 触发条件：`next_obs[6]==1.0 and next_obs[7]==1.0`（双腿同时触地）
- 奖励值：`20.0 * centering_factor`，其中 `centering_factor = exp(-5.0 * x²)`
- 不使用速度门（避免 iter 3 的过严门控导致分数下降的覆辙；velocity_penalty 已独立处理软着陆约束）
- 保持 goal_proximity=4.0、velocity_penalty=0.3、upright_penalty=1.0 不变

### falsifiable_hypothesis

当前 agent 缺乏终端着陆激励。landing_approach 提供的是「下降就给分」的均匀信号，没有在「成功着陆」这一关键事件上形成奖励峰值。contact_bonus=20 提供明确的稀疏终端奖励：如果 agent 成功着陆（双腿触地、居中），获得 ~20 分的跳跃。结合 goal_proximity 的密集引导，agent 有动机：(a) 接近目标垫（goal_proximity 的密集奖励），(b) 完成着陆（contact_bonus 的终端奖励）。接触奖励 20 足够大以克服每步的轻微净惩罚（~500步 × -0.003 ≈ -1.5），但不至于像 iter 2 的 50 那样主导整个 reward 构成。

### expected_next_round

- **terminated > 0**：至少应出现若干次成功着陆（不再 0/20）
- **contact_bonus active_rate > 0%**：应被触发
- **contact_bonus magnitude_share < 70%**：不应主导
- **score 应回升**：目标 > 0（摆脱负分），理想情况接近或超过 iter 2 的 63.72
- **Checkpoint 趋势应单调改善**：不再出现后期崩溃

### main_risk

contact_bonus=20 可能仍不足以激励 agent 探索着陆行为（如果 goal_proximity 的引导不够强），导致 contact_bonus 再次死亡（active=0%）。若下一轮仍为 0/20 且 contact_bonus dead，则需要同时加强 goal_proximity 或引入额外的下降激励。反向风险：若 20 仍过大，可能重演 iter 2 的 exploit 模式（magnitude_share > 80%），此时需降至 12-15。

---

## 2. 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 7): landing_approach replaced with contact_bonus.
    The environment card identifies contact_success as a mandatory role —
    the only explicit "landing complete" signal. Iter 6's landing_approach
    was a continuous descent potential that never peaked at the landing event,
    providing only 0.974 ep_sum with zero successful landings (0/20).

    contact_bonus = 20 * centering_factor (moderate, between the dominating
    50 of iter 2 and the dead 8 of iter 4-5). No velocity gate — the
    velocity_penalty already enforces soft landing, and iter 3's velocity
    gate (exp(-3*speed²)) caused score collapse from 64→21 by making the
    bonus too hard to trigger.

    Velocity penalty kept at 0.3 (softened in iter 6 from 0.5) to avoid
    the "penalty wall" that prevented the agent from approaching the pad.
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

    # --- contact bonus: moderate terminal reward for dual-leg landing ---
    # Restores the mandatory contact_success role (removed in iter 6).
    # Value 20 is chosen between the dominating 50 (iter 2) and dead 8 (iter 4-5).
    # centering_factor ensures off-pad landings receive proportionally less.
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 20.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return float(total_reward), components
```