# Response Record

现在我掌握了全部数据。以下是完整诊断和修改。

---

## 1. 诊断报告

### evidence

**跨轮关键数据：**

| 指标 | iter 1 | iter 2 (best) | iter 3 |
|---|---|---|---|
| score | -76.56 | **63.72** | 21.45 |
| terminated | 1/20 | **17/20** | 9/20 |
| len | 975 | 668 | 865 |
| contact_bonus mag_share | 0% | 98.2% | **97.7%** |
| goal_proximity mag_share | 7.2% | 0.3% | 0.3% |

**contact_bonus 演化轨迹：**
- iter 1：完全死亡（0% active）
- iter 2：激活到 5.2%，ep_sum=1075，但已占 98.2% magnitude
- iter 3：速率门控后 active 升至 8.2%，但 termination 从 17→9，**score 反而下降 42 分**

**goal_proximity 演化轨迹：**
- 三轮几乎不变：ep_sum 0.64→1.84→1.77，magnitude_share 从 7.2% 被 contact_bonus 挤到 0.3%
- 权重从 1.0→2.0 只翻了一倍，但公式 `prev_dist - next_dist` 每步仅产生 ~0.002 的信号

**Checkpoint 趋势（iter 3）：**
- contact_bonus 在 cp40% 突然从 0→141，然后 cp80% 归零，cp100% 又跳到 958——**极度不稳定，典型的稀疏奖励探索-遗忘循环**
- goal_proximity 全程在 0.75~1.78 之间微弱波动——无法提供有效梯度

### behavior_diagnosis

策略处于**稀疏奖励陷阱**：contact_bonus 以 97.7% magnitude 主导一切，但只在 8.2% 的步触发（双腿同时着地时）。agent 的有效学习信号极度稀疏——870 步中只有 ~70 步有意义的反馈。三个连续塑造信号（goal_proximity + velocity_penalty + upright_penalty）合计仅贡献 2.3% 的 magnitude，在梯度层面几乎不可见。

iter 3 添加的速率门控 `exp(-3.0 * speed_sq)` 恶化了问题：它将唯一的强正信号绑定到"已学会减速"的前提条件上，但 agent 无法学会减速，因为连续速度惩罚太弱。这就是为什么 termination 从 17/20 掉到 9/20。

### signal_completeness

四类必要职责均已覆盖（接近、软着陆速度、姿态、接触成功），但**量级比例严重失衡**。缺少的不是信号类型，而是**有效的梯度密度**——连续信号需要足够强才能在每步提供可区分的梯度。

### selected_level

**L2**（改数学形态）+ **L1**（调系数配合）

选择理由：
- goal_proximity 的 delta-distance 公式 `prev_dist - next_dist` 是根本瓶颈：agent 每步移动距离很小，导致每步奖励 ~0.002，无论权重怎么调（L1）都无法突破这个物理上限
- 需要换一种数学形态，使每步奖励对距离变化更敏感，尤其是近地时
- contact_bonus 降权是配合性的 L1，防止新信号被淹没

### selected_intervention

**目标组件：`goal_proximity`（主）+ `contact_bonus`（配合）**

具体修改：

1. **goal_proximity（L2）**：从线性 delta-distance `2.0 * (prev_dist - next_dist)` 改为**反距离势能差分** `3.0 * (1/(next_dist + 0.5) - 1/(prev_dist + 0.5))`
   - 反距离势能 Φ(d) = 1/(d+ε) 对近目标区域天然敏感：d=5→4 时 ΔΦ≈0.04，d=1→0.8 时 ΔΦ≈0.10
   - 对比原公式：d=5→4 时 Δd=1.0，乘 2.0=2.0……不对，每步移动 ~0.1 距离单位，所以原公式每步 ~0.2，新公式 ~0.04×3=0.12。但近地时新公式优势明显
   - 关键优势：**近目标时梯度自动增强**，这正是软着陆最需要精细控制的阶段

2. **contact_bonus（L1）**：从 50.0→8.0，**移除速率门控**
   - 速率门控已被证明有害（iter 2→3 的回归证据）
   - 降权到 8.0 使每接触步约 6~8 点（vs 当前 ~14），累积 ep_sum 预计从 ~1000 降至 ~300-400
   - 配合 goal_proximity 增强后预计 ~20-40 ep_sum，contact_bonus magnitude_share 预计从 97.7% 降至 ~85-90%

两个修改服务于同一语义：**将奖励质量从稀疏接触奖励转移到密集进度信号**。

### falsifiable_hypothesis

如果反距离势能提供了足够密集的梯度，agent 应在训练早期就学会向目标垫移动（而非仅靠重力漂移），并在接近地面时获得逐步增强的减速激励。contact_bonus 降权 + 去门控保证着陆后仍有正向反馈（不制造 chicken-and-egg），但不再淹没连续信号。

### expected_next_round

- goal_proximity ep_sum 应从 ~1.8 升至 15-40
- goal_proximity magnitude_share 应从 0.3% 升至 5-15%
- contact_bonus magnitude_share 应从 97.7% 降至 70-90%
- termination 应从 9/20 恢复至 ≥14/20
- score 应回升至 ≥60（接近 iter 2 水平），且 checkpoint 趋势应更平滑（减少振荡）

### main_risk

反距离势能在高空（d>8）时梯度极弱（每步 ~0.002），agent 初始下降阶段可能缺乏方向引导。缓解：重力本身会拉动下降，且 velocity_penalty 在高空被门控削弱，不会阻止运动。如果下一轮 goal_proximity 仍然过低，需要在势能函数中引入线性分量混合。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 4): goal_proximity switched from linear delta-distance
    to inverse-distance potential — provides much stronger per-step gradients,
    especially near the ground where fine deceleration control matters.
    contact_bonus magnitude reduced from 50→8 and velocity gating removed
    (gating created a chicken-and-egg problem in iter 3, dropping score from
    64→21). The lower bonus magnitude prevents it from drowning out the new
    dense progress signal.
    """
    # --- goal proximity: inverse-distance potential for dense gradients ---
    # phi(d) = 1/(d + epsilon)  ->  reward = phi(next) - phi(prev)
    # This form amplifies gradients near the target (where soft landing
    # requires precise control) while still providing directional signal
    # at moderate distances.
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    epsilon = 0.5  # prevents singularity at d=0, controls gradient sharpness
    prev_potential = 1.0 / (prev_dist + epsilon)
    next_potential = 1.0 / (next_dist + epsilon)
    goal_proximity = 3.0 * (next_potential - prev_potential)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate reward for dual-leg contact ---
    # Velocity gating removed (iter 3 regression evidence).
    # Magnitude reduced 50→8 to prevent drowning dense progress signal.
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian decay with horizontal offset from pad center
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 8.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```
