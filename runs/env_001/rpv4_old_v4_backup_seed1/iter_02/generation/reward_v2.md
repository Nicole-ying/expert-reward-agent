现在我已经掌握了所有必要信息。让我进行诊断并编写修改后的奖励函数。

---

## 1. 诊断报告

### evidence
- **最终得分**: -76.56（目标 200），仅 1/20 成功着陆，19/20 被截断（~975 步后超时）
- **contact_bonus**: active_rate=0.0%，完全死亡——agent 从未实现双腿同时触地
- **velocity_penalty**: ep_sum=-12.56，magnitude_share=71.7%，主导了整个奖励信号
- **goal_proximity**: ep_sum=+0.635，magnitude_share 仅 7.2%，接近激励比速度惩罚弱 **~20 倍**
- **Checkpoint 轨迹**: contact_bonus 在训练中途多次出现高峰（cp10%: 40.79, cp40%: 63.25, cp70%: 51.36），但 cp80%→100% 全部归零——agent 探索到了接触，但奖励结构无法稳定强化这一行为
- **整体得分趋势**: cp10%=-117.6 → cp100%=-80.4，有改善但远未达标

### behavior_diagnosis
Agent 学会了"悬停保命"策略：最小化速度以避免主导性的 velocity_penalty，几乎不向目标移动（全 episode 仅累计接近 0.635 距离单位），从不尝试着陆。velocity_penalty 在所有高度一视同仁地惩罚运动，而 goal_proximity 太弱无法驱动接近行为。

### signal_completeness
四个必要职责均已覆盖但权重严重失衡：
- `goal_proximity`（接近）：存在但被压制
- `soft_landing_velocity_penalty`（软着陆）：存在但无高度区分，扼杀了接近行为
- `upright_orientation`（姿态）：正常工作
- `contact_success`（着陆成功）：存在但稀疏，因 agent 从不接近地面而从未触发

缺失的关键信号：**高度感知的速度惩罚**——高处应允许自由移动以接近目标，低处应严格限速。

### selected_level
**L2**：修改 velocity_penalty 的数学形态，从均匀惩罚变为高度门控惩罚。同时 L1 调高 goal_proximity 权重作为配套（同一语义方向）。

### selected_intervention
**目标组件**: `velocity_penalty`

**具体修改**:
1. **L2**: 将速度惩罚乘以高度门控因子 `exp(-height / h_scale)`。在高空（height 大），门控因子趋近 0，速度惩罚几乎消失，agent 可自由移动接近目标。近地（height→0），门控因子→1.0，速度惩罚完全生效，强制软着陆。
2. **L1（配套）**: goal_proximity 权重从 1.0 → 2.0，增强接近驱动力。

两项修改服务于同一语义：**解除"悬停保命"困局，让 agent 敢移动、敢接近**。

### falsifiable_hypothesis
速度惩罚的高度门控消除了"移动即受罚"的矛盾。Agent 在高空不再因水平/垂向移动被重罚，goal_proximity 的接近信号（已加强至 2.0）将引导 agent 向目标垫下降。随着高度降低，门控逐步收紧速度惩罚，在接触前自然过渡到软着陆模式。

### expected_next_round
- `goal_proximity` 的 ep_sum 应显著增长（从 0.635 → 预计 3-8），因为 agent 不再被速度惩罚压制
- `velocity_penalty` 的 ep_sum 绝对值可能先增后减：agent 先开始移动（增大），近地减速（缩小）
- `contact_bonus` 的 active_rate 应从 0% 上升至 >5%，因为 agent 开始接近地面
- 终止率应改善（terminated > 1/20），episode 长度应缩短
- 得分应在 -30 到 +50 范围内（仍可能为负，因为着陆信号仍稀疏）

### main_risk
agent 可能在近地时减速不够快——门控从 0.37（height=2）到 1.0（height=0）的过渡区较窄。如果 agent 下降速度过快，可能在门控尚未完全收紧时就以高速撞击地面。可通过调整 `h_scale` 参数（当前 2.0）或后续增加近地速度急刹组件来缓解。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 2): velocity penalty is now height-gated via
    exp(-height / 2.0). At high altitude, velocity penalty is relaxed,
    allowing the agent to move freely toward the target. Near the ground,
    the penalty ramps up to full strength for soft landing.
    Goal proximity weight increased to 2.0 to strengthen approach incentive.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 2.0 * (prev_dist - next_dist)   # delta reward (weight raised from 1.0)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)  # clamp to avoid amplification below pad
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: reward simultaneous leg contact, modulated by horizontal centering ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian-like decay with horizontal offset
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 50.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```