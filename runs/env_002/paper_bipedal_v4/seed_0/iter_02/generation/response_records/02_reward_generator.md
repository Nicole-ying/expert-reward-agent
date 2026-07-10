# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract usable signals from the 24‑dim observation
    hv = obs[2]          # horizontal velocity (forward speed)
    angle = obs[0]       # hull angle (tilt from upright)
    leg1c = obs[8]       # leg‑1 ground contact (0/1)
    leg2c = obs[13]      # leg‑2 ground contact (0/1)

    # --- 1. Bounded forward‑velocity reward ---
    # Formula: w * x / (1 + |x|)   (bounded_signal, formula library 2.2)
    # This saturates at high speeds, preventing the agent from trying to
    # exploit unbounded velocity spikes before falling.
    w_vel = 2.0
    vel_reward = w_vel * hv / (1.0 + abs(hv))

    # --- 2. Soft health gate to penalise instability ---
    # Formula: gate = 1 / (1 + k * |angle|)   (soft_health_gate, formula library 2.6)
    # The velocity reward is scaled down when the hull tilts, turning an
    # explicit stability penalty into a continuous driving‑signal.
    k_angle = 20.0
    gate = 1.0 / (1.0 + k_angle * abs(angle))
    inst_penalty = vel_reward * (1.0 - gate)   # how much of vel_reward is gated away

    # --- 3. Gait alternation bonus ---
    # Encourages a complementary contact pattern: exactly one foot on the ground.
    # Expression = leg1 + leg2 - 2*leg1*leg2, which is 1 when contacts differ, 0 otherwise.
    w_gait = 0.2
    alternation = leg1c + leg2c - 2.0 * leg1c * leg2c
    gait_reward = w_gait * alternation

    # Total reward: velocity reward gated by stability, plus gait alternation
    total_reward = vel_reward - inst_penalty + gait_reward

    components = {
        'forward_velocity': vel_reward,
        'instability_penalty': -inst_penalty,   # shown as a negative contribution
        'gait_alternation': gait_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

**task_family / dynamics_subtype**  
`locomotion_continuous_control` / `planar_bipedal_gait`（从 environment_card.md 直接读取）。

**selected reward roles**  
根据 `reward_role_decomposition` 选择以下职责：

- **forward_progress**（mandatory）：用 bounded 形式将水平速度映射为主学习信号，避免线性速度奖励导致的数值爆炸与“冲刺后再摔倒”的 exploit。  
- **stability_and_survival**（mandatory）：不采用独立的稳定性惩罚，而是使用 soft health gate 乘在前进奖励上。身体倾斜时前进奖励被平滑缩减，既维持了前进驱动力，又迫使策略保持直立。  
- **gait_symmetry_or_phase**（conditional）：通过 leg contact 信号鼓励双腿交替接触，抑制双支撑静止或单腿跳跃，提升步态协调性。  

**role‑to‑signal mapping**  
- forward_progress → `obs[2]` (horizontal_velocity)  
- stability_and_survival → `obs[0]` (hull_angle)  
- gait_symmetry_or_phase → `obs[8]` (leg1_contact) and `obs[13]` (leg2_contact)  

所有信号均在 observation space 中显式可用。

**formula operators selected**  
- `bounded_signal`（formula library 2.2）用于速度奖励：`w * hv / (1 + |hv|)`  
- `soft_health_gate`（2.6）用于稳定性约束：`gate = 1 / (1 + k * |angle|)`  
- 自定义连续量用于步态交替：`leg1c + leg2c – 2·leg1c·leg2c`（本质是 XOR 的连续形式，符合“鼓励交替接触”的语义）  

**excluded roles 及原因**  
- **energy_efficiency**：v1 阶段先让智能体学会稳定前进，能耗优化留到后续迭代，避免过早压制探索。  
- **terminal_success_reward / terminal_failure_penalty**：environment_card 明确声明 `explicit_success_flag_available: false` 且 `info` 为空，无法使用任何终止标志。  
- **curriculum_weighting / dynamic gated rewards**：env 未授权 `training_progress`，且 v1 不宜引入复杂课程。  

**与之前尝试的区别（materially different design）**  
之前尝试的结构为 `energy + forward_velocity + stability`（线性速度奖励 + 独立稳定性惩罚 + 动作能耗项）。本方案从以下两个维度改变了设计假设：  
1. **数学机制**：前进奖励改用 bounded 形式（饱和避免极端速度），稳定性通过乘性 soft gate 而非加性惩罚实现，彻底改变了时间序列上的奖励语义。  
2. **组件语义**：引入 gait alternation bonus 替代 energy penalty，引导双腿协调而非单纯抑制扭矩。  

**训练后应观察的潜在 failure modes**  
- 若 `k_angle` 过大或 `w_vel` 过小，agent 可能过于保守，选择原地站立或极慢行走以保持完美直立。  
- 步态交替奖励可能被单腿跳跃策略满足（一直接触、一腿反复离地，而另一腿拖地），需要后续结合速度或姿态进一步甄别。  
- bounded velocity 的饱和值可能限制了高速奔跑的学习上限，若后期需要更高速度需调整饱和参数或切换为其它形式。  

以上职责与 weight 组合仅作为 v1 基准线，后续迭代可根据实际 rollouts 的步态对称性、摔倒频率和速度分布进行调参或引入能耗约束。
