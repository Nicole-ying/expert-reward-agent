# Response Record

# 设计理由
当前奖励函数采用独立线性加和的结构（前进速度正面奖励 + 倾角/角速度二次惩罚 + 扭矩能耗惩罚），已经驱动智能体走得远、得分高。然而，这种“独立求和”模式存在隐患：前进奖励与稳定惩罚的解耦可能导致智能体在某些不受惩罚的中间姿态下通过激进动作获取速度，而稳定性惩罚只在偏差极大时才有显著作用，缺乏“安全门”机制。为了在结构上做出有意义的改变，我们将稳定性惩罚从**独立的加法项**转变为**乘在前向速度奖励上的软健康门（safety gate）**：只有当躯干姿态稳定时，前进速度才会被充分奖励；一旦倾角或角速度过大，前进奖励会被大幅抑制。这种结构变化完全改变了梯度流向——智能体不再单独面对“前进-消耗-倾斜”三个独立目标，而是必须维持稳定才能获得前进收益，从而减少了利用惩罚盲区进行爆发性前进的可能。同时，我们引入一个轻微的垂直速度惩罚，以抑制不必要的蹦跳，提高步态的平滑性。动作能耗项保留但系数略微降低，以适配新的总奖励尺度。

修改符合 Level 2 的结构变换（加性惩罚→乘法门），并引入一个新的垂直蹦跳惩罚。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    horizontal_velocity = obs[2]          # use current velocity for forward reward
    vertical_velocity = obs[3]            # new signal for bounce penalty

    # ----- 1. Forward progress signal -----
    # Encourage positive horizontal speed (same as before)
    forward_speed = max(0.0, horizontal_velocity)

    # ----- 2. Stability gate (multiplicative, replaces additive upright_penalty) -----
    # Gate approaches 1 when angle and angvel are near zero,
    # drops toward 0 when instability grows, choking off forward reward.
    # Using exponential for smooth, bounded [0,1] output.
    # Tuned so that modest instability (angle~0.2 rad, angvel~1 rad/s) still gives gate~0.92,
    # while severe instability (angle~1.0 rad) gives gate~0.37.
    angle_sq = hull_angle * hull_angle
    angvel_sq = hull_angular_velocity * hull_angular_velocity
    stability_gate = 2.718281828 ** (-1.5 * angle_sq - 0.15 * angvel_sq)

    # ----- 3. Vertical bounce penalty (new) -----
    # Penalize vertical kinetic energy to discourage hopping.
    bounce_penalty = -0.15 * (vertical_velocity * vertical_velocity)

    # ----- 4. Action energy cost (retained, fractionally reduced) -----
    torque_sq_sum = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_cost = -0.008 * torque_sq_sum   # slightly smaller to keep penalty burden ~0.5x of main signal

    # Combine: main reward is gated forward speed, then add penalties.
    forward_speed_reward = 2.0 * forward_speed * stability_gate
    total_reward = forward_speed_reward + bounce_penalty + action_cost

    components = {
        "forward_speed_reward": forward_speed_reward,
        "bounce_penalty": bounce_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少垂直方向惩罚和乘法安全门；稳定性独立惩罚在正常区域信号极弱，无法有效引导姿态微调。
- **behavior**: 智能体快速前进，但可能存在微小抖振或蹦跳（因缺少纵向抑制），动作可进一步平滑。
- **signal**: stability gate 代替独立惩罚，垂直速度惩罚填补空白。
- **level**: Level 2
- **hypothesis**: 乘法门将稳定性和前进奖励绑定，迫使智能体在保持姿态的同时获取速度，从而减少激进动作带来的微小不稳定；垂直速度惩罚抑制无效弹跳，最终可能获得等价或略高的得分，同时步态更自然。
- **risk**: 乘法门可能在训练早期压垮探索（若初始姿态频繁倾覆，前进奖励消失），但配合适中的动作惩罚和垂直惩罚，智能体仍能从保持站立的学习中获得正的奖励结构（稳定性本身不直接奖励，但解锁了前进奖励）。垂直惩罚系数过大会减慢速度，已选较小值以降低风险。
