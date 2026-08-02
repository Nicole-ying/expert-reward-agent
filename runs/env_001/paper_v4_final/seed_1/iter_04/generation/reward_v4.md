# 设计理由
当前 agent 主要通过**持续获取着陆激励（landing_incentive）**得分，而该激励仅依赖**距离（dist）与接触标志**，不要求低速。这导致 agent 学会在目标垫上或附近保持接触但仍保持移动（滑动/弹跳），从而无法满足环境最终稳定停靠的 `body_not_awake_or_settled` 终止条件——绝大多数 episode 超时（truncated=19/20），len≈973。  
**信号缺口审计**发现观测 `obs[2] (x_velocity)` 和 `obs[3] (y_velocity)` 完全未被使用，而它们正是区分“接触垫子但未稳定”与“真正平稳着陆”的关键。  
因此本次修改在 **landing_incentive 组件**中乘入一个**连续速度衰减因子**，使该奖励同时要求**低速**：速度越低，奖励越接近原值；速度越高，奖励会被大幅抑制。这能将 agent 从单纯维持接触引导至**减速至静止**，从而触发成功终止（settled），进一步提升 score。  
本次修改不引入新的惩罚项，只调整原有主奖励组件的结构，属于 **Level 2 结构变换**，符合“每轮只改一个组件”的约束。数学形式采用 `1/(1+α·speed)` 有界衰减，α=3.0 使 speed=0.3 时因子≈0.5，产生足够梯度。  
**系数校准**：主信号 per-step ≈ 0.34，速度因子在低速区域（<0.1）因子>0.75，不会导致 reward 严重塌缩；高速区域（>1.0）因子<0.25，有效抑制滑动奖励。  
角度惩罚 active_rate 极低（1.7%）且幅度极小，保留无碍，不作修改。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Progress: distance reduction ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = dist - next_dist

    # --- Landing incentive with contact gate + speed decay ---
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    contact_gate = 0.1 + 0.9 * leg_contact

    # Speed magnitude (linear velocities)
    speed = (next_vx ** 2 + next_vy ** 2) ** 0.5
    # Continuous bounded factor: 1/(1+alpha*speed)
    # alpha=3.0 gives: speed=0.1 -> 0.77, speed=0.3 -> 0.53, speed=1.0 -> 0.25
    speed_factor = 1.0 / (1.0 + 3.0 * speed)

    w_landing = 0.5
    landing_incentive = contact_gate * w_landing / (1.0 + next_dist * 5.0) * speed_factor

    # --- Health constraint: body angle (kept as safe guard) ---
    w_angle = 0.5
    safe_angle = 0.3
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # --- Total reward ---
    total_reward = w_progress * progress + landing_incentive + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失——未使用速度观测，导致 agent 可以在目标垫上滑动仍获取高额着陆奖励；需添加速度门控使奖励与低速绑定。
- **behavior**: agent 学会了维持接触获取 ≈0.34/步的奖励，但未减速至 settle， episode 几乎全部超时（truncated 19/20），无法触发成功终止。
- **signal**: 缺少对高速状态的抑制；现有 landing_incentive 只关心距离和接触，不关心速度大小。
- **level**: Level 2
- **hypothesis**: 将速度因子乘入 landing_incentive 后，高速移动的奖励大幅缩水，迫使 agent 减速至静止以获取高奖励，从而增加 settled 触发率，缩短 length 并提升 task score。
- **risk**: 若速度因子在下降初期抑制过强，可能削弱 agent 下降动力，导致距离奖励不足；通过调整 alpha=3.0 保持低速区足够吸引力，副作用可控。