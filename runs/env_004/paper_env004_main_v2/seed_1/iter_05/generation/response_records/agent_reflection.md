# Response Record

1. `evidence`：V4将w_fwd2从0.1提至0.25后score从3260降至3038，forward_progress从~1.80略升至~2.04但总回报反而下降，所有episode均truncated（1000步无跌倒），表明agent陷入中等速度稳态，更强的二次型未帮助探索更快步态。
2. `behavior_diagnosis`：agent已掌握稳定存活，以约2.04 m/s的中等速度持续前进，无法发现更快步态以缩小与target 3800的差距；二次型加成提高了原地速度但未能提供有效的边际探索梯度。
3. `signal_completeness`：前进主信号和姿态约束完备且可达；缺失的是帮助agent从“中等速度稳态”跃迁到“高速步态”的探索性梯度——持续状态型加成在稳态附近梯度虽存在但不足以引导策略跨过局部最优。
4. `selected_level`：Level 2，触发条件为Level 1系数调整（w_fwd2 0.1→0.25）已使尺度特征改变但外部得分未实质改善反而下降，同一骨架家族已迭代2轮且未刷新best，按规则须进行有方向的数学结构变换。
5. `selected_intervention`：将`forward_speed_bonus`从状态值型二次项（v_fwd²）替换为状态改善型加速度奖励`acceleration_bonus = w_accel * max(0, next_obs[5] - obs[5])`，对应`state_to_improvement`变换。
6. `falsifiable_hypothesis`：加速度奖励在蹬地推离阶段直接奖励速度增量，能为“更强推离→更快步态”提供局部信用分配；与线性forward_progress结合后，agent既有动力探索更高速度（通过加速度奖励），又有动力维持高速（通过forward_progress），应突破中等速度平台。
7. `expected_next_round`：acceleration_bonus应主要在推离步触发（active_rate显著低于forward_progress），forward_progress episode_sum_mean应上升，score应刷新best（>3260），episode仍保持1000步truncated。
8. `main_risk`：加速度奖励可能诱导急促、不稳定的推离动作，若姿态惩罚尺度不足以约束，可能出现速度波动增大甚至偶发跌倒；若active_rate过高（每步都触发），说明agent在抖动而非有效蹬地。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V5 reward: Replace quadratic forward speed bonus with acceleration reward.
    Uses state_to_improvement: reward speed increases (delta-v) rather than
    rewarding the speed state itself, to help the agent escape the moderate-speed
    plateau and discover faster hopping gaits.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_accel = 1.0        # acceleration bonus for speed increases
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Acceleration bonus: reward increases in forward speed
    # Uses obs[5] (pre-step velocity) and next_obs[5] (post-step velocity)
    delta_v = next_obs[5] - obs[5]
    acceleration_bonus = w_accel * max(0.0, delta_v)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + acceleration_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "acceleration_bonus": acceleration_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components
```
