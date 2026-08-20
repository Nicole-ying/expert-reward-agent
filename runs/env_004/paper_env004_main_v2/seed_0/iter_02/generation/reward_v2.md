`evidence`：Score 2926/3800（77%），全部20条评估轨迹截断于1000步，无提前终止；speed_posture_reward为唯一正向信号（episode_sum_mean=1837.14，magnitude_share=97.8%），ang_vel_penalty贡献极小（41.64，2.2%）；k_angle=10.0使posture_factor=exp(-10·θ²)在轻微躯干倾斜时急剧衰减，有效奖励被大幅压缩。

`behavior_diagnosis`：Agent保持稳定直立姿态完整存活，但前进速度仅处于中等水平（每步奖励约1.84）。过强的姿态约束（k=10）惩罚了跳跃步态中自然的躯干振荡，导致Agent为维持近乎完美的直立而牺牲前进速度，陷入“保守行走/小跳”的次优策略。

`signal_completeness`：前进信号（speed×posture）和稳定信号（ang_vel_penalty）均已存在且可达，数学形态合理。缺少能耗惩罚，但当前失败主因是前进速度不足而非能耗过大。姿态约束的尺度（k=10）是主要瓶颈。

`selected_level`：Level 1（尺度修复）——posture_factor的指数衰减形态和符号均正确，但k_angle=10.0对需要自然躯干振荡的跳跃步态过于严厉，使有效奖励在中高速时被过度压缩。

`selected_intervention`：单一组件修改——将posture_factor中的k_angle从10.0降至5.0，其余所有组件和系数不变。

`falsifiable_hypothesis`：k=5时，在相同躯干倾斜（如0.3 rad）下posture_factor从0.41提升至0.64，Agent在训练中将探索更有力的跳跃模式以获得更高前进速度，而非过度抑制躯干运动来追求完美直立；score应向3800靠拢。

`expected_next_round`：speed_posture_reward的episode_sum_mean应上升（更高posture_factor叠加可能更高的forward_vel）；score应增加；episode_length应保持1000（不引发摔倒）；ang_vel_penalty可能随动作幅度增大而轻微上升，但占比仍应有限。

`main_risk`：姿态约束减半后，Agent可能探索过度倾斜的步态，若躯干倾角超出健康范围会触发提前终止。k=5在0.7 rad（≈40°）时posture_factor≈0.08，仍提供显著抑制，风险可控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键信号
    torso_angle = next_obs[1]          # 躯干倾角
    forward_vel = next_obs[5]          # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]        # 躯干角速度

    # 姿态因子：躯干越直立（角度接近0），因子越接近1
    k_angle = 5.0
    posture_factor = 2.718281828 ** (-k_angle * (torso_angle ** 2))

    # 主学习信号：前进速度 × 姿态因子
    #   - 前进速度为正则奖励，为负则惩罚，引导 agent 持续向前
    #   - 姿态因子在躯干倾斜时自动衰减，抑制为追求速度而过度倾斜
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    w_ang_vel = 0.05
    ang_vel_penalty = w_ang_vel * abs(torso_ang_vel)

    total_reward = speed_posture_reward - ang_vel_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty
    }

    return float(total_reward), components
```