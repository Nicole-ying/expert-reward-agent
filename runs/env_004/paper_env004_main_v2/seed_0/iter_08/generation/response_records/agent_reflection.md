# Response Record

`evidence`：当前代码把best的乘性姿态耦合改为独立forward_reward+阈值posture_penalty后，score从2926降至1647，ep_len从1000降至656，且posture_penalty的active_rate=0%（阈值0.3从未触发），说明独立阈值惩罚在此任务中完全失效、姿态约束实质缺失；best的乘性结构是唯一验证有效的方案。

`behavior_diagnosis`：策略在缺乏有效姿态约束的情况下，forward_reward完全独立于姿态，导致agent可以牺牲姿态换取速度，但环境自身的终止条件（躯干倾角超限）会截断episode，造成8/20提前终止、平均ep_len仅656步、score远低于best。

`signal_completeness`：当前奖励缺少可工作的姿态约束——posture_penalty形态为阈值触发但阈值过高从未激活，形式上存在但实质上不可达；best的乘性耦合已验证可达且有效。

`selected_level`：Level 2 — 当前posture约束的数学结构（独立加性阈值惩罚）已被本轮证据直接否定（active_rate=0%），必须变换为乘性耦合形态；同时best的k=10虽有效但留下约874分gap至target，需做有证据的软化。

`selected_intervention`：唯一修改 — 将posture_penalty的独立加性阈值结构变换回乘性姿态因子，并将指数衰减系数从best的k=10软化为k=4.0，形成`posture_factor = e^(-4.0 * angle²)`；保留best的forward_vel * posture_factor主结构和ang_vel_penalty不变。

`falsifiable_hypothesis`：乘性耦合迫使agent在优化速度时必须同时优化姿态，消除"牺牲姿态刷速度后被环境终止"的漏洞；k=4比k=10更允许跳跃所需的短暂动态倾斜（在0.25rad时k=4给~0.78因子 vs k=10给~0.54），应提高有效前进速度而不引发频繁终止，从而缩小与target的差距。

`expected_next_round`：ep_len应回升至接近1000（terminated比例大幅下降），score应超过当前best的2926，forward_reward的episode_sum_mean因姿态因子软化而提升，ang_vel_penalty保持相近尺度。

`main_risk`：k=4可能软化过度，如果agent利用宽松的姿态容限发展出边界策略（长时间处于~0.35rad倾斜），可能触发环境终止而拉低score；若发生则下一轮需将k回调至6-7范围或加入门控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    torso_angle = next_obs[1]
    forward_vel = next_obs[5]
    torso_ang_vel = next_obs[7]

    # 乘性姿态因子：躯干越直立因子越接近1，软化k从10→4以允许跳跃所需的动态倾斜
    k_angle = 4.0
    posture_factor = 2.718281828 ** (-k_angle * (torso_angle ** 2))

    # 主学习信号：前进速度 × 姿态因子（乘性耦合，迫使同时优化速度与姿态）
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    ang_vel_penalty = 0.05 * abs(torso_ang_vel)

    total_reward = speed_posture_reward - ang_vel_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': -ang_vel_penalty
    }

    return float(total_reward), components
```
