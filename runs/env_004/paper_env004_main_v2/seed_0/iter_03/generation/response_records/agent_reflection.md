# Response Record

`evidence`: 当前得分2607，best=2926（k_angle=10.0），当前(k_angle=5.0)反而更差；ang_vel_penalty仅占magnitude_share 1.2%，几乎不起作用；episode_length从1000降至977，姿态放松未改善表现；同一骨架已迭代2轮未刷新best。

`behavior_diagnosis`: 策略在姿态约束放松后表现下降，说明tight coupling（乘法耦合）对当前策略有效，但best仅2926远低于target 3800，agent可能陷入中等速度的稳态前移而非高效跳跃，乘法耦合导致必要的小幅倾斜也被等比惩罚。

`signal_completeness`: 现有信号职责基本完备（前进引导+姿态约束+稳定惩罚），但速度与姿态的乘法耦合可能导致agent为避免姿态惩罚而牺牲跳跃所需的小幅倾斜，使前进速度受限；缺失明确的高效跳跃激励，但缺乏接触/事件信号无法直接检测跳跃。

`selected_level`: Level 2 — 同一骨架家族已迭代2轮未刷新best，且第2轮尺度修复（k_angle 10→5）未改善反而恶化，触发从Level 1升级到结构变换；证据模式为乘积耦合可能限制任务所需的行为空间。

`selected_intervention`: 将`speed_posture_reward`从乘法耦合`forward_vel * exp(-k*angle²)`改为加法解耦`forward_vel - w_posture * angle²`，w_posture=5.0（使0.2rad倾角时惩罚约0.2，约为典型forward_vel的13%），ang_vel_penalty保持不变。

`falsifiable_hypothesis`: 加法解耦后agent可以在跳跃所需的小幅倾斜（~0.1-0.2rad）下获得接近全额的forward_vel奖励，同时大幅倾斜仍受惩罚，应能学习更高效的前向跳跃策略，提升forward_vel均值从而提升外部得分。

`expected_next_round`: episode_length应维持接近1000（存活良好），speed_posture_reward的episode_sum_mean应上升，外部score应向3000+逼近并有望刷新best。

`main_risk`: 完全解耦后agent可能学会大幅倾斜换取极高forward_vel（因为高vel下固定姿态惩罚占比变小），若出现此模式需引入forward_vel相关的自适应姿态惩罚或回到有界乘法形式。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键信号
    torso_angle = next_obs[1]          # 躯干倾角
    forward_vel = next_obs[5]          # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]        # 躯干角速度

    # 姿态惩罚：躯干倾角平方，引导直立但允许跳跃所需的小幅倾斜
    # w_posture=5.0: 0.1rad→罚0.05, 0.2rad→罚0.20, 0.3rad→罚0.45
    w_posture = 5.0
    posture_penalty = w_posture * (torso_angle ** 2)

    # 主学习信号：前进速度与姿态惩罚加法解耦
    #   - forward_vel 始终以原始尺度被奖励/惩罚，不再被姿态因子缩放
    #   - 姿态偏差独立惩罚，agent可接受小幅倾斜以换取更高前进速度
    speed_posture_reward = forward_vel - posture_penalty

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
