# Response Record

`evidence`：当前得分2846（best=2926），16/20截断存活（len≈964），speed_posture_reward占magnitude_share 99.3%，ang_vel_penalty仅0.7%且活跃率100%；上次将k从10→4以期允许跳跃动态倾斜，但得分未升反微降80分，说明软化姿态约束未改善行为，返回k=10附近是合理方向但仅复现best不允许。

`behavior_diagnosis`：agent已学会稳定存活并持续前进，但前进速度与姿态的乘性耦合可能造成梯度混淆——躯干倾斜时前进速度反馈被同时压缩，agent无法清晰区分"提速"与"纠姿"的独立梯度，导致停留在中等速度的局部最优。

`signal_completeness`：当前信号有前进引导、姿态约束和角速度稳定，职责基本完备但耦合形态可能限制了优化效率；无能耗或跳跃质量信号，但无证据表明缺失这些会阻止突破当前平台。

`selected_level`：Level 2——乘性耦合 `forward_vel × e^(-k·angle²)` 中两个独立目标（速度与姿态）被强制捆绑，构成`coupled_to_diagnostic_components`，证据是k=10→4仅改变耦合松紧未改变耦合本身，且两轮均未突破2926平台，需解耦为独立可诊断组件。

`selected_intervention`：将乘性速度-姿态耦合拆分为加性独立组件——**forward_reward = forward_vel**（无界前向速度奖励）与 **posture_penalty = 10.0 × torso_angle²**（二次姿态惩罚），保持ang_vel_penalty不变。

`falsifiable_hypothesis`：解耦后agent能独立感知前进速度梯度（不再被姿态因子调制），同时姿态惩罚单独施加约束，梯度信号更清晰，应推动agent探索更高的前向速度策略，从而突破2926平台向3800靠近。

`expected_next_round`：speed_posture_reward消失，forward_reward应成为主正分量（episode_sum_mean明显>1850），posture_penalty均值应在合理范围（预计-50到-200区间），score应刷新best超过2926；若score未刷新或posture_penalty过大导致net reward塌缩，说明解耦后的尺度需要调整。

`main_risk`：加性结构下agent可能通过牺牲姿态换取极端速度（如大幅前倾获得高forward_vel，posture_penalty不足以遏制），若posture_penalty active_rate不变但episode_sum_mean异常大且terminated增多，则需上调k或重引入门控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    torso_angle = next_obs[1]
    forward_vel = next_obs[5]
    torso_ang_vel = next_obs[7]

    # 前进速度奖励：无界，直接鼓励更快的前向移动
    forward_reward = forward_vel

    # 姿态惩罚：二次型，躯干越倾斜惩罚越重
    # k=10，在0.2rad时惩罚0.4（约为典型forward_vel~2.0的20%）
    k_angle = 10.0
    posture_penalty = k_angle * (torso_angle ** 2)

    # 轻量稳定约束：抑制躯干角速度，防止快速晃动
    ang_vel_penalty = 0.05 * abs(torso_ang_vel)

    total_reward = forward_reward - posture_penalty - ang_vel_penalty

    components = {
        'forward_reward': forward_reward,
        'posture_penalty': -posture_penalty,
        'ang_vel_penalty': -ang_vel_penalty
    }

    return float(total_reward), components
```
