# Response Record

**evidence**: 当前score=171，上一轮从-94大幅提升。17/20 terminated（成功着陆），但episode长度526步偏长。landing_proxy占98.6%份额（episode_sum_mean=212.4），progress_reward仅0.6%（1.21）——两个信号尺度差距超175倍，progress在实际优化中几乎无效。

**behavior_diagnosis**: agent已学会着陆（85%成功率），但由于landing_proxy过于主导，agent缺乏快速接近目标的激励，可能采取迂回或悬停策略累积proxy奖励，导致episode过长。

**signal_completeness**: 三个职责（进程引导、着陆条件、稳定性约束）数学形态均合理且可达。landing_proxy的乘积形式正确表达了联合满足；progress_reward提供距离改善信号；stability_penalty提供正则。缺失的不是职责而是相对尺度。

**selected_level**: Level 1。着陆proxy的职责、符号和乘积形态都合理——问题纯粹是系数过强导致progress被淹没。符合尺度修复条件。

**selected_intervention**: 唯一修改landing_proxy系数2.0→0.8，progress_reward系数1.0→4.0。stability_penalty保持不变。预期将landing:progress的每步贡献比从~175:1降到~35:1，使progress成为可感知的引导信号同时保持landing的主导地位。

**falsifiable_hypothesis**: progress_reward的magnitude_share应从0.6%上升到3-8%，episode_length应下降（agent更快接近平台），同时terminated比率应保持≥15/20。若episode_length不降反升或terminated比率下降，则说明尺度修复不足以改变行为。

**expected_next_round**: episode_length预计降至300-450步，progress_reward的episode_sum_mean预计升至3-8，landing_proxy的signed_share预计降至85-95%，score可能因积累步数减少而绝对值下降但外部着陆成功率应保持或改善。

**main_risk**: landing_proxy系数降低可能削弱最终着陆阶段的精细控制信号，导致接近平台后crash增加。0.8的选择在保持~35:1主导比的同时测试progress的引导效果，若terminated下降应回退或转Level 2。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 距离进展：每一步靠近目标的欧氏距离变化
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = 4.0 * (dist_obs - dist_next)

    # 稳定性约束：抑制过度速度和姿态变化
    stability_penalty = (
        -0.01 * (abs(next_obs[2]) + abs(next_obs[3]))
        - 0.01 * abs(next_obs[4])
        - 0.01 * abs(next_obs[5])
    )

    # 稠密着陆代理信号：连续评估各着陆条件的满足度
    D_x = 2.0
    D_y = 2.0
    D_v = 1.0
    D_angle = 0.5

    x_sat = max(0.0, 1.0 - abs(next_obs[0]) / D_x)
    y_sat = max(0.0, 1.0 - abs(next_obs[1]) / D_y)
    vx_sat = max(0.0, 1.0 - abs(next_obs[2]) / D_v)
    vy_sat = max(0.0, 1.0 - abs(next_obs[3]) / D_v)
    angle_sat = max(0.0, 1.0 - abs(next_obs[4]) / D_angle)
    contact_factor = max(0.05, 0.5 * (next_obs[6] + next_obs[7]))

    landing_proxy = x_sat * y_sat * vx_sat * vy_sat * angle_sat * contact_factor
    landing_reward = 0.8 * landing_proxy

    total_reward = progress_reward + stability_penalty + landing_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_reward
    }

    return float(total_reward), components
```
