# Response Record

`search_reward_design_knowledge` 返回了 locomotion 能量惩罚主导和 dense proxy 中等分平台的处理经验，但当前 agent 不属于这些失败模式——它已完成全部地形（terminated=20/20），只是能量惩罚是主要扣分来源。`get_reward_transformation` 确认了稠密 proxy 相关的变换，但 agent 并未停滞在平台上。

---

**evidence**：score=311.12，全部20回合terminated（走完地形），len=1051.8；forward_reward 主导(504.38, 87.4%)，energy_penalty 是唯一显著负项(-64.11, 11.1%)，angle与vert惩罚近乎为零，说明姿态已很稳定。

**behavior_diagnosis**：agent 已学会稳定行走并完成地形，但 energy_penalty 使用 sum(action²) 惩罚所有力矩，无论关节是否运动——这无法区分"推蹬发力"和"等长维持"，属于能量评估的粗糙代理。

**signal_completeness**：前进速度、姿态稳定、垂直约束职责均已具备且 active_rate=100%，但能量效率信号使用了物理上不精确的 action² 形式，缺失了对"力矩是否在关节运动中做功"的区分能力。

**selected_level**：Level 2——agent 没有失败或尺度异常，但 energy_penalty 的数学形态（action² 替代机械功率）在双足行走中无法引导高效步态时序。改变该组件的数学结构，同步设置匹配新值域的系数。

**selected_intervention**：将 energy_penalty 从 `sum(action²)` 改为 `sum(|action[i] × joint_velocity[i]|)`，使用 obs[5]/obs[7]/obs[10]/obs[12] 作为各关节角速度，形成真实的机械功率惩罚。系数从 0.05 微调至 0.08 以匹配新值域。

**falsifiable_hypothesis**：agent 应学会将力矩集中在关节运动阶段（推蹬），在被动摆动阶段放松，从而在维持或提高前进速度的同时降低机械功率消耗，energy_penalty 的 magnitude_share 应下降、forward_reward 至少不降、score 应超过 311.12。

**expected_next_round**：energy_penalty 的 episode_sum_mean 绝对值应下降，其在 magnitude_share 中的占比减小；forward_reward 保持或略增；总 score 提升；terminated 保持 20/20。

**main_risk**：绝对值的机械功率惩罚同时抑制加速和制动，可能干扰摆动腿末端的自然减速；若关节速度量级很小，惩罚可能塌缩为无约束，此时需在下一轮调整系数或改用 max(0, a×v) 只惩罚正功。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 核心进步信号：鼓励沿前进方向快速移动
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # 姿态稳定约束：惩罚躯干倾角偏离直立
    hull_angle = next_obs[0]
    w_angle = 1.0
    angle_penalty = -w_angle * (hull_angle ** 2)

    # 垂直稳定约束：抑制跳跃或剧烈起伏
    vertical_velocity = next_obs[3]
    w_vert = 0.5
    vert_penalty = -w_vert * (vertical_velocity ** 2)

    # 能耗约束：机械功率惩罚，力矩×关节角速度的绝对值之和
    # 更准确地反映实际能量消耗，区分推蹬发力与等长维持
    joint_velocities = [obs[5], obs[7], obs[10], obs[12]]  # hip1, knee1, hip2, knee2 角速度
    mechanical_power = sum(abs(action[i] * joint_velocities[i]) for i in range(4))
    w_energy = 0.08
    energy_penalty = -w_energy * mechanical_power

    total_reward = forward_reward + angle_penalty + vert_penalty + energy_penalty

    components = {
        'forward_reward': forward_reward,
        'angle_penalty': angle_penalty,
        'vert_penalty': vert_penalty,
        'energy_penalty': energy_penalty
    }

    return float(total_reward), components
```
