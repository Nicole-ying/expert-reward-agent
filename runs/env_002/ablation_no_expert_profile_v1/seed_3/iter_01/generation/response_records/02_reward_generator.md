# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 主学习信号：前进速度 ----------
    # horizontal_velocity 在 obs[2]，正值表示向前，负值表示向后
    # 使用 bounded 形式抑制极端速度，同时保持符号信息
    forward_velocity = obs[2]
    bounded_forward = forward_velocity / (1.0 + abs(forward_velocity))
    forward_reward = 1.0 * bounded_forward   # 权重 1.0

    # ---------- 稳定/安全约束：躯干直立惩罚 ----------
    # hull_angle 在 obs[0]，理想值为 0（竖直）
    hull_angle = obs[0]
    upright_penalty = -0.5 * abs(hull_angle)   # 权重 0.5

    # ---------- 健康约束：抑制双脚同时离地 ----------
    #  leg1_contact: obs[8],  leg2_contact: obs[13]
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    # 当双脚都未触地时，(1-leg1_contact)*(1-leg2_contact) = 1
    double_flight_penalty = -0.2 * ((1.0 - leg1_contact) * (1.0 - leg2_contact))   # 权重 0.2

    # ---------- 合计 ----------
    total_reward = forward_reward + upright_penalty + double_flight_penalty

    components = {
        "forward_velocity_reward": float(forward_reward),
        "upright_penalty": float(upright_penalty),
        "double_flight_penalty": float(double_flight_penalty)
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与动态类型
- **task_family:** locomotion_continuous_control
- **dynamics_subtype:** planar_bipedal_gait
- **核心目标:** 在起伏地形上尽可能远、尽可能快地向前行走，同时维持躯干直立，避免倾倒。

## 选用的奖励角色（reward roles）
根据 environment_card 中隐含的核心与次要目标，v1 选择了 3 个组件：
1. **主学习信号：前向速度奖励**  
   - 角色：鼓励持续向前运动，与“远且快”直接对应，提供稠密梯度。
2. **稳定约束：躯干直立惩罚**  
   - 角色：防止身体倾倒，保障生存前提。
3. **健康约束：双脚离地惩罚**  
   - 角色：抑制双脚同时离地的跳跃式行为，诱导更稳定的交替支撑步态。

## 角色-信号映射（role_to_signal_mapping）
| 角色                     | 使用信号                | 来源索引 | 算子类型            |
|--------------------------|-------------------------|----------|---------------------|
| 前向速度奖励             | horizontal_velocity     | obs[2]   | bounded_signal       |
| 躯干直立惩罚             | hull_angle              | obs[0]   | absolute_penalty     |
| 双脚离地惩罚             | leg1_contact, leg2_contact | obs[8], obs[13] | product_trigger_penalty |

- `forward_velocity` 通过 `signal / (1 + |signal|)` 映射为有界奖励，避免无限制速度涨落主导总奖励。
- `hull_angle` 使用绝对值惩罚，因为角度离零越远惩罚越大，且线性惩罚在原点附近梯度稳定。
- `double_flight_penalty` 只在双脚同时未触地时触发，重量轻，防止 agent 利用跳跃刷速度分数。

## 排除的角色与原因
- **terminal_success_reward**：没有显式 success flag（`explicit_success_flag_available=false`），且 info 为空，无法获得 `reached_end_of_terrain` 信息。
- **terminal_failure_penalty**：同样没有显式 failure flag（`explicit_failure_flag_available=false`），无法获得 `body_fallen_over` 信号。
- **能耗惩罚（动作代价）**：根据 component budget，v1 暂不引入动作代价，优先让 agent 学会前进与稳定。能耗优化留到后续迭代。
- **基于 LIDAR 的地形奖励**：LIDAR 属于感知信号，不宜直接作为奖励优化目标，避免策略利用特定地形读数。
- **距离/位移奖励**：观测中无全局位置，无法奖励绝对前进距离，只能通过速度近似。

## 后续迭代预留
- 能量消耗项（如 `action` 平方和惩罚）将在 agent 掌握基本行走后加入，平衡速度与效率。
- 步态韵律与关节协调（如对称性、抬脚高度）可在引入更多运动学信息后考虑。
- 可根据 LIDAR 信号构造地形适应性引导，但需谨慎设计以防止 reward hacking。

## 训练后需要观察的failure modes
- **velocity-burst-then-fall**：前向速度奖励可能导致 agent 短暂冲刺后倾倒，应注意躯干直立惩罚是否足够强。
- **stand-still / hover**：如果 upright_penalty 过重，可能使 agent 选择原地站立不动。需检查前向速度奖励是否仍有足够驱动力。
- **contact reward hacking**：双脚离地惩罚过轻时，agent 可能学会跳跃式前进；过重可能抑制正常跑步步态中的腾空相，需根据实际步行类型微调权重。
- **unstable oscillation**：因缺少对角速度或关节动作平滑的约束，可能出现高频抖动，可在 v2 中加入动作平滑项。
