# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    forward_velocity = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_velocity

    # 稳定性约束：惩罚身体倾角、角速度和剧烈垂直振荡
    hull_angle = next_obs[0]
    hull_angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    w_angle = 0.5
    w_angvel = 0.1
    w_vvel = 0.2
    stability_cost = (
        -w_angle * abs(hull_angle)
        - w_angvel * abs(hull_angular_vel)
        - w_vvel * abs(vertical_vel)
    )

    total_reward = forward_reward + stability_cost

    components = {
        "forward_reward": forward_reward,
        "stability_cost": stability_cost,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色

1. **forward_reward（主学习信号）**  
   直接奖励水平速度 `next_obs[2]`，与任务“尽可能远、尽可能快地向前行走”一致。该信号每步都有稠密梯度，无需稀疏事件，是 locomotion 类任务的标准核心驱动。

2. **stability_cost（稳定/安全约束）**  
   对三项影响平衡与平稳性的状态变量施加线性惩罚：  
   - 身体倾角 `hull_angle`（大幅倾斜可能预示摔倒）  
   - 身体角速度 `hull_angular_velocity`（剧烈摇晃）  
   - 垂直速度 `vertical_velocity`（过大的上下振荡，如跳跃或高频起伏）  
   这些惩罚项轻柔地引导智能体维持稳定姿态，防止因追逐速度而摔倒。

三个权重经过量级平衡，使稳定惩罚不会压制前进奖励，但仍能提供有意义的梯度。

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因

- **explicit_success_flag_available=false**：环境没有提供成功到达地形尽头的显式标志（info 为空），无法在 reward 中为成功完成给予奖励。
- **explicit_failure_flag_available=false**：环境同样缺少显式失败标志。即使回合因摔倒而终止，`body_fallen_over` 只在环境终止时生效，不会作为 reward 的输入。因此终端失败惩罚无法在 v1 中安全实现。
- 摔倒天然导致 episode 终止（从而结束奖励累积），本身已构成隐式负反馈，v1 不再额外增加硬性惩罚。

## 哪些组件留到后续迭代

- **能量/动作代价（energy_penalty）**  
  任务要求“尽量减小能量消耗”，但 v1 阶段智能体应首先学会稳定行走，暂不加入力矩惩罚，以免 agent 怯于发力。
- **动作平滑性（action_smoothness_penalty）**  
  需要前一步动作历史，但当前接口未提供，需通过环境扩展或自定义 wrapper 后再引入。
- **基于 lidar 的潜能奖励**  
  当前版本未使用激光雷达信息，后续可以利用地形前瞻构造前进潜能，进一步增强快速通过崎岖地形的能力。
- **terminal_success_reward / terminal_failure_penalty**  
  待环境能够通过 info 提供明确的成功/失败标志后（或通过终止原因解析），再考虑加入。
- **更复杂的步态引导（如接触模式奖励）**  
  leg1_contact/leg2_contact 可用于鼓励交替着地，但过早引入可能引入 exploit（如拖着脚走），可以在稳定行走已经学会后作为精细化调整手段。

## 训练后应观察的 failure mode

- **快速但易摔倒**：forward_reward 权重过大可能诱导过于激进的迈步，导致摔倒率升高。可考虑增大稳定性惩罚或降低前进权重。
- **原地站立或极小步幅**：若稳定性惩罚过强，智能体可能选择保持静止来避免 cost，此时水平速度接近零，无法完成任务。需适当放松稳定性约束或提高前进奖励。
- **高频振动/抽搐步态**：若关节动作幅度大而不平滑，虽能前进但能量效率差，且不易收敛。这类问题应在后续迭代中通过动作平滑或能量惩罚处理，v1 阶段可容忍一定程度的粗糙步态。
- **过度跳跃式前进**：垂直速度和角速度惩罚不足时，可能出现“蹦跳前进”策略，虽速度较快但不符合行走形态。调大 w_vvel 或加入离地时间约束能缓解。
