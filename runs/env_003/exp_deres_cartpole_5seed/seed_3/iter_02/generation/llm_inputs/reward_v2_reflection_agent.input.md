# ⚠️ 上一版代码验证失败
错误信息：Reward v2 failed validation: runs\env_003\exp_deres_cartpole_5seed\seed_3\iter_02\generation\validations\reward_v2.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 500.000000）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v1。
    使用连续负惩罚引导摆杆保持竖直并使底座接近轨道中心。
    """
    # 提取下一步观察
    base_pos = next_obs[0]       # 底座水平位置
    pole_angle = next_obs[2]     # 杆偏角 (rad)
    pole_angular_vel = next_obs[3]  # 杆角速度

    # 惩罚系数
    w_angle = 5.0
    w_angular_vel = 0.5
    w_pos = 0.1

    # 核心学习信号：偏离直立和中心的二次惩罚
    progress_reward = -(
        w_angle * (pole_angle ** 2) +
        w_angular_vel * (pole_angular_vel ** 2) +
        w_pos * (base_pos ** 2)
    )

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=500.000000, len=500.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | -0.058963 | 0.058963 | 1.000000 | -1.000000 |
| total_reward | -0.058963 | 0.058963 | 1.000000 | -1.000000 |
| generated_reward | -0.058963 | 0.058963 | 1.000000 | -1.000000 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 16.959865 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 16.959865 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=500.000000, min=500.000000, max=500.000000
- episode_length: mean=500.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
在一个水平的直线轨道上，通过向移动底座施加固定大小的水平力（离散两个方向），控制底座移动，使底座顶端通过无动力关节连接的摆杆尽可能长时间地保持直立（即杆偏离竖直的角度不超过阈值），同时底座本身不能超出轨道边界。**本质是一个生存平衡任务：最大化存活步数（或时间）。**

## 2. 任务类型选择
selected_route_id: survival_balance_task
confidence: high
reason: 任务要求持续维持系统在不稳定平衡点附近，没有预设目标位置或导航点，终止仅由平衡状态破坏或轨道越界触发，奖励稀疏且与存活时长正相关，属于典型的生存/平衡任务。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (4,)
- dtype: float32
- 各维度含义（按索引）：
  - obs[0]: 底座水平位置 (base_position)，取值范围约 [-4.8, 4.8]，但有效安全范围实际为 [-2.4, 2.4]，超出则终止。
  - obs[1]: 底座水平速度 (base_velocity)，无界。
  - obs[2]: 杆相对于竖直方向的偏角 (pole_angle)，单位弧度，范围约 [-0.4189, 0.4189]；终止阈值为 [-0.20944, 0.20944]。
  - obs[3]: 杆角速度 (pole_angular_velocity)，无界。

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 2
- 动作含义：
  - action 0: 向轨道负方向施加固定大小的水平力 (push_negative_direction)
  - action 1: 向轨道正方向施加固定大小的水平力 (push_positive_direction)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无**。环境不定义“成功”概念，存活得越久越好。
- failure-like termination:
  - 底座位置绝对值超过 2.4（即底座掉出轨道有效区域）。
  - 杆偏转角绝对值超过 0.20943951 弧度（约 12°）。
  - 以上任一发生均视为**平衡失败**，episode 终止。
- ambiguous termination: 无。
- truncation: episode 达到 500 步后强制截断，属于时间截断，不代表任务失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: info 为空字典 `{}`，无任何额外字段。
- forbidden_or_uncertain_info_fields: 所有在 info 中未声明的字段（例如 success、failure、termination_reason、reward 等）均不存在，禁止使用。

## 7. 可用于奖励函数的信号
- **底座位置**：`obs[0]` / `next_obs[0]`，可用于惩罚远离中心。
- **底座速度**：`obs[1]` / `next_obs[1]`，可用于平滑控制。
- **杆偏角**：`obs[2]` / `next_obs[2]`，核心平衡信号。
- **杆角速度**：`obs[3]` / `next_obs[3]`，可用于倾向稳定性。
- **动作**：`action`，可用于鼓励能量最小或对称控制。

## 8. 不确定或不可用的信号
- **官方原始奖励**：被显式隐藏（masked），不可用。
- **显式成功/失败标志**：不存在。
- **存活步数/时间**：未直接提供在观察或 info 中，无法直接获取。
- **底座所处区域精细状态**（如是否接近边界）：只能通过底座位置计算，但位置是可用信号。
- **任何 info 中的额外奖励或终止原因**：info 为空，不存在。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward | 500.00 | 500.00 | 0.00 | 500.00 | progress_reward=-0.059 | target_solved_new_best |
