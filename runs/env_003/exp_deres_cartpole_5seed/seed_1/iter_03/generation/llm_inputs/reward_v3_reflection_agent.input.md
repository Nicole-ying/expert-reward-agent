# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: runs\env_003\exp_deres_cartpole_5seed\seed_1\iter_03\generation\validations\reward_v3.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 500.000000）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward.
    Uses:
    - progress_reward: dense improvement towards stable upright and center.
    - stability_penalty: light penalty on linear/angular speeds.
    
    Iter 2 change: reduced stability_penalty weights 10x (0.01→0.001)
    because penalty ratio_to_progress was -3.16, drowning out the progress signal.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    # Reduced 10x from 0.01 → 0.001 to fix penalty dominance (ratio was -3.16)
    w_vel = 0.001
    w_angvel = 0.001

    # -- cost helper --
    def cost(o):
        return w_pos * abs(o[0]) + w_angle * abs(o[2])

    # -- main progress reward --
    progress_reward = scale_progress * (cost(obs) - cost(next_obs))

    # -- light stability penalty (only on next step) --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty
    }

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
| progress_reward | -0.000371 | 0.002596 | 1.000000 | -0.370989 |
| stability_penalty | -0.000322 | 0.000322 | 1.000000 | -0.322026 |
| total_reward | -0.000693 | 0.002663 | 1.000000 | -0.693015 |
| generated_reward | -0.000693 | 0.002663 | 1.000000 | -0.693015 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 1000.000000 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 1000.000000 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=500.000000, min=500.000000, max=500.000000
- episode_length: mean=500.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
通过向移动底座施加固定大小的水平推力（负方向或正方向），使底座上的自由旋转杆保持竖直，同时底座自身不能越出轨道边界。目标是尽可能长时间地存活，即在底座位置超出 ±2.4 或杆倾角超出 ±0.2094 rad 前，持续控制。

## 2. 任务类型选择
selected_route_id: survival_balance_task
confidence: high
reason: 任务要求一直保持杆平衡和底座在轨道内，没有明确的到达目标，唯一的成功方式是尽可能长地避免终止，属于经典的生存‑平衡控制任务。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: [4]
- dtype: float32
- obs[0]: base_position — 移动底座的水平位置，范围 [-4.8, 4.8]（注意实际终止边界为 |base_position|>2.4）
- obs[1]: base_velocity — 底座水平速度，无界
- obs[2]: pole_angle — 杆相对于竖直方向的角度（rad），范围 [-0.41888, 0.41888]（终止边界为 |pole_angle|>0.20944）
- obs[3]: pole_angular_velocity — 杆的角速度，无界

## 4. 动作空间 action_space
- type: Discrete (2)
- action 0: push_negative_direction — 朝向轨道负方向施加固定推力
- action 1: push_positive_direction — 朝向轨道正方向施加固定推力

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: 无
- failure‑like termination:
  - |base_position| > 2.4 （底座越界）
  - |pole_angle| > 0.20943951 rad （杆倾倒）
  任一条件满足即终止，可视为失败。
- ambiguous termination: 无
- truncation:
  - 达到 500 步自动截断（可视为存活成功）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空，无可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 键均不可用

## 7. 可用于奖励函数的信号
- position: base_position (obs[0] / next_obs[0])
- velocity: base_velocity (obs[1] / next_obs[1])，pole_angular_velocity (obs[3] / next_obs[3])
- orientation: pole_angle (obs[2] / next_obs[2])
- action: 可获取当前选择的离散动作（0 或 1），用于动作平滑或惩罚高频切换等
- 隐式终止标记：通过检查 next_obs[0] 或 next_obs[2] 是否超出终止阈值（2.4 和 0.20944）来生成稀疏失败惩罚

## 8. 不确定或不可用的信号
- 原始环境奖励：已被遮蔽，不可用
- info 字典：为空，不包含任何额外信号（如 success、failure、终止原因等）
- 杆的绝对角度或位置之外的衍生状态：除非从已有观测中计算，否则不可用

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + stability_penalty | 466.15 | 466.15 | 0.00 | 466.15 | progress_reward=-0.001 stability_penalty=-0.004 | new_best |
| 2 | progress_reward + stability_penalty | 500.00 | 500.00 | 0.00 | 500.00 | progress_reward=-0.000 stability_penalty=-0.000 | target_solved_new_best |
