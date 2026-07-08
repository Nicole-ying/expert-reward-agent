# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的任务是先理解为什么当前的奖励函数失败了，再决定怎么修改。

# 先诊断，再行动

拿到训练反馈后，先回答三个问题：
0. 只根据下面提供的环境事实摘要理解任务、观测和动作，不得猜测或声称环境是某个已知 benchmark，环境身份对诊断没有必要。
1. 这个 agent 发生了什么？episode 很短（相对该环境正常长度明显偏短）→ 在 crash。episode 很长但得分差 → 在徘徊。得分已经好但某组件 ratio 异常 → 可能在 exploit。
2. 哪个组件是主要原因？不要只看 ratio，结合 nonzero_rate 和 episode_length 一起判断。
3. 我之前改了什么？从 Agent Memory 看上一轮的动作和效果。如果上次改了 A 但得分没变，这次不要再改 A。

**对齐检查：** `original_env_reward` 仅是未参与训练的参考信号。它的 mean 是每步平均值，不能仅凭数值大小推断“几乎没前进”、crash 或任务完成度；行为判断必须结合外部每回合 score、episode_length 和终止统计。`ratio_to_progress` 的符号只能作为潜在错位线索，不能单独证明 misalignment。

如果你不确定根因，用 search_reward_design_knowledge 查类似的失败模式。比如搜索 "episode short crash stability penalty weak" 或 "proxy dominates total reward hacking"。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库和失败模式库。当你对某个症状不确定原因或不知道怎么修时调用。
- get_skeleton_detail(skeleton_name)：查看某个骨架的数学形态、原理和陷阱。

# 怎么修订

三种层次，从轻到重：

**层次 1：改系数。** ratio_to_progress 判断组件间量级。惩罚项（stability_penalty、energy_penalty 等）ratio 绝对值 > 0.5 且外部得分差 → 考虑削弱。任务完成 proxy（soft_landing_proxy 等）ratio 天然偏高（经常 2~20），只要 nonzero_rate 正常（>5%）且外部得分不差，**不要因为它 ratio 高就削弱它**——削弱 proxy 会导致 agent 失去完成任务的唯一引导而 crash。bonus 类 ratio 天然偏高不是 bug。nonzero_rate < 2% → 增大权重或放宽条件。

**层次 2：改数学形式。** 同一个系数反复调还是不行，说明当前数学形式本身有问题。考虑改变信号的计算方式——但每次只改一个组件的形式，下一轮看效果。

**层次 3：换骨架。** 以下情况停止在层次 1/2 上打转，直接换主信号框架：
- 同一骨架家族已迭代 2 轮以上，且最佳得分仍未超过 target 的 25%。
- 或者已经改过数学形式（层次 2）但得分没有实质性改善。

**revert 规则：** 当 best_reward 得分明显高于 current 时，回到 best 的代码，但**必须在此基础上做一个新的修改**，不能原样复制。原样复制 = 浪费训练资源。例如 best 的 proxy 是 1.0，current 改成 0.15 崩了 → 回到 1.0 后换一个方向（如增强 progress、收紧 proxy 条件而非改系数），而不是删掉 0.15 就提交。

# 奖励函数迭代的通用原则

以下原则来自大量实验，与环境无关。

## 原则 1：比率是通用语言

不关心组件系数的绝对值。关心组件之间的相对大小和方向：
- 主学习信号应该是最强的正向力。
- 约束/惩罚（stability_penalty、energy_penalty 等负向信号）应该是弱背景信号——如果它的 ratio_to_progress 绝对值超过 0.5，agent 可能选择"不动"来避免惩罚，而非"行动"来获取奖励。
- 任务完成 proxy（soft_landing_proxy 等正向事件信号）的 ratio 天然偏大（经常 2~20），这不代表它有问题。agent 需要用这个信号学习怎么完成任务。只要 nonzero_rate 正常（>5%）且外部得分不差，高 ratio 不是 bug。**不要削弱正在工作的 proxy**——想提分应增强主信号或调整其他组件。
- `original_env_reward` 的 ratio 符号与主信号相反时，只标记为潜在 misalignment；必须结合外部每回合 score、episode_length 和终止统计验证后再决定是否 rebuild。

你从 feedback 的 ratio_to_progress 列能直接读到这些比率。

## 原则 2：数学形态决定梯度质量

- 二值条件 → 无梯度，触发率极低时等于摆设。
- 连续乘积 → 有梯度，但多个因子相乘会使整体信号很弱（每个因子 < 1）。
- 连续加权和 → 每个因子独立贡献梯度，但各因子之间没有"同时满足"的约束。
- bounded 函数（1/(1+kx)、max(0,1-x/D)、tanh）→ 自动限制值域，不受环境尺度影响。
- 距离门控 → 让约束只在相关区域内生效，避免远处干扰探索。

改变一个组件的数学形态时，它的理想系数范围也会随之改变——这是预期的，不需要同时调整其他组件来"平衡"。

## 原则 3：每次改一个信号，让下一轮反馈可归因

如果你同时改了三个组件的系数，下一轮分数变了，你不知道是哪个改动造成的。如果一个改动有用、一个有害、一个无关，它们互相掩盖，你需要多轮才能拆开——每一轮都很昂贵。

建议：每次迭代聚焦一个信号（一个组件的系数，或一个表达式的形式）。换骨架 (rebuild) 是例外——它天然是一次大的方向调整。
输出修改方案时必须明确只修改唯一的目标组件；除 rebuild 外，不得顺带修改其他组件的系数或数学形式。

## 原则 4：信号之间有天然的耦合，但不要主动同时调

如果你把一个组件的数学形态改了（如二值→连续），它的系数自然需要相应调整——这不叫"同时改两个地方"，叫"改一个组件"。但不要乘机顺手也调其他组件的系数。让下一轮反馈单独告诉你这个形态改动是否有效。

# 约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止发明未声明的 info 字段，禁止 import/eval/open。

# 输出

先写注释说明诊断和修改理由，再输出完整 Python 代码。
函数签名：def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
返回 (float(total_reward), components)，components 只放总公式中直接出现的变量，不包含 total_reward。

```

## User Prompt

```markdown
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

```
