# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的任务是先理解为什么当前的奖励函数失败了，再决定怎么修改。

# 先诊断，再行动

拿到训练反馈后，先回答三个问题：
0. 只根据下面提供的环境事实摘要理解任务、观测和动作，不得猜测或声称环境是某个已知 benchmark，环境身份对诊断没有必要。
1. 这个 agent 发生了什么？episode 很短（相对该环境正常长度明显偏短）→ 可能在快速失败。episode 很长但得分差 → 可能在徘徊。得分已经好但某组件尺度异常 → 可能存在 exploit，需要行为证据验证。
2. 哪个组件最值得干预？结合组件数学形态、episode_sum_mean、signed_share、magnitude_share、active_rate、外部 score 和 episode_length 判断。不得把数值占比直接表述为因果贡献。
3. 我之前改了什么？从 Agent Memory 看上一轮的动作和效果。如果上次改了 A 但得分没变，这次不要再改 A。

**证据边界：** feedback 的组件表来自训练完成后的固定策略评估，并与外部 score 使用同一批轨迹。组件份额描述观察到的奖励构成，不是因果贡献；行为判断必须结合 score、episode_length 和终止统计。

**开放设计空间：** expert_reward_context 和工具返回的骨架是奖励设计原语与风险提示，不是允许列表。当已有骨架无法解释或修复失败时，可以根据环境事实组合、变形或构造新的引导信号，不要求新组件对应已有 skeleton_id。

如果你不确定根因，用 search_reward_design_knowledge 查类似的失败模式。比如搜索 "episode short crash stability penalty weak" 或 "proxy dominates total reward hacking"。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库和失败模式库。当你对某个症状不确定原因或不知道怎么修时调用。
- get_skeleton_detail(skeleton_name)：查看某个骨架的数学形态、原理和陷阱。

# 怎么修订

三种层次，从轻到重：

**层次 1：改系数。** signed_share、magnitude_share 和 active_rate 只能作为组件尺度、方向与触发频率的线索。只有当组件数学形式未变，并且外部行为证据支持“过强或过弱”时，才调整系数。差分信号、持续状态奖励和稀疏事件奖励不能套用统一占比阈值。

**层次 2：改数学形式。** 同一个系数反复调还是不行，说明当前数学形式本身有问题。考虑改变信号的计算方式——但每次只改一个组件的形式，下一轮看效果。

**层次 3：换骨架。** 以下情况停止在层次 1/2 上打转，直接换主信号框架：
- 同一骨架家族已迭代 2 轮以上，且最佳得分仍未超过 target 的 25%。
- 或者已经改过数学形式（层次 2）但得分没有实质性改善。
- 或者同一结构家族连续 3 轮没有刷新历史 best，且仍未达到 target；即使当前分数已超过 target 的 25%，也要警惕中等分数平台和局部最优。

**revert 规则：** 当 best_reward 得分明显高于 current 时，回到 best 的代码，但**必须在此基础上做一个新的修改**，不能原样复制。原样复制 = 浪费训练资源。例如 best 的 proxy 是 1.0，current 改成 0.15 崩了 → 回到 1.0 后换一个方向（如增强 progress、收紧 proxy 条件而非改系数），而不是删掉 0.15 就提交。

# 奖励函数迭代的通用原则

以下原则来自大量实验，与环境无关。

## 原则 1：组件份额是观察证据，不是因果结论

feedback 中的 `episode_sum_mean` 是最终策略每回合的平均有符号累计贡献；`magnitude_share` 是该组件绝对累计量占所有直接组件绝对累计量的比例；`signed_share` 保留净方向；`active_rate` 表示触发频率。使用时必须遵守：
- 不同数学形态具有不同时间语义。差分进度、持续状态奖励、惩罚和事件 bonus 的份额不可直接解释为“对策略的因果贡献”。
- 同一组件、同一数学形式跨轮比较时，累计贡献、份额和触发率最有可比性。
- 惩罚项 signed_share 为正可能提示角色与符号不一致；惩罚 magnitude_share 较大只表示值得检查，不能单独证明 agent 因此不敢行动。
- 高 magnitude_share 既可能表示主导，也可能是持续发放或任务本身需要。必须结合外部 score、episode_length、终止情况和上一轮修改结果验证。

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
错误信息：Reward v4 failed validation: runs\env_001\paper_lander_deres_main_v1\seed_2\iter_04\generation\validations\reward_v4.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -153.053812）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v3: 移除 landing_bonus 中的 contact_factor，让接近+低速的梯度在触地前就生效。

    诊断：v2 的 landing_bonus 有 contact_factor（二值），导致 nonzero_rate 仅 4.38%。
    95%+ 的 step 没有正向信号，agent 缺乏"减速接近"的引导，crash 率 70%。

    修改：landing_bonus = bonus_scale * dist_factor * speed_factor（去掉 contact_factor）。
    靠近目标垫且速度低即有奖励，不等腿触地。

    Components:
    1. progress_reward:   -distance_to_landing_pad（密集引导，不变）
    2. orientation_penalty: 小惩罚非零姿态角和角速度（不变）
    3. landing_bonus:       连续乘积 —— 距离近 × 速度低（不再要求触地）
    """
    # ---- unpack observations ----
    x_pos = obs[0]
    y_pos = obs[1]
    body_angle = obs[4]
    angular_vel = obs[5]

    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: continuous proxy — proximity × low speed (no contact gate) ----
    # 去掉了 contact_factor，让 agent 在接近+减速时就收到正向梯度
    dist_thresh = 0.5
    speed_thresh = 1.0
    bonus_scale = 10.0

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5

    dist_factor = max(0.0, 1.0 - n_distance / dist_thresh)
    speed_factor = max(0.0, 1.0 - n_speed / speed_thresh)

    landing_bonus = bonus_scale * dist_factor * speed_factor

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v2: 将 landing_bonus 从二值条件改为连续乘积形式，提供着陆梯度。

    Components:
    1. progress_reward:   -distance_to_landing_pad (密集引导)
    2. orientation_penalty: 小惩罚非零姿态角和角速度
    3. landing_bonus:       连续乘积 proxy —— 距离近 × 速度低 × 腿触地
    """
    # ---- unpack observations ----
    x_pos = obs[0]
    y_pos = obs[1]
    body_angle = obs[4]
    angular_vel = obs[5]

    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: continuous proxy for successful landing ----
    # 三个连续因子相乘，每个提供梯度，共同约束"接近 + 低速 + 触地"
    dist_thresh = 0.5      # 距离阈值（放宽，提前给梯度）
    speed_thresh = 1.0     # 速度阈值（放宽，提前给梯度）
    bonus_scale = 10.0     # 最大可能奖励（完美着陆时）

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5

    dist_factor = max(0.0, 1.0 - n_distance / dist_thresh)
    speed_factor = max(0.0, 1.0 - n_speed / speed_thresh)
    contact_factor = min(1.0, n_left_contact + n_right_contact)  # 0, 0→0; 1, 0→1; 1, 1→1

    landing_bonus = bonus_scale * dist_factor * speed_factor * contact_factor

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-153.053812, len=133.900000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_bonus | 1.212373 | 1.212373 | 0.253987 | 4.773368 | 1.548481 |
| orientation_penalty | -0.018956 | 0.018956 | 1.000000 | -0.018956 | -0.024211 |
| progress_reward | -0.782943 | 0.782943 | 1.000000 | -0.782943 | -1.000000 |
| total_reward | 0.410474 | 1.909171 | 1.000000 | 0.410474 | 0.524270 |
| original_env_reward | -1.591747 | 3.602428 | 1.000000 | -1.591747 | -2.033030 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 119.283934 | 119.283934 | 0.000000 | 8001.381261 | 10190 |
| orientation_penalty | -1.865463 | 1.865463 | -18.949881 | -0.048128 | 10190 |
| progress_reward | -77.057154 | 77.057154 | -574.683072 | -41.965808 | 10190 |
| total_reward | 40.361317 | 161.360132 | -460.289887 | 7905.675253 | 10190 |

## Distribution
- score: mean=-153.053812, min=-249.691188, max=-32.767124
- episode_length: mean=133.900000
- early_terminal (<150 steps + score<-50): 17/20 (85%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个 2D 飞行器从画面顶部中央出发，受随机初始力影响。核心目标是让飞行器尽快到达画面中央的目标着陆垫并稳定停稳，同时尽量减少引擎推力使用。智能体需要学习靠近目标、减速、保持姿态稳定并实现安全着陆。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: 飞行器相对于目标垫中心的水平坐标 (x_position)
- obs[1]: 飞行器相对于目标垫高度的垂直坐标 (y_position)
- obs[2]: 水平线速度 (x_velocity)
- obs[3]: 垂直线速度 (y_velocity)
- obs[4]: 机身姿态角 (body_angle)
- obs[5]: 角速度 (angular_velocity)
- obs[6]: 左支撑腿接触标志，接触为 1.0，否则为 0.0 (left_support_contact)
- obs[7]: 右支撑腿接触标志，接触为 1.0，否则为 0.0 (right_support_contact)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: 无引擎推力 (no_engine)
- action 1: 点燃左侧姿态引擎 (left_orientation_engine)
- action 2: 点燃主引擎 (main_engine)
- action 3: 点燃右侧姿态引擎 (right_orientation_engine)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（飞行器停止活动并稳定，可能表示成功着陆并停稳）
- failure-like termination: crash_or_body_contact（撞击或非正常身体接触），horizontal_position_outside_viewport（水平出界）
- ambiguous termination: body_not_awake_or_settled 在未接触目标垫或错误位置时可能不表示真正成功
- truncation: 未明确提及，假设无时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，info为空字典）
- forbidden_or_uncertain_info_fields: info（空字典，不可提供任何额外信号）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])，可用于衡量与目标垫的距离
- velocity: x_velocity (obs[2]), y_velocity (obs[3])，可用于衡量降落时的速度大小
- orientation: body_angle (obs[4]), angular_velocity (obs[5])，可用于姿态稳定
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])，可用于判断着陆
- action/engine: 动作选择本身（主引擎或姿态引擎），可用于惩罚推力使用

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + orientation_penalty + progress_reward | -161.08 | -161.08 | 0.00 | 77.85 | landing_bonus=0.254 orientation_penalty=-0.020 progress_reward=-0.945 | new_best |
| 2 | landing_bonus + orientation_penalty + progress_reward | -120.87 | -120.87 | 0.00 | 77.85 | landing_bonus=0.183 orientation_penalty=-0.020 progress_reward=-0.935 | new_best |
| 3 | landing_bonus + orientation_penalty + progress_reward | -153.05 | -120.87 | -32.18 | 133.90 | landing_bonus=1.212 orientation_penalty=-0.019 progress_reward=-0.783 | no_meaningful_improvement |

```
